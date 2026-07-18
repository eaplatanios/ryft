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

/// Checks types against a structural or semantic type contract. All forms use an `@` selector and return
/// [`TypeError`](crate::TypeError)s as appropriate, converted into the enclosing function's error type,
/// when the selected contract is not satisfied. The available selectors are:
///
///   - `@same`: Requires the provided expected and actual flat type signatures to be identical.
///   - `@numeric`: Accepts only numeric [`DataType`](crate::DataType)s.
///   - `@floating_or_complex`: Accepts only floating-point and complex [`DataType`](crate::DataType)s.
///   - `@no_unreduced`: Rejects [`ArrayType`](crate::ArrayType)s carrying any unreduced mesh axes.
///   - `@same_unreduced_axes`: Requires exactly two [`ArrayType`](crate::ArrayType)s with matching unreduced-axis sets.
///   - `@same_reduced_axes`: Requires exactly two [`ArrayType`](crate::ArrayType)s with matching reduced-axis sets.
///
/// # Parameters
///
///   - `$selector`: Selector identifying the structural or semantic contract to validate.
///   - `$descriptor`: Expression evaluating to a string that identifies the checked operation or signature in errors.
///   - `$types`: Expression evaluating to the data or array types checked by `$selector`.
///   - `$signatures`: Bracketed pair containing the expected and actual flat type signatures checked by `@same`.
#[macro_export]
macro_rules! check_types {
    (@same, $descriptor:expr, [$expected:expr, $actual:expr $(,)?] $(,)?) => {{
        let expected = &$expected[..];
        let actual = &$actual[..];
        if expected != actual {
            return Err($crate::TypeError {
                message: format!(
                    "{} type signature mismatch: expected [{}] but got [{}]",
                    $descriptor,
                    expected.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
                    actual.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
                ),
            });
        }
    }};

    (@numeric, $descriptor:expr, $types:expr $(,)?) => {{
        let descriptor = $descriptor;
        let types = &$types[..];
        if let Some(input_type) = types.iter().find(|input_type| {
            matches!(input_type, $crate::DataType::Token | $crate::DataType::Zero | $crate::DataType::Boolean)
        }) {
            return Err($crate::TypeError {
                message: format!("'{descriptor}' does not support input data type {input_type}"),
            }
            .into());
        }
    }};

    (@floating_or_complex, $descriptor:expr, $types:expr $(,)?) => {{
        let descriptor = $descriptor;
        let types = &$types[..];
        if let Some(input_type) = types.iter().find(|input_type| {
            !matches!(
                input_type,
                $crate::DataType::F4E2M1FN
                    | $crate::DataType::F6E2M3FN
                    | $crate::DataType::F6E3M2FN
                    | $crate::DataType::F8E3M4
                    | $crate::DataType::F8E4M3
                    | $crate::DataType::F8E4M3FN
                    | $crate::DataType::F8E4M3FNUZ
                    | $crate::DataType::F8E4M3B11FNUZ
                    | $crate::DataType::F8E5M2
                    | $crate::DataType::F8E5M2FNUZ
                    | $crate::DataType::F8E8M0FNU
                    | $crate::DataType::BF16
                    | $crate::DataType::F16
                    | $crate::DataType::F32
                    | $crate::DataType::F64
                    | $crate::DataType::C64
                    | $crate::DataType::C128
            )
        }) {
            return Err($crate::TypeError {
                message: format!("'{descriptor}' does not support input data type {input_type}"),
            }
            .into());
        }
    }};

    (@no_unreduced, $descriptor:expr, $types:expr $(,)?) => {{
        let descriptor = $descriptor;
        let types = &$types[..];
        if types.iter().any(|r#type| !r#type.unreduced_axes().is_empty()) {
            return Err(
                $crate::TypeError { message: format!("'{descriptor}' does not support unreduced operands") }.into()
            );
        }
    }};

    (@same_unreduced_axes, $descriptor:expr, $types:expr $(,)?) => {{
        let descriptor = $descriptor;
        let types = &$types[..];
        if types.len() != 2 {
            return Err($crate::TypeError { message: format!("expected 2 inputs but got {}", types.len()) }.into());
        }
        if types[0].unreduced_axes() != types[1].unreduced_axes() {
            return Err($crate::TypeError {
                message: format!("'{descriptor}' operands must be unreduced over the same axes"),
            }
            .into());
        }
    }};

    (@same_reduced_axes, $descriptor:expr, $types:expr $(,)?) => {{
        let descriptor = $descriptor;
        let types = &$types[..];
        if types.len() != 2 {
            return Err($crate::TypeError { message: format!("expected 2 inputs but got {}", types.len()) }.into());
        }
        if types[0].reduced_axes() != types[1].reduced_axes() {
            return Err($crate::TypeError {
                message: format!("'{descriptor}' operands must be reduced over the same axes"),
            }
            .into());
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

/// Defines the structural implementations shared by elementwise operations. The generated base
/// includes the unit operation struct, its [`Display`](std::fmt::Display), [`Operation`](crate::Operation),
/// [`ElementwiseOperation`](crate::ElementwiseOperation), [`InterpretableOperation`](crate::InterpretableOperation),
/// and [`PartiallyEvaluatableOperation`](crate::PartiallyEvaluatableOperation) implementations.
///
/// # Parameters
///
///   - `@unary` / `@binary`: Selects the operation arity.
///   - `$(#[$documentation])*`: Documentation attributes attached to the generated operation struct.
///   - `$operation`: Identifier of the generated unit-struct operation (e.g., `SinOperation`).
///   - `$name`: Identifier of an existing operation-name constant (e.g., `SIN_OPERATION_NAME`).
///   - `$capability`: Identifier of the value-level capability trait bound by the generated
///     [`InterpretableOperation`](crate::InterpretableOperation) implementation (e.g., `Sin`).
///   - `$method`: Identifier of the capability trait method used for interpretation (e.g., `sin`).
///   - `check_data_types = [@selector, ...]`: Optional ordered list of [`check_types!`] selectors applied to scalar
///     input types and array element types before type inference.
///   - `check_array_types = [@selector, ...]`: Optional ordered list of [`check_types!`] selectors applied to array
///     input types before array broadcasting.
#[macro_export]
macro_rules! define_elementwise_operation {
    (
        @unary
        $(#[$documentation:meta])*
        $operation:ident, $name:ident,
        $capability:ident, $method:ident
        $(, check_data_types = [$(@$data_type_check:ident),* $(,)?])?
        $(, check_array_types = [$(@$array_type_check:ident),* $(,)?])? $(,)?
    ) => {
        $(#[$documentation])*
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
                $($($crate::check_types!(@$data_type_check, $name, input_types);)*)?
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
                $($($crate::check_types!(@$data_type_check, $name, &[input_types[0].data_type()]);)*)?
                $($($crate::check_types!(@$array_type_check, $name, input_types);)*)?
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
        $(#[$documentation:meta])*
        $operation:ident, $name:ident,
        $capability:ident, $method:ident
        $(, check_data_types = [$(@$data_type_check:ident),* $(,)?])?
        $(, check_array_types = [$(@$array_type_check:ident),* $(,)?])? $(,)?
    ) => {
        $(#[$documentation])*
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
                $($($crate::check_types!(@$data_type_check, $name, input_types);)*)?
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
                $($($crate::check_types!(@$data_type_check, $name, &[input_types[0].data_type(), input_types[1].data_type()]);)*)?
                $($($crate::check_types!(@$array_type_check, $name, input_types);)*)?
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

/// Defines a value-level capability trait paired with an elementwise operation and its dispatch-domain implementation.
///
/// # Parameters
///
///   - `@unary` / `@binary`: Selects whether the capability consumes only `self` or also a right-hand-side value.
///   - `$(#[$documentation])*`: Documentation attributes attached to the generated capability trait.
///   - `$capability`: Identifier of the generated value-level capability trait (e.g., `Sin`).
///   - `$method`: Identifier of the generated capability method (e.g., `sin`).
///   - `$operation`: Unit-struct operation bound through the value's dispatch domain (e.g., `SinOperation`).
#[macro_export]
macro_rules! define_elementwise_capability {
    (
        @unary
        $(#[$documentation:meta])*
        $capability:ident, $method:ident, $operation:ident $(,)?
    ) => {
        $(#[$documentation])*
        pub trait $capability: Sized {
            #[doc = concat!("Computes [`", stringify!($operation), "`] elementwise for this value.")]
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
        $(#[$documentation:meta])*
        $capability:ident, $method:ident, $operation:ident $(,)?
    ) => {
        $(#[$documentation])*
        pub trait $capability: Sized {
            #[doc = concat!("Computes [`", stringify!($operation), "`] elementwise for this value and `right`.")]
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

/// Implements the [`DifferentiableOperation`](crate::DifferentiableOperation) rule for an operation whose outputs carry
/// no tangent, such as a Boolean-codomain predicate or an explicit gradient barrier. The primal operation is replayed
/// on the input primals, and each output is paired with a structural zero tangent, which stays symbolic and stages
/// nothing. Because such a rule stages no live tangent, the operation can never appear on a linear operand in a valid
/// tangent program, and so it is typically paired with [`impl_non_transposable_operation!`].
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
            #[inline]
            fn jvp<D: $crate::DifferentiationDriver<C>>(
                &self,
                context: &C,
                _driver: &D,
                inputs: &[$crate::DifferentiationDual<C::Value>],
            ) -> Result<Vec<$crate::DifferentiationDual<C::Value>>, $crate::DifferentiationError> {
                // The outputs carry no tangent. We replay the primal operation on the input primals and pair each
                // output with a structural zero tangent, which stays symbolic and stages nothing.
                Ok($crate::Context::bind(
                    context,
                    self.clone(),
                    ::std::vec::Vec::new(),
                    &inputs.iter().map(|dual| dual.primal().clone()).collect::<Vec<_>>(),
                )?
                .into_iter()
                .map($crate::DifferentiationDual::new_with_zero_tangent)
                .collect())
            }
        }
    };
}

/// Implements the erroring [`TransposableOperation`](crate::TransposableOperation) rule for an operation that is not a
/// linear map on any input/operand. A valid tangent program never contains such an operation on a linear operand (its
/// forward-mode rule pairs replayed primals with tangents computed by other operations), so the generated rule reports
/// an [`UnsupportedOperation`](crate::ProgramError::UnsupportedOperation) error. The reason non-transposable operations
/// still implement [`TransposableOperation`](crate::TransposableOperation) at all is that transposition is driven
/// through whole operation families. Ryft used to have a separate linear operation family type, but that resulted
/// in overly complicated backend and operation implementations for very little benefit in practice, and so the two
/// families were unified.
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
            #[inline]
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
                Err($crate::ProgramError::UnsupportedOperation {
                    message: format!("operation `{}` is not transposable", $crate::Operation::<T>::name(self)),
                }
                .into())
            }
        }
    };
}

/// Implements the [`TransposableOperation`](crate::TransposableOperation) trait for a [`Region`](crate::Region)-less
/// nullary [`Operation`](crate::Operation). The generated implementation validates that the operation application has
/// no inputs, infers and validates its output count, and returns no operand cotangents. The optional leading generic
/// list declares operation-specific type parameters; the macro supplies the standard `T`, `V`, and `O` transposition
/// parameters and derives behavioral bounds from [`Operation<T>`](crate::Operation). An optional `where` clause can
/// provide bounds required to make the operation type itself well-formed.
///
/// # Parameters
///
///   - `$generic`: Optional operation-specific type parameters used by `$operation`.
///   - `$operation`: Regionless nullary operation type for which the implementation is generated.
///   - `$bounds`: Optional bounds required to make `$operation` well-formed.
#[macro_export]
macro_rules! impl_nullary_transposable_operation {
    (<$($generic:ident),+> $operation:ty where $($bounds:tt)+) => {
        $crate::impl_nullary_transposable_operation!(@impl [$($generic),+] ($operation) { $($bounds)+ });
    };
    (<$($generic:ident),+> $operation:ty $(,)?) => {
        $crate::impl_nullary_transposable_operation!(@impl [$($generic),+] ($operation) {});
    };
    ($operation:ty where $($bounds:tt)+) => {
        $crate::impl_nullary_transposable_operation!(@impl [] ($operation) { $($bounds)+ });
    };
    ($operation:ty $(,)?) => {
        $crate::impl_nullary_transposable_operation!(@impl [] ($operation) {});
    };
    (@impl [$($generic:ident),*] ($operation:ty) { $($bounds:tt)* }) => {
        impl<T: $crate::Type, V: $crate::Value<Type = T>, O: $crate::Operation<T> $(, $generic)*>
            $crate::TransposableOperation<V, O> for $operation
        where
            $operation: $crate::Operation<T>,
            $($bounds)*
        {
            #[inline]
            fn transpose<D: $crate::TranspositionDriver<V, O>>(
                &self,
                _context: &mut $crate::TracingContext<V, O>,
                _driver: &D,
                inputs: &[$crate::PartialValue<$crate::Tracer<$crate::TracingContext<V, O>>>],
                outputs: &[$crate::MaybeZero<$crate::Tracer<$crate::TracingContext<V, O>>>],
            ) -> Result<
                Vec<$crate::MaybeZero<$crate::Tracer<$crate::TracingContext<V, O>>>>,
                $crate::DifferentiationError,
            > {
                $crate::check_count!("input", inputs, 0, ProgramError);
                let output_count = $crate::Operation::<T>::infer_output_types(self, &[], &[])?.len();
                $crate::check_count!("output", outputs, output_count, ProgramError);
                Ok(Vec::new())
            }
        }
    };
}

/// Implements the [`BatchableOperation`](crate::BatchableOperation) trait for a [`Region`](crate::Region)-less
/// nullary [`Operation`](crate::Operation) according to the selected batching policy. The `@replicated` policy
/// interprets the operation once through the parent [`Context`](crate::Context) and marks every output as replicated
/// because the operation is invariant across the mapped axis. Nullary operations whose result depends on that axis,
/// such as [`AxisIndexOperation`](crate::AxisIndexOperation), require a custom batching rule instead. The optional
/// leading generic list declares operation-specific type parameters. Behavioral bounds are derived from
/// [`InterpretableOperation<C>`](crate::InterpretableOperation). An optional `where` clause can provide
/// bounds required to make the operation type itself well-formed.
///
/// # Parameters
///
///   - `@replicated`: Selects batching that evaluates the operation once and marks every output as replicated.
///   - `$generic`: Optional operation-specific type parameters used by `$operation`.
///   - `$operation`: Regionless nullary operation type for which the implementation is generated.
///   - `$bounds`: Optional bounds required to make `$operation` well-formed.
#[macro_export]
macro_rules! impl_nullary_batchable_operation {
    (@replicated <$($generic:ident),+> $operation:ty where $($bounds:tt)+) => {
        $crate::impl_nullary_batchable_operation!(@impl_replicated [$($generic),+] ($operation) { $($bounds)+ });
    };
    (@replicated <$($generic:ident),+> $operation:ty $(,)?) => {
        $crate::impl_nullary_batchable_operation!(@impl_replicated [$($generic),+] ($operation) {});
    };
    (@replicated $operation:ty where $($bounds:tt)+) => {
        $crate::impl_nullary_batchable_operation!(@impl_replicated [] ($operation) { $($bounds)+ });
    };
    (@replicated $operation:ty $(,)?) => {
        $crate::impl_nullary_batchable_operation!(@impl_replicated [] ($operation) {});
    };
    (@impl_replicated [$($generic:ident),*] ($operation:ty) { $($bounds:tt)* }) => {
        impl<C: $crate::Context<Type = $crate::ArrayType> $(, $generic)*> $crate::BatchableOperation<C> for $operation
        where
            $operation: $crate::InterpretableOperation<C>,
            $($bounds)*
        {
            #[inline]
            fn batch<D: $crate::BatchingDriver<C>>(
                &self,
                context: &$crate::BatchingContext<C>,
                _driver: &D,
                inputs: &[$crate::ArrayBatch<C::Value>],
            ) -> Result<Vec<$crate::ArrayBatch<C::Value>>, $crate::BatchingError> {
                $crate::check_count!("input", inputs, 0, ProgramError);
                Ok($crate::InterpretableOperation::interpret(
                    self,
                    context.parent(),
                    &$crate::EmptyRegionDriver,
                    &[],
                )?
                .into_iter()
                .map($crate::ArrayBatch::replicated)
                .collect())
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

// TODO(eaplatanios): Review this macro.
/// Checks a concrete [`Operation`](crate::Operation) contract through Ryft's reference transform machinery. This macro
/// is intended for operation unit tests in Ryft and downstream crates. Its selectors describe the semantic test case,
/// while its input and output lists encode [`Operation`](crate::Operation) arity directly:
///
///   - `@batching @exact` / `@batching @approx(epsilon = ...)`: Applies an operation to one or more batching cases.
///     Each input and expected output is written as `(@mapped(axis = ...), value)` or `(@replicated, value)`.
///     The default form uses the eager [`Array`](crate::Array) reference context, an
///     [`EmptyRegionDriver`](crate::EmptyRegionDriver), and replicated mapped-axis sharding. The extended
///     form accepts `context`, `driver`, and `axis_sharding` expressions. Exact comparison checks complete
///     [`ArrayBatch`](crate::ArrayBatch) equality. Approximate comparison still checks output types and
///     batch axes exactly and applies the epsilon only to `f64` payload values.
///   - `@partial_evaluation`: Builds a one-instruction program for each case and checks its partial-evaluation output
///     classification, residual instruction count, and replayed values. `@partial_evaluation @fold_and_residualize`
///     is the concise form for a single-output operation using the default partial-evaluation rule. That form checks
///     the all-known case, every individual unknown-input position, and the all-unknown case. Explicit cases use
///     `(@known, value)` or `(@unknown(type = ..., replay = ...))`. Outputs use `(@known, value)` or `(@residual,
///     value)`. The default form uses the eager [`Scalar`](crate::Scalar) reference backend. The extended
///     `backend = (Value, Operation)` form supports downstream value and operation families.
///   - `@reject @unreduced`: Checks that array inputs carrying an unreduced mesh axis are rejected.
///   - `@reject @mismatched_reduced`: Checks both operand orders for a binary operation whose operands
///     must carry the same reduced-axis markers.
///   - `@reject @transposition`: Builds a one-instruction array program and checks that transposition
///     reaches the operation's unsupported-transposition error. The input type list encodes arity.
///
/// [`Region`](crate::Region)-ful operations may use the extended batching form with their
/// [`Instruction`](crate::Instruction)-scoped driver, but tests whose main subject is nested-region
/// transformation should generally keep that setup explicit.
///
/// # Example
/// 
/// This is an example for how to use this macro to check the elementwise [`AddOperation`](crate::AddOperation):
///
/// ```rust
/// # use ryft_core::{Array, AddOperation, check_operation};
/// check_operation!(
///     @batching @exact,
///     operation = AddOperation,
///     axis_size = 2,
///     cases = [
///         {
///             inputs = [
///                 (@mapped(axis = 0), Array::vector(vec![1.0, 2.0])),
///                 (@replicated, Array::scalar(3.0)),
///             ],
///             outputs = [
///                 (@mapped(axis = 0), Array::vector(vec![4.0, 5.0])),
///             ],
///         },
///     ],
/// );
/// ```
///
/// # Parameters
///
///   - `operation = $operation`: [`Operation`](crate::Operation) expression evaluated once per macro invocation.
///   - `axis_size = $axis_size`: Size of the mapped batching axis. It remains explicit because no mapped input exists
///     from which to infer it in an all-replicated case.
///   - `cases = $cases`: Batching or partial-evaluation cases. Every case declares its inputs and expected outputs.
///     Partial-evaluation cases additionally declare the expected residual instruction count.
///   - `context = $context`: Optional parent [`Context`](crate::Context) for the extended batching form.
///   - `driver = $driver`: Optional [`BatchingDriver`](crate::BatchingDriver) for the extended batching form.
///   - `axis_sharding = $axis_sharding`: Optional [`ShardingDimension`](crate::ShardingDimension) assigned to the
///     mapped axis by the extended batching form.
///   - `backend = ($value, $operation_family)`: Optional value and operation-family types used to construct
///     partial-evaluation or transposition programs for downstream operation families.
///   - `input_types = $input_types`: Input types used by rejection checks, in operation input order.
#[macro_export]
macro_rules! check_operation {
    (
        @batching @exact,
        operation = $operation:expr,
        axis_size = $axis_size:expr,
        cases = $cases:tt $(,)?
    ) => {
        $crate::check_operation!(
            @batching_run (@exact),
            context = $crate::contexts::EagerContext::<
                $crate::backends::arrays::Array,
                $crate::backends::arrays::ArrayOperation<$crate::backends::arrays::Array>,
            >::new(),
            driver = &$crate::programs::regions::EmptyRegionDriver,
            axis_sharding = $crate::sharding::ShardingDimension::Replicated,
            operation = $operation,
            axis_size = $axis_size,
            cases = $cases,
        )
    };

    (
        @batching @approx(epsilon = $epsilon:expr),
        operation = $operation:expr,
        axis_size = $axis_size:expr,
        cases = $cases:tt $(,)?
    ) => {
        $crate::check_operation!(
            @batching_run (@approx($epsilon)),
            context = $crate::contexts::EagerContext::<
                $crate::backends::arrays::Array,
                $crate::backends::arrays::ArrayOperation<$crate::backends::arrays::Array>,
            >::new(),
            driver = &$crate::programs::regions::EmptyRegionDriver,
            axis_sharding = $crate::sharding::ShardingDimension::Replicated,
            operation = $operation,
            axis_size = $axis_size,
            cases = $cases,
        )
    };

    (
        @batching @exact,
        context = $context:expr,
        driver = $driver:expr,
        axis_sharding = $axis_sharding:expr,
        operation = $operation:expr,
        axis_size = $axis_size:expr,
        cases = $cases:tt $(,)?
    ) => {
        $crate::check_operation!(
            @batching_run (@exact),
            context = $context,
            driver = $driver,
            axis_sharding = $axis_sharding,
            operation = $operation,
            axis_size = $axis_size,
            cases = $cases,
        )
    };

    (
        @batching @approx(epsilon = $epsilon:expr),
        context = $context:expr,
        driver = $driver:expr,
        axis_sharding = $axis_sharding:expr,
        operation = $operation:expr,
        axis_size = $axis_size:expr,
        cases = $cases:tt $(,)?
    ) => {
        $crate::check_operation!(
            @batching_run (@approx($epsilon)),
            context = $context,
            driver = $driver,
            axis_sharding = $axis_sharding,
            operation = $operation,
            axis_size = $axis_size,
            cases = $cases,
        )
    };

    (
        @batching_run $comparison:tt,
        context = $context:expr,
        driver = $driver:expr,
        axis_sharding = $axis_sharding:expr,
        operation = $operation:expr,
        axis_size = $axis_size:expr,
        cases = [
            $(
                {
                    inputs = [$($input:tt),* $(,)?],
                    outputs = [$($output:tt),* $(,)?] $(,)?
                }
            ),* $(,)?
        ] $(,)?
    ) => {{
        let operation = $operation;
        let driver = $driver;
        let axis_size = $axis_size;
        let axis_sharding = $axis_sharding;
        let context = $crate::batching::BatchingContext::new($context, axis_size)
            .with_axis_sharding(axis_sharding);
        $(
            let inputs = vec![$($crate::check_operation!(@batch_value $input)),*];
            let expected_outputs = vec![$($crate::check_operation!(@batch_value $output)),*];
            let actual_outputs = $crate::batching::BatchableOperation::batch(
                &operation,
                &context,
                driver,
                inputs.as_slice(),
            )
            .unwrap();
            $crate::check_operation!(@compare_batches $comparison, actual_outputs, expected_outputs);
        )*
    }};

    (@batch_value (@mapped(axis = $axis:expr), $value:expr)) => {{
        let value = $value;
        let r#type = $crate::programs::types::Typed::r#type(&value).into_owned();
        $crate::batching::ArrayBatch::new(r#type, value, $crate::batching::BatchAxis::new($axis)).unwrap()
    }};

    (@batch_value (@replicated, $value:expr)) => {
        $crate::batching::ArrayBatch::replicated($value)
    };

    (@compare_batches (@exact), $actual:expr, $expected:expr) => {{
        assert_eq!($actual, $expected);
    }};

    (@compare_batches (@approx($epsilon:expr)), $actual:expr, $expected:expr) => {{
        let actual = $actual;
        let expected = $expected;
        let epsilon: f64 = $epsilon;
        assert_eq!(actual.len(), expected.len());
        for (output_index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                $crate::programs::types::Typed::r#type(actual),
                $crate::programs::types::Typed::r#type(expected),
                "batching output {output_index} has the wrong type",
            );
            assert_eq!(
                actual.batch_axis(),
                expected.batch_axis(),
                "batching output {output_index} has the wrong batch axis",
            );
            let actual_values = actual.value().to_f64s();
            let expected_values = expected.value().to_f64s();
            assert_eq!(actual_values.len(), expected_values.len());
            for (value_index, (actual, expected)) in
                actual_values.iter().zip(expected_values.iter()).enumerate()
            {
                assert!(
                    (actual - expected).abs() <= epsilon,
                    "batching output {output_index} value {value_index} differs: {actual} vs {expected}",
                );
            }
        }
    }};

    (
        @partial_evaluation @fold_and_residualize,
        operation = $operation:expr,
        inputs = [$($input:expr),+ $(,)?],
        expected = $expected:expr $(,)?
    ) => {{
        let operation = $operation;
        let inputs: Vec<$crate::backends::scalars::Scalar> =
            vec![$(::core::convert::Into::into($input)),+];
        let expected: $crate::backends::scalars::Scalar = ::core::convert::Into::into($expected);
        let mut builder = $crate::programs::builders::ProgramBuilder::<
            $crate::backends::scalars::Scalar,
            $crate::backends::scalars::ScalarOperation<$crate::backends::scalars::Scalar>,
        >::new();
        let input_ids = inputs
            .iter()
            .map(|input| builder.add_input($crate::programs::types::Typed::r#type(input).into_owned()))
            .collect::<Vec<_>>();
        let operation: $crate::backends::scalars::ScalarOperation<$crate::backends::scalars::Scalar> =
            ::core::convert::Into::into(operation);
        let output_ids = builder.add_instruction(operation, Vec::new(), input_ids).unwrap().to_vec();
        assert_eq!(output_ids.len(), 1);
        let program = builder
            .build::<
                Vec<$crate::backends::scalars::Scalar>,
                Vec<$crate::backends::scalars::Scalar>,
            >(
                output_ids,
                vec![$crate::parameters::Placeholder; inputs.len()],
                vec![$crate::parameters::Placeholder],
            )
            .unwrap();
        let context = $crate::contexts::EagerContext::<
            $crate::backends::scalars::Scalar,
            $crate::backends::scalars::ScalarOperation<$crate::backends::scalars::Scalar>,
        >::new();

        let known = inputs
            .iter()
            .cloned()
            .map($crate::partial::PartialValue::Known)
            .collect::<Vec<_>>();
        let evaluation = program.partially_evaluate(known.as_slice()).unwrap();
        assert!(evaluation.program().instructions().is_empty());
        assert_eq!(evaluation.outputs().len(), 1);
        assert!(evaluation.outputs()[0].is_known());
        assert_eq!(evaluation.interpret(&context, &[]).unwrap(), vec![expected.clone()]);

        for unknown_index in 0..inputs.len() {
            let mut knowledge = known.clone();
            knowledge[unknown_index] = $crate::partial::PartialValue::Unknown(
                $crate::programs::types::Typed::r#type(&inputs[unknown_index]).into_owned(),
            );
            let evaluation = program.partially_evaluate(knowledge.as_slice()).unwrap();
            assert_eq!(evaluation.program().instructions().len(), 1);
            assert_eq!(evaluation.outputs().len(), 1);
            assert!(evaluation.outputs()[0].is_unknown());
            assert_eq!(
                evaluation.interpret(&context, &[inputs[unknown_index].clone()]).unwrap(),
                vec![expected.clone()],
            );
        }

        if inputs.len() > 1 {
            let unknown = inputs
                .iter()
                .map(|input| {
                    $crate::partial::PartialValue::Unknown(
                        $crate::programs::types::Typed::r#type(input).into_owned(),
                    )
                })
                .collect::<Vec<_>>();
            let evaluation = program.partially_evaluate(unknown.as_slice()).unwrap();
            assert_eq!(evaluation.program().instructions().len(), 1);
            assert_eq!(evaluation.outputs().len(), 1);
            assert!(evaluation.outputs()[0].is_unknown());
            assert_eq!(evaluation.interpret(&context, inputs.as_slice()).unwrap(), vec![expected]);
        }
    }};

    (
        @partial_evaluation,
        operation = $operation:expr,
        cases = $cases:tt $(,)?
    ) => {
        $crate::check_operation!(
            @partial_evaluation,
            backend = (
                $crate::backends::scalars::Scalar,
                $crate::backends::scalars::ScalarOperation<$crate::backends::scalars::Scalar>
            ),
            operation = $operation,
            cases = $cases,
        )
    };

    (
        @partial_evaluation,
        backend = ($value:ty, $operation_family:ty),
        operation = $operation:expr,
        cases = [
            $(
                {
                    inputs = [$($input:tt),* $(,)?],
                    outputs = [$($output:tt),* $(,)?],
                    residual_instructions = $instruction_count:expr $(,)?
                }
            ),* $(,)?
        ] $(,)?
    ) => {{
        let operation = $operation;
        $(
        {
            let inputs: Vec<($crate::partial::PartialValue<$value>, Option<$value>)> =
                vec![$($crate::check_operation!(@partial_input $value, $input)),*];
            let expected_outputs: Vec<(bool, $value)> =
                vec![$($crate::check_operation!(@partial_output $output)),*];
            let mut builder = $crate::programs::builders::ProgramBuilder::<$value, $operation_family>::new();
            let input_ids = inputs
                .iter()
                .map(|(input, _)| {
                    builder.add_input($crate::programs::types::Typed::r#type(input).into_owned())
                })
                .collect::<Vec<_>>();
            let operation: $operation_family = ::core::convert::Into::into(operation.clone());
            let output_ids = builder.add_instruction(operation, Vec::new(), input_ids).unwrap().to_vec();
            let output_count = output_ids.len();
            let program = builder
                .build::<Vec<$value>, Vec<$value>>(
                    output_ids,
                    vec![$crate::parameters::Placeholder; inputs.len()],
                    vec![$crate::parameters::Placeholder; output_count],
                )
                .unwrap();
            let knowledge = inputs.iter().map(|(input, _)| input.clone()).collect::<Vec<_>>();
            let evaluation = program.partially_evaluate(knowledge.as_slice()).unwrap();
            assert_eq!(evaluation.program().instructions().len(), $instruction_count);
            assert_eq!(evaluation.outputs().len(), expected_outputs.len());
            for (actual, (expected_known, _)) in evaluation.outputs().iter().zip(expected_outputs.iter()) {
                assert_eq!(actual.is_known(), *expected_known);
            }
            let unknown_inputs = inputs
                .into_iter()
                .filter_map(|(_, replay)| replay)
                .collect::<Vec<_>>();
            let actual_outputs = evaluation
                .interpret(
                    &$crate::contexts::EagerContext::<$value, $operation_family>::new(),
                    unknown_inputs.as_slice(),
                )
                .unwrap();
            let expected_outputs = expected_outputs
                .into_iter()
                .map(|(_, value)| value)
                .collect::<Vec<_>>();
            assert_eq!(actual_outputs, expected_outputs);
        }
        )*
    }};

    (@partial_input $value:ty, (@known, $input:expr)) => {{
        let input: $value = ::core::convert::Into::into($input);
        ($crate::partial::PartialValue::Known(input), Option::<$value>::None)
    }};

    (@partial_input $value:ty, (@unknown(type = $r#type:expr, replay = $input:expr))) => {{
        let input: $value = ::core::convert::Into::into($input);
        ($crate::partial::PartialValue::Unknown($r#type), Some(input))
    }};

    (@partial_output (@known, $output:expr)) => {
        (true, ::core::convert::Into::into($output))
    };

    (@partial_output (@residual, $output:expr)) => {
        (false, ::core::convert::Into::into($output))
    };

    (
        @reject @unreduced,
        operation = $operation:expr,
        input_types = [$($input_type:expr),+ $(,)?] $(,)?
    ) => {{
        let operation = $operation;
        let descriptor = $crate::programs::operations::Operation::<$crate::types::ArrayType>::name(&operation);
        let mesh = $crate::sharding::LogicalMesh::new(vec![
            $crate::sharding::MeshAxis::new("x", 2, $crate::sharding::MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let input_types = vec![$($input_type),+]
            .into_iter()
            .map(|input_type| {
                let dimensions = vec![$crate::sharding::ShardingDimension::Replicated; input_type.rank()];
                let sharding = $crate::sharding::Sharding::new(mesh.clone(), dimensions)
                    .unwrap()
                    .with_unreduced_axes(["x"])
                    .unwrap();
                input_type.with_sharding(sharding).unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            $crate::programs::operations::Operation::infer_output_types(&operation, input_types.as_slice(), &[]),
            Err($crate::programs::types::TypeError {
                message: format!("'{descriptor}' does not support unreduced operands"),
            }),
        );
    }};

    (
        @reject @mismatched_reduced,
        operation = $operation:expr,
        input_types = [$left_type:expr, $right_type:expr $(,)?] $(,)?
    ) => {{
        let operation = $operation;
        let descriptor = $crate::programs::operations::Operation::<$crate::types::ArrayType>::name(&operation);
        let mesh = $crate::sharding::LogicalMesh::new(vec![
            $crate::sharding::MeshAxis::new("x", 2, $crate::sharding::MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let plain = |input_type: $crate::types::ArrayType| {
            let dimensions = vec![$crate::sharding::ShardingDimension::Replicated; input_type.rank()];
            input_type
                .with_sharding($crate::sharding::Sharding::new(mesh.clone(), dimensions).unwrap())
                .unwrap()
        };
        let left = plain($left_type);
        let right = plain($right_type);
        let reduced_left = left
            .clone()
            .with_sharding(left.sharding().unwrap().clone().with_reduced_axes(["x"]).unwrap())
            .unwrap();
        let reduced_right = right
            .clone()
            .with_sharding(right.sharding().unwrap().clone().with_reduced_axes(["x"]).unwrap())
            .unwrap();
        let expected = Err($crate::programs::types::TypeError {
            message: format!("'{descriptor}' operands must be reduced over the same axes"),
        });
        assert_eq!(
            $crate::programs::operations::Operation::infer_output_types(
                &operation,
                &[reduced_left, right.clone()],
                &[],
            ),
            expected.clone(),
        );
        assert_eq!(
            $crate::programs::operations::Operation::infer_output_types(
                &operation,
                &[left, reduced_right],
                &[],
            ),
            expected,
        );
    }};

    (
        @reject @transposition,
        operation = $operation:expr,
        input_types = $input_types:tt $(,)?
    ) => {
        $crate::check_operation!(
            @reject @transposition,
            backend = (
                $crate::backends::arrays::Array,
                $crate::backends::arrays::ArrayOperation<$crate::backends::arrays::Array>
            ),
            operation = $operation,
            input_types = $input_types,
        )
    };

    (
        @reject @transposition,
        backend = ($value:ty, $operation_family:ty),
        operation = $operation:expr,
        input_types = [$($input_type:expr),+ $(,)?] $(,)?
    ) => {{
        let operation: $operation_family = ::core::convert::Into::into($operation);
        let descriptor = $crate::programs::operations::Operation::name(&operation);
        let mut builder = $crate::programs::builders::ProgramBuilder::<$value, $operation_family>::new();
        let input_types = vec![$($input_type),+];
        let input_count = input_types.len();
        let input_ids = input_types
            .into_iter()
            .map(|input_type| builder.add_input(input_type))
            .collect::<Vec<_>>();
        let output_ids = builder.add_instruction(operation, Vec::new(), input_ids).unwrap().to_vec();
        let output_count = output_ids.len();
        let program = builder
            .build::<Vec<$value>, Vec<$value>>(
                output_ids,
                vec![$crate::parameters::Placeholder; input_count],
                vec![$crate::parameters::Placeholder; output_count],
            )
            .unwrap();
        let input_indices = (0..input_count).collect::<Vec<_>>();
        assert!(matches!(
            program.transpose_with_respect_to(input_indices.as_slice()),
            Err($crate::differentiation::DifferentiationError::Program(
                $crate::programs::ProgramError::UnsupportedOperation { message },
            )) if message == format!("operation `{descriptor}` is not transposable"),
        ));
    }};
}

/// Asserts that the reverse-mode gradient of a function at an input matches a central finite-difference estimate of its
/// derivative within an absolute tolerance. This is the standard oracle for testing operation gradient rules without
/// hand-deriving the expected derivative and without trusting the machinery under test (i.e., the gradient side runs
/// the function through [`gradient`](crate::gradient), while the finite-difference side evaluates the function directly
/// on concrete values at the perturbed points, never touching the differentiation machinery that it is checking). That
/// double instantiation is why this is a macro: the function must be a closure literal (or a generic function), and the
/// internal `@check` rule shared by both selectors instantiates it once over
/// [`LinearizationTracer`](crate::LinearizationTracer) inputs and once over concrete [`Scalar`](crate::Scalar)
/// or [`Array`](crate::Array) inputs before dispatching to the selector's internal assertion rule.
///
/// An `f64`-typed input estimates the ordinary derivative `(f(x + h) - f(x - h)) / (2h)` for `@scalar`. For `@array`,
/// this is computed once per input element with all other elements held fixed, assembling the estimated gradient array.
/// A `c128`-typed input requires a ℂ → ℝ function and estimates both real partials (per element under `@array`) with
/// central differences along the real and imaginary axes, assembling `complex(∂f/∂re, -∂f/∂im)`, the conjugate
/// steepest-ascent gradient the bilinear transposition pairing returns (e.g., `2z̄` for `f(z) = |z|²`). Other input
/// data types (including `c64`, whose `f32` precision cannot support a meaningful central difference) panic!
///
/// # Parameters
///
///   - `@scalar` / `@array`: Selects the value universe: `@scalar` checks a [`Scalar`](crate::Scalar)-valued function,
///     while `@array` checks an [`Array`](crate::Array)-valued function of any input shape whose output is a rank-0
///     real `f64` array (i.e., the only shape the plain [`gradient`](crate::gradient) entry point accepts).
///   - `$function`: Closure literal (or generic function) to differentiate. The function may return its output value
///     either directly or wrapped in a [`Result`] whose error type converts into the differentiation machinery's error
///     types (which holds for the [`ProgramError`](crate::ProgramError) that the value capability traits return), so
///     fallible capability calls like `x.sin()` need no `.unwrap()`. Refer to [`MaybeFallible`](crate::MaybeFallible)
///     for the exact contract.
///   - `at = $input`: Expression convertible into the selected universe's value, at which the gradient is checked.
///   - `step = $step`: Central finite-difference spacing `h`.
///   - `tolerance = $tolerance`: Absolute tolerance for the comparison. Pick one compatible with the `O($step²)`
///     truncation error of the central difference.
#[macro_export]
macro_rules! check_gradient {
    (@scalar, $function:expr, at = $input:expr, step = $step:expr, tolerance = $tolerance:expr $(,)?) => {
        $crate::check_gradient!(
            @check(
                $crate::backends::scalars::Scalar,
                $crate::backends::scalars::ScalarOperation<$crate::backends::scalars::Scalar>,
                @assert_scalar,
            )
            $function, $input, $step, $tolerance
        )
    };

    (@array, $function:expr, at = $input:expr, step = $step:expr, tolerance = $tolerance:expr $(,)?) => {
        $crate::check_gradient!(
            @check(
                $crate::backends::arrays::Array,
                $crate::backends::arrays::ArrayOperation<$crate::backends::arrays::Array>,
                @assert_array,
            )
            $function, $input, $step, $tolerance
        )
    };

    // Internal rule shared by both scalar and array branches of this macro. `$value` and `$operation` pick the eager
    // context whose linearization tracer pins the traced instantiation of `$function`, and `@$assert` names the
    // internal rule below that checks the resulting gradient against the concrete-side finite-difference estimate.
    (
        @check($value:ty, $operation:ty, @$assert:ident $(,)?)
        $function:expr, $input:expr, $step:expr, $tolerance:expr
    ) => {{
        // Closure parameter types flow one way in Rust. A closure literal takes its signature from the expected type at
        // the point where the closure expression appears, and function calls in its body require the parameter type to
        // already be known there (i.e., a later call of the bound closure cannot type the body retroactively). Each
        // "paste" of `$function` therefore flows through an identity function that supplies the expected signature at
        // the "paste" site. For this reason, `pin_eager` is load-bearing. Without it, the concrete copy is bound by a
        // plain `let` with no expected type and every call site fails with "type annotations needed". `pin_traced` is
        // belt-and-braces today as the `gradient` bounds let inference recover the signature from the concretely typed
        // input, but it pins the canonical eager reference context explicitly instead of leaving that choice to
        // inference through `Input::To<LinearizationTracer<V::ExecutionDomain>>`, and it keeps closure-shape errors
        // anchored to one concrete expected signature.
        //
        // Both pins leave the function's output type generic so that `$function` may return its value either directly
        // or wrapped in a `Result` (refer to `MaybeFallible` for the exact contract). The traced copy's output is
        // consumed by `gradient`, which accepts both shapes natively, while `pin_eager` normalizes the concrete copy's
        // output by unwrapping through `MaybeFallible::into_result`, so that the assertion rules below always receive
        // an infallible function.

        fn pin_traced<
            F: Fn(
                $crate::differentiation::LinearizationTracer<$crate::contexts::EagerContext<$value, $operation>>,
            ) -> Output,
            Output,
        >(function: F) -> F {
            function
        }

        fn pin_eager<F: Fn($value) -> Output, Output: $crate::MaybeFallible<$value, $crate::ProgramError>>(
            function: F,
        ) -> impl Fn($value) -> $value {
            move |input| {
                $crate::MaybeFallible::into_result(function(input)).unwrap_or_else(|error| panic!("{error}"))
            }
        }

        let input: $value = ::core::convert::Into::into($input);
        let step: f64 = $step;
        let tolerance: f64 = $tolerance;
        let gradient = $crate::differentiation::gradient(pin_traced($function), input.clone()).unwrap();

        $crate::check_gradient!(@$assert(gradient, pin_eager($function), input, step, tolerance))
    }};

    // Internal rule behind the `@scalar` branch of this macro. It checks a reverse-mode `$gradient` of the ℝ → ℝ or
    // ℂ → ℝ function `$evaluate` at `$input` against the central finite-difference estimate of its derivative.
    (@assert_scalar($gradient:expr, $evaluate:expr, $input:expr, $step:expr, $tolerance:expr $(,)?)) => {{
        let gradient = $gradient;
        let evaluate = $evaluate;
        let input = $input;
        let step = $step;
        let tolerance = $tolerance;

        let central_difference = |plus: $crate::backends::scalars::Scalar, minus: $crate::backends::scalars::Scalar| {
            (evaluate(plus) - evaluate(minus)) / $crate::backends::scalars::Scalar::from(2.0 * step)
        };

        match input {
            $crate::backends::scalars::Scalar::F64(input) => {
                let estimate = central_difference(
                    $crate::backends::scalars::Scalar::from(input + step),
                    $crate::backends::scalars::Scalar::from(input - step),
                );
                ::approx::assert_abs_diff_eq!(gradient, estimate, epsilon = tolerance);
            }
            $crate::backends::scalars::Scalar::C128(_) => {
                // The two central differences estimate the real partials that assemble the conjugate
                // steepest-ascent gradient `complex(∂f/∂re, -∂f/∂im)`.
                let (real_step, imaginary_step) = $crate::check_gradient!(@complex_perturbation_steps(step));
                let real_estimate = central_difference(input + real_step, input - real_step);
                let imaginary_estimate = central_difference(input + imaginary_step, input - imaginary_step);
                let estimate =
                    $crate::operations::complex::Complex::complex(&real_estimate, &(-imaginary_estimate)).unwrap();
                ::approx::assert_abs_diff_eq!(gradient, estimate, epsilon = tolerance);
            }
            other => panic!(
                "finite-difference gradient checking requires an f64 or c128 input but got {}",
                $crate::programs::types::Typed::r#type(&other).into_owned(),
            ),
        }
    }};

    // Internal rule behind the `@array` branch of this macro. It checks a reverse-mode `$gradient` of the ℝⁿ → ℝ or
    // ℂⁿ → ℝ function `$evaluate` at `$input` (an array of any shape whose output is a rank-0 real `f64` array) against
    // the central finite-difference estimates of its partials, perturbing one input element at a time with all others
    // held fixed.
    (@assert_array($gradient:expr, $evaluate:expr, $input:expr, $step:expr, $tolerance:expr $(,)?)) => {{
        let gradient = $gradient;
        let evaluate = $evaluate;
        let input = $input;
        let step = $step;
        let tolerance = $tolerance;

        // The function output is a rank-0 real array, so the central difference reads its single `f64` element.
        let central_difference = |plus: $crate::backends::arrays::Array, minus: $crate::backends::arrays::Array| {
            (evaluate(plus).to_f64s()[0] - evaluate(minus).to_f64s()[0]) / (2.0 * step)
        };

        let input_type = $crate::programs::types::Typed::r#type(&input).into_owned();
        let element_count = input.values().len();
        match input_type.data_type() {
            $crate::types::DataType::F64 => {
                let perturbed = |index: usize, delta: f64| {
                    let mut values = input.to_f64s();
                    values[index] += delta;
                    $crate::backends::arrays::Array::from_f64s(input_type.clone(), values)
                };
                let estimates = (0..element_count)
                    .map(|index| central_difference(perturbed(index, step), perturbed(index, -step)))
                    .collect::<Vec<_>>();
                let estimate = $crate::backends::arrays::Array::from_f64s(input_type.clone(), estimates);
                ::approx::assert_abs_diff_eq!(gradient, estimate, epsilon = tolerance);
            }
            $crate::types::DataType::C128 => {
                // Per input element, the two central differences estimate the real partials that assemble the
                // conjugate steepest-ascent gradient `complex(∂f/∂re, -∂f/∂im)`.
                let (real_step, imaginary_step) = $crate::check_gradient!(@complex_perturbation_steps(step));
                let perturbed = |index: usize, delta: $crate::backends::scalars::Scalar| {
                    let mut values = input.values().to_vec();
                    values[index] = values[index] + delta;
                    $crate::backends::arrays::Array::new(input_type.clone(), values).unwrap()
                };
                let mut real_estimates = Vec::with_capacity(element_count);
                let mut imaginary_estimates = Vec::with_capacity(element_count);
                for index in 0..element_count {
                    real_estimates
                        .push(central_difference(perturbed(index, real_step), perturbed(index, -real_step)));
                    imaginary_estimates.push(-central_difference(
                        perturbed(index, imaginary_step),
                        perturbed(index, -imaginary_step),
                    ));
                }
                let part_type = input_type.clone().with_data_type($crate::types::DataType::F64);
                let estimate = $crate::operations::complex::Complex::complex(
                    &$crate::backends::arrays::Array::from_f64s(part_type.clone(), real_estimates),
                    &$crate::backends::arrays::Array::from_f64s(part_type, imaginary_estimates),
                )
                .unwrap();
                ::approx::assert_abs_diff_eq!(gradient, estimate, epsilon = tolerance);
            }
            other => panic!("finite-difference gradient checking requires an f64 or c128 input but got {other}"),
        }
    }};

    // Internal rule shared by the complex arms of both assertion rules of this macro. It builds the complex-valued
    // real- and imaginary-axis perturbation steps so that the central differences remain in the complex tangent space.
    (@complex_perturbation_steps($step:expr)) => {{
        let real_step = $crate::operations::complex::Complex::complex(
            &$crate::backends::scalars::Scalar::from($step),
            &$crate::backends::scalars::Scalar::from(0.0),
        )
        .unwrap();
        let imaginary_step = $crate::operations::complex::Complex::complex(
            &$crate::backends::scalars::Scalar::from(0.0),
            &$crate::backends::scalars::Scalar::from($step),
        )
        .unwrap();
        (real_step, imaginary_step)
    }};
}

pub use crate::{
    check_builders, check_count, check_gradient, check_operation, check_sharding, check_types,
    define_elementwise_capability, define_elementwise_operation, define_tracer_operator,
    impl_non_differentiable_operation, impl_non_transposable_operation, impl_nullary_batchable_operation,
    impl_nullary_transposable_operation,
};

#[cfg(test)]
mod tests {
    use std::fmt::{Debug, Display, Formatter};
    use std::marker::PhantomData;
    use std::rc::Rc;

    use num_complex::Complex;

    use crate::backends::arrays::Array;
    use crate::backends::scalars::Scalar;
    use crate::batching::{ArrayBatch, BatchableOperation, BatchingContext, BatchingTracer};
    use crate::contexts::{Domain, EagerContext, StagingContext};
    use crate::differentiation::{
        DifferentiableOperation, DifferentiationContext, DifferentiationDual, DifferentiationError,
        DifferentiationTracer, TransposableOperation,
    };
    use crate::interpretation::{InterpretableOperation, InterpretationDriver};
    use crate::operations::ElementwiseOperation;
    use crate::operations::constants::ZeroOperation;
    use crate::operations::manipulation::{BroadcastOperation, TransposeOperation};
    use crate::operations::math::{
        Abs, Add, AddOperation, Neg, NegOperation, Reduce, ReductionKind, SinOperation, SubOperation,
    };
    use crate::partial::{
        PartialEvaluationContext, PartialEvaluationValue, PartialTracer, PartialValue, PartiallyEvaluatableOperation,
    };
    use crate::programs::ProgramError;
    use crate::programs::atoms::MaybeZero;
    use crate::programs::operations::Operation;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::programs::types::{Type, TypeError};
    use crate::sharding::{Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingError};
    use crate::tracing::{Tracer, TracingContext};
    use crate::types::{ArrayType, DataType, Shape, Size};

    const TEST_UNARY_OPERATION_NAME: &str = "test_unary";
    const TEST_BINARY_OPERATION_NAME: &str = "test_binary";

    define_elementwise_operation!(
        @unary
        /// Unary operation used to test [`define_elementwise_operation!`].
        TestUnaryOperation, TEST_UNARY_OPERATION_NAME,
        Neg, neg,
        check_data_types = [@numeric],
        check_array_types = [@no_unreduced],
    );

    define_elementwise_operation!(
        @binary
        /// Binary operation used to test [`define_elementwise_operation!`].
        TestBinaryOperation, TEST_BINARY_OPERATION_NAME,
        Add, add,
        check_data_types = [@numeric],
        check_array_types = [@same_unreduced_axes, @same_reduced_axes],
    );

    define_elementwise_capability!(
        @unary
        /// Unary capability used to test [`define_elementwise_capability!`].
        TestUnary, test_unary, TestUnaryOperation,
    );

    define_elementwise_capability!(
        @binary
        /// Binary capability used to test [`define_elementwise_capability!`].
        TestBinary, test_binary, TestBinaryOperation,
    );

    impl_non_differentiable_operation!(TestUnaryOperation);
    impl_non_transposable_operation!(TestUnaryOperation);
    impl_non_differentiable_operation!(TestBinaryOperation);

    impl From<ZeroOperation<DataType>> for TestUnaryOperation {
        fn from(_operation: ZeroOperation<DataType>) -> Self {
            Self
        }
    }

    impl From<ZeroOperation<DataType>> for TestBinaryOperation {
        fn from(_operation: ZeroOperation<DataType>) -> Self {
            Self
        }
    }

    impl From<TransposeOperation> for TestUnaryOperation {
        fn from(_operation: TransposeOperation) -> Self {
            Self
        }
    }

    impl From<BroadcastOperation> for TestUnaryOperation {
        fn from(_operation: BroadcastOperation) -> Self {
            Self
        }
    }

    impl From<NegOperation> for TestUnaryOperation {
        fn from(_operation: NegOperation) -> Self {
            Self
        }
    }

    impl From<TransposeOperation> for TestBinaryOperation {
        fn from(_operation: TransposeOperation) -> Self {
            Self
        }
    }

    impl From<BroadcastOperation> for TestBinaryOperation {
        fn from(_operation: BroadcastOperation) -> Self {
            Self
        }
    }

    impl From<AddOperation> for TestBinaryOperation {
        fn from(_operation: AddOperation) -> Self {
            Self
        }
    }

    /// Unary operator used to test [`define_tracer_operator!`].
    trait TestUnaryOperator {
        /// Result of applying this operator.
        type Output;

        /// Applies the operator.
        fn apply_unary(self) -> Self::Output;
    }

    /// Binary operator used to test [`define_tracer_operator!`].
    trait TestBinaryOperator {
        /// Result of applying this operator.
        type Output;

        /// Applies the operator.
        fn apply_binary(self, right: Self) -> Self::Output;
    }

    define_tracer_operator!(
        @unary TestUnaryOperator,
        apply_unary,
        TestUnaryOperation,
        "test unary operation failed",
    );

    define_tracer_operator!(
        @binary TestBinaryOperator,
        apply_binary,
        TestBinaryOperation,
        "test binary operation failed",
    );

    /// Nullary operation used to execute generated transposition and batching rules.
    #[derive(Clone, Debug, Default)]
    struct TestNullaryOperation;

    impl Display for TestNullaryOperation {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("test_nullary")
        }
    }

    impl Operation<DataType> for TestNullaryOperation {
        fn name(&self) -> &'static str {
            "test_nullary"
        }

        fn infer_output_types(
            &self,
            input_types: &[DataType],
            _region_interfaces: &[crate::RegionInterface<DataType>],
        ) -> Result<Vec<DataType>, TypeError> {
            check_count!("input", input_types, 0, TypeError);
            Ok(vec![DataType::F64, DataType::F64])
        }
    }

    impl Operation<ArrayType> for TestNullaryOperation {
        fn name(&self) -> &'static str {
            "test_nullary"
        }

        fn infer_output_types(
            &self,
            input_types: &[ArrayType],
            _region_interfaces: &[crate::RegionInterface<ArrayType>],
        ) -> Result<Vec<ArrayType>, TypeError> {
            check_count!("input", input_types, 0, TypeError);
            Ok(vec![ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)])
        }
    }

    impl InterpretableOperation<EagerContext<Array, TestNullaryOperation>> for TestNullaryOperation {
        fn interpret<D: crate::InterpretationDriver<EagerContext<Array, TestNullaryOperation>>>(
            &self,
            _context: &EagerContext<Array, TestNullaryOperation>,
            _driver: &D,
            inputs: &[Array],
        ) -> Result<Vec<Array>, ProgramError> {
            check_count!("input", inputs, 0, ProgramError);
            Ok(vec![Array::scalar(3.0), Array::scalar(4.0)])
        }
    }

    impl_nullary_transposable_operation!(TestNullaryOperation);
    impl_nullary_batchable_operation!(@replicated TestNullaryOperation);

    /// Nullary operation used to instantiate the non-generic `where` macro forms.
    #[derive(Clone, Debug, Default)]
    struct TestWhereNullaryOperation;

    impl Display for TestWhereNullaryOperation {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("test_where_nullary")
        }
    }

    impl<T: Type> Operation<T> for TestWhereNullaryOperation {
        fn name(&self) -> &'static str {
            "test_where_nullary"
        }

        fn infer_output_types(
            &self,
            input_types: &[T],
            _region_interfaces: &[crate::RegionInterface<T>],
        ) -> Result<Vec<T>, TypeError> {
            check_count!("input", input_types, 0, TypeError);
            Ok(Vec::new())
        }
    }

    impl<C: Domain<Type = ArrayType>> InterpretableOperation<C> for TestWhereNullaryOperation {
        fn interpret<D: crate::InterpretationDriver<C>>(
            &self,
            _context: &C,
            _driver: &D,
            inputs: &[C::Value],
        ) -> Result<Vec<C::Value>, ProgramError> {
            check_count!("input", inputs, 0, ProgramError);
            Ok(Vec::new())
        }
    }

    impl_nullary_transposable_operation!(TestWhereNullaryOperation where DataType: Type);
    impl_nullary_batchable_operation!(@replicated TestWhereNullaryOperation where DataType: Type);

    /// Generic nullary operation used to instantiate generic macro forms.
    struct TestGenericNullaryOperation<Marker>(PhantomData<fn() -> Marker>);

    impl<Marker> Clone for TestGenericNullaryOperation<Marker> {
        fn clone(&self) -> Self {
            Self(PhantomData)
        }
    }

    impl<Marker> Debug for TestGenericNullaryOperation<Marker> {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("TestGenericNullaryOperation")
        }
    }

    impl<Marker> Display for TestGenericNullaryOperation<Marker> {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("test_generic_nullary")
        }
    }

    impl<T: Type, Marker> Operation<T> for TestGenericNullaryOperation<Marker> {
        fn name(&self) -> &'static str {
            "test_generic_nullary"
        }

        fn infer_output_types(
            &self,
            input_types: &[T],
            _region_interfaces: &[crate::RegionInterface<T>],
        ) -> Result<Vec<T>, TypeError> {
            check_count!("input", input_types, 0, TypeError);
            Ok(Vec::new())
        }
    }

    impl<C: Domain<Type = ArrayType>, Marker> InterpretableOperation<C> for TestGenericNullaryOperation<Marker> {
        fn interpret<D: crate::InterpretationDriver<C>>(
            &self,
            _context: &C,
            _driver: &D,
            inputs: &[C::Value],
        ) -> Result<Vec<C::Value>, ProgramError> {
            check_count!("input", inputs, 0, ProgramError);
            Ok(Vec::new())
        }
    }

    impl_nullary_transposable_operation!(<Marker> TestGenericNullaryOperation<Marker>);
    impl_nullary_batchable_operation!(@replicated <Marker> TestGenericNullaryOperation<Marker>);

    /// Generic nullary operation used to instantiate generic-plus-`where` macro forms.
    struct TestBoundedNullaryOperation<Marker>(PhantomData<fn() -> Marker>);

    impl<Marker> Clone for TestBoundedNullaryOperation<Marker> {
        fn clone(&self) -> Self {
            Self(PhantomData)
        }
    }

    impl<Marker> Debug for TestBoundedNullaryOperation<Marker> {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("TestBoundedNullaryOperation")
        }
    }

    impl<Marker> Display for TestBoundedNullaryOperation<Marker> {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("test_bounded_nullary")
        }
    }

    impl<T: Type, Marker> Operation<T> for TestBoundedNullaryOperation<Marker> {
        fn name(&self) -> &'static str {
            "test_bounded_nullary"
        }

        fn infer_output_types(
            &self,
            input_types: &[T],
            _region_interfaces: &[crate::RegionInterface<T>],
        ) -> Result<Vec<T>, TypeError> {
            check_count!("input", input_types, 0, TypeError);
            Ok(Vec::new())
        }
    }

    impl<C: Domain<Type = ArrayType>, Marker> InterpretableOperation<C> for TestBoundedNullaryOperation<Marker> {
        fn interpret<D: crate::InterpretationDriver<C>>(
            &self,
            _context: &C,
            _driver: &D,
            inputs: &[C::Value],
        ) -> Result<Vec<C::Value>, ProgramError> {
            check_count!("input", inputs, 0, ProgramError);
            Ok(Vec::new())
        }
    }

    impl_nullary_transposable_operation!(<Marker> TestBoundedNullaryOperation<Marker> where Marker: Clone);
    impl_nullary_batchable_operation!(@replicated <Marker> TestBoundedNullaryOperation<Marker> where Marker: Clone);

    #[test]
    fn test_check_count() {
        let check_input = |values: &[usize]| -> Result<(), ProgramError> {
            check_count!("input", values, 1, ProgramError);
            Ok(())
        };
        let check_output = |values: &[usize]| -> Result<(), ProgramError> {
            check_count!("output", values, 2, ProgramError);
            Ok(())
        };
        let check_operand = |values: &[usize], expected: usize| -> Result<(), TypeError> {
            check_count!("operand", values, expected, TypeError);
            Ok(())
        };
        assert_eq!(check_input(&[0]), Ok(()));
        assert_eq!(check_input(&[]), Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }));
        assert_eq!(check_output(&[0, 1]), Ok(()));
        assert_eq!(check_output(&[0]), Err(ProgramError::InvalidOutputCount { expected: 2, actual: 1 }));
        assert_eq!(check_operand(&[0], 1), Ok(()));
        assert_eq!(check_operand(&[], 1), Err(TypeError { message: "expected 1 operand but got 0".to_string() }),);
        assert_eq!(check_operand(&[0], 2), Err(TypeError { message: "expected 2 operands but got 1".to_string() }),);
    }

    #[test]
    fn test_check_types_same() {
        let check = |expected: &[DataType], actual: Vec<DataType>| -> Result<(), TypeError> {
            check_types!(@same, "test", [expected, actual]);
            Ok(())
        };
        let expected = [DataType::F32, DataType::F64];
        assert_eq!(check(&expected, expected.to_vec()), Ok(()));
        assert_eq!(
            check(&expected, vec![DataType::F32, DataType::I64]),
            Err(TypeError {
                message: "test type signature mismatch: expected [f32, f64] but got [f32, i64]".to_string(),
            }),
        );
    }

    #[test]
    fn test_check_types_data_types() {
        let check_numeric = |types: &[DataType]| -> Result<(), TypeError> {
            check_types!(@numeric, "test", types);
            Ok(())
        };
        let check_floating_or_complex = |types: &[DataType]| -> Result<(), TypeError> {
            check_types!(@floating_or_complex, "test", types);
            Ok(())
        };
        assert_eq!(check_numeric(&[DataType::I32, DataType::F64, DataType::C128]), Ok(()));
        for r#type in [DataType::Boolean, DataType::Token, DataType::Zero] {
            assert_eq!(
                check_numeric(&[DataType::F32, r#type]),
                Err(TypeError { message: format!("'test' does not support input data type {type}", type = r#type) }),
            );
        }
        assert_eq!(check_floating_or_complex(&[DataType::BF16, DataType::F64, DataType::C64]), Ok(()));
        assert_eq!(
            check_floating_or_complex(&[DataType::I64]),
            Err(TypeError { message: "'test' does not support input data type i64".to_string() }),
        );
    }

    #[test]
    fn test_check_types_array_types() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 1, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let plain = ArrayType::scalar(DataType::F32);
        let unreduced_x = plain
            .clone()
            .with_sharding(Sharding::new(mesh.clone(), vec![]).unwrap().with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let unreduced_y = plain
            .clone()
            .with_sharding(Sharding::new(mesh.clone(), vec![]).unwrap().with_unreduced_axes(["y"]).unwrap())
            .unwrap();
        let reduced_x = plain
            .clone()
            .with_sharding(Sharding::new(mesh.clone(), vec![]).unwrap().with_reduced_axes(["x"]).unwrap())
            .unwrap();
        let reduced_y = plain
            .clone()
            .with_sharding(Sharding::new(mesh, vec![]).unwrap().with_reduced_axes(["y"]).unwrap())
            .unwrap();
        let check_no_unreduced = |types: &[ArrayType]| -> Result<(), TypeError> {
            check_types!(@no_unreduced, "test", types);
            Ok(())
        };
        let check_unreduced_axes = |types: &[ArrayType]| -> Result<(), TypeError> {
            check_types!(@same_unreduced_axes, "test", types);
            Ok(())
        };
        let check_reduced_axes = |types: &[ArrayType]| -> Result<(), TypeError> {
            check_types!(@same_reduced_axes, "test", types);
            Ok(())
        };
        assert_eq!(check_no_unreduced(std::slice::from_ref(&plain)), Ok(()));
        assert_eq!(
            check_no_unreduced(std::slice::from_ref(&unreduced_x)),
            Err(TypeError { message: "'test' does not support unreduced operands".to_string() }),
        );
        assert_eq!(check_unreduced_axes(&[unreduced_x.clone(), unreduced_x.clone()]), Ok(()));
        assert_eq!(
            check_unreduced_axes(&[unreduced_x, unreduced_y]),
            Err(TypeError { message: "'test' operands must be unreduced over the same axes".to_string() }),
        );
        assert_eq!(
            check_unreduced_axes(std::slice::from_ref(&plain)),
            Err(TypeError { message: "expected 2 inputs but got 1".to_string() }),
        );
        assert_eq!(check_reduced_axes(&[reduced_x.clone(), reduced_x.clone()]), Ok(()));
        assert_eq!(
            check_reduced_axes(&[reduced_x, reduced_y]),
            Err(TypeError { message: "'test' operands must be reduced over the same axes".to_string() }),
        );
        assert_eq!(
            check_reduced_axes(std::slice::from_ref(&plain)),
            Err(TypeError { message: "expected 2 inputs but got 1".to_string() }),
        );
    }

    #[test]
    fn test_check_sharding() {
        let expected_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let actual_mesh = LogicalMesh::new(vec![MeshAxis::new("y", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let device_mesh = DeviceMesh::new(expected_mesh.clone(), vec![Device::new(0, 0)]).unwrap();
        let matching = Sharding::new(expected_mesh.clone(), vec![]).unwrap();
        let mismatching = Sharding::new(actual_mesh.clone(), vec![]).unwrap();
        let check = |sharding: &Sharding| -> Result<(), ShardingError> {
            check_sharding!(&device_mesh, sharding);
            Ok(())
        };
        assert_eq!(check(&matching), Ok(()));
        assert_eq!(
            check(&mismatching),
            Err(ShardingError::MeshMismatch { expected: expected_mesh, actual: actual_mesh }),
        );
    }

    #[test]
    fn test_check_builders() {
        let reference = Rc::new(0);
        let same = Rc::clone(&reference);
        let different = Rc::new(0);
        assert_eq!(check_builders!(&reference, &same), Ok(()));
        assert_eq!(check_builders!(&reference, &different), Err(ProgramError::MismatchedProgramBuilders));
        assert_eq!(check_builders!(&reference, [std::iter::empty::<&Rc<i32>>()]), Ok(()));
        assert_eq!(check_builders!(&reference, [[&same, &reference].into_iter()]), Ok(()));
        assert_eq!(
            check_builders!(&reference, [[&same, &different].into_iter()]),
            Err(ProgramError::MismatchedProgramBuilders),
        );
    }

    // TODO(eaplatanios): Review this test.
    #[test]
    fn test_check_operation_batching() {
        #[derive(Clone)]
        struct TestPairOperation;

        impl Operation<ArrayType> for TestPairOperation {
            fn name(&self) -> &'static str {
                "test_pair"
            }

            fn infer_output_types(
                &self,
                input_types: &[ArrayType],
                _region_interfaces: &[crate::RegionInterface<ArrayType>],
            ) -> Result<Vec<ArrayType>, TypeError> {
                check_count!("input", input_types, 1, TypeError);
                Ok(vec![input_types[0].clone(), input_types[0].clone()])
            }
        }

        impl ElementwiseOperation for TestPairOperation {
            fn input_count(&self) -> usize {
                1
            }

            fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
                Operation::infer_output_types(self, input_types, &[])
            }
        }

        impl<C: Domain<Type = ArrayType>> InterpretableOperation<C> for TestPairOperation {
            fn interpret<D: InterpretationDriver<C>>(
                &self,
                _context: &C,
                _driver: &D,
                inputs: &[C::Value],
            ) -> Result<Vec<C::Value>, ProgramError> {
                check_count!("input", inputs, 1, ProgramError);
                Ok(vec![inputs[0].clone(), inputs[0].clone()])
            }
        }

        check_operation!(
            @batching @exact,
            operation = ZeroOperation::new(ArrayType::scalar(DataType::F64)),
            axis_size = 2,
            cases = [{
                inputs = [],
                outputs = [(@replicated, Array::scalar(0.0))],
            }],
        );

        check_operation!(
            @batching @exact,
            context = EagerContext::<Array>::new(),
            driver = &EmptyRegionDriver,
            axis_sharding = crate::ShardingDimension::Replicated,
            operation = TestPairOperation,
            axis_size = 2,
            cases = [
                {
                    inputs = [
                        (@mapped(axis = 1), Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0])),
                    ],
                    outputs = [
                        (@mapped(axis = 1), Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0])),
                        (@mapped(axis = 1), Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0])),
                    ],
                },
                {
                    inputs = [
                        (@replicated, Array::scalar(3.0)),
                    ],
                    outputs = [
                        (@replicated, Array::scalar(3.0)),
                        (@replicated, Array::scalar(3.0)),
                    ],
                },
            ],
        );

        check_operation!(
            @batching @approx(epsilon = 1e-9),
            operation = SubOperation,
            axis_size = 2,
            cases = [
                {
                    inputs = [
                        (@mapped(axis = 0), Array::vector(vec![1.0, -2.0])),
                        (@replicated, Array::scalar(3.0)),
                    ],
                    outputs = [
                        (@mapped(axis = 0), Array::vector(vec![-2.0, -5.0])),
                    ],
                },
                {
                    inputs = [
                        (@replicated, Array::scalar(3.0)),
                        (@mapped(axis = 0), Array::vector(vec![1.0, -2.0])),
                    ],
                    outputs = [
                        (@mapped(axis = 0), Array::vector(vec![2.0, 5.0])),
                    ],
                },
                {
                    inputs = [
                        (@mapped(axis = 0), Array::vector(vec![1.0, -2.0])),
                        (@mapped(axis = 0), Array::vector(vec![4.0, 1.0])),
                    ],
                    outputs = [
                        (@mapped(axis = 0), Array::vector(vec![-3.0, -3.0])),
                    ],
                },
                {
                    inputs = [
                        (@replicated, Array::scalar(3.0)),
                        (@replicated, Array::scalar(1.0)),
                    ],
                    outputs = [
                        (@replicated, Array::scalar(2.0)),
                    ],
                },
            ],
        );

        check_operation!(
            @batching @approx(epsilon = 1e-9),
            operation = SinOperation,
            axis_size = 2,
            cases = [
                {
                    inputs = [
                        (@mapped(axis = 0), Array::vector(vec![0.5, -1.0])),
                    ],
                    outputs = [
                        (@mapped(axis = 0), Array::vector(vec![0.5f64.sin(), (-1.0f64).sin()])),
                    ],
                },
            ],
        );
    }

    // TODO(eaplatanios): Review this test.
    #[test]
    fn test_check_operation_partial_evaluation() {
        check_operation!(
            @partial_evaluation @fold_and_residualize,
            operation = NegOperation,
            inputs = [Scalar::from(2.0)],
            expected = Scalar::from(-2.0),
        );

        check_operation!(
            @partial_evaluation,
            operation = AddOperation,
            cases = [
                {
                    inputs = [
                        (@known, Scalar::from(2.0)),
                        (@known, Scalar::from(3.5)),
                    ],
                    outputs = [
                        (@known, Scalar::from(5.5)),
                    ],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(type = DataType::F64, replay = Scalar::from(2.0))),
                        (@known, Scalar::from(3.5)),
                    ],
                    outputs = [
                        (@residual, Scalar::from(5.5)),
                    ],
                    residual_instructions = 1,
                },
                {
                    inputs = [
                        (@known, Scalar::from(2.0)),
                        (@unknown(type = DataType::F64, replay = Scalar::from(3.5))),
                    ],
                    outputs = [
                        (@residual, Scalar::from(5.5)),
                    ],
                    residual_instructions = 1,
                },
                {
                    inputs = [
                        (@unknown(type = DataType::F64, replay = Scalar::from(2.0))),
                        (@unknown(type = DataType::F64, replay = Scalar::from(3.5))),
                    ],
                    outputs = [
                        (@residual, Scalar::from(5.5)),
                    ],
                    residual_instructions = 1,
                },
            ],
        );
    }

    // TODO(eaplatanios): Review this test.
    #[test]
    fn test_check_operation_rejections() {
        check_operation!(
            @reject @unreduced,
            operation = SinOperation,
            input_types = [ArrayType::scalar(DataType::F64)],
        );
        check_operation!(
            @reject @mismatched_reduced,
            operation = AddOperation,
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
        check_operation!(
            @reject @transposition,
            operation = SinOperation,
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_define_elementwise_operation_unary() {
        let operation = TestUnaryOperation;
        assert_eq!(format!("{operation:?}"), "TestUnaryOperation");
        assert_eq!(format!("{operation}"), TEST_UNARY_OPERATION_NAME);
        assert_eq!(Operation::<DataType>::name(&operation), TEST_UNARY_OPERATION_NAME);
        assert_eq!(ElementwiseOperation::input_count(&operation), 1);
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F32], &[]),
            Ok(vec![DataType::F32])
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[], &[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::Boolean], &[]),
            Err(TypeError { message: "'test_unary' does not support input data type bool".to_string() }),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::I64], &[]),
            Ok(vec![DataType::I64]),
        );
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[ArrayType::scalar(DataType::F32)], &[]),
            Ok(vec![ArrayType::scalar(DataType::F32)]),
        );
        let matrix_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[matrix_type.clone()], &[]),
            Ok(vec![matrix_type]),
        );
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let unreduced_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::new(mesh, vec![]).unwrap().with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[unreduced_type], &[]),
            Err(TypeError { message: "'test_unary' does not support unreduced operands".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar, TestUnaryOperation>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(2.0f32)],
            ),
            Ok(vec![Scalar::from(-2.0f32)]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar, TestUnaryOperation>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
        let context = PartialEvaluationContext::new(EagerContext::<Scalar, TestUnaryOperation>::new());
        let outputs = operation
            .partially_evaluate(&context, &EmptyRegionDriver, &[PartialEvaluationValue::known(Scalar::from(2.0f32))])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].as_known(), Some(&Scalar::from(-2.0f32)));
    }

    #[test]
    fn test_define_elementwise_operation_binary() {
        let operation = TestBinaryOperation;
        assert_eq!(format!("{operation:?}"), "TestBinaryOperation");
        assert_eq!(format!("{operation}"), TEST_BINARY_OPERATION_NAME);
        assert_eq!(Operation::<DataType>::name(&operation), TEST_BINARY_OPERATION_NAME);
        assert_eq!(ElementwiseOperation::input_count(&operation), 2);
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F32, DataType::F64], &[]),
            Ok(vec![DataType::F64]),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F32], &[]),
            Err(TypeError { message: "expected 2 inputs but got 1".to_string() }),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::Boolean, DataType::Boolean], &[]),
            Err(TypeError { message: "'test_binary' does not support input data type bool".to_string() }),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::I64, DataType::I64], &[]),
            Ok(vec![DataType::I64]),
        );
        let scalar_type = ArrayType::scalar(DataType::F32);
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)]));
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[scalar_type, vector_type.clone()], &[]),
            Ok(vec![vector_type]),
        );
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 1, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let unreduced_x = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::new(mesh.clone(), vec![]).unwrap().with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let unreduced_y = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::new(mesh, vec![]).unwrap().with_unreduced_axes(["y"]).unwrap())
            .unwrap();
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[unreduced_x, unreduced_y], &[]),
            Err(TypeError { message: "'test_binary' operands must be unreduced over the same axes".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar, TestBinaryOperation>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(2.0f32), Scalar::from(3.0f32)],
            ),
            Ok(vec![Scalar::from(5.0f32)]),
        );
        let context = PartialEvaluationContext::new(EagerContext::<Scalar, TestBinaryOperation>::new());
        let outputs = operation
            .partially_evaluate(
                &context,
                &EmptyRegionDriver,
                &[
                    PartialEvaluationValue::known(Scalar::from(2.0f32)),
                    PartialEvaluationValue::known(Scalar::from(3.0f32)),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].as_known(), Some(&Scalar::from(5.0f32)));
    }

    #[test]
    fn test_define_elementwise_capability_unary() {
        let context = TracingContext::<Scalar, TestUnaryOperation>::new();
        let output = context.input(DataType::F32).test_unary().unwrap();
        let builder = output.builder().borrow();
        assert_eq!(builder.instructions().len(), 1);
        assert_eq!(Operation::<DataType>::name(builder.instructions()[0].operation()), TEST_UNARY_OPERATION_NAME);
        assert_eq!(builder.instructions()[0].inputs().len(), 1);
        assert_eq!(builder.instructions()[0].outputs(), &[output.atom_id().unwrap()]);
    }

    #[test]
    fn test_define_elementwise_capability_binary() {
        let context = TracingContext::<Scalar, TestBinaryOperation>::new();
        let left = context.input(DataType::F32);
        let right = context.input(DataType::F32);
        let output = left.test_binary(&right).unwrap();
        let builder = output.builder().borrow();
        assert_eq!(builder.instructions().len(), 1);
        assert_eq!(Operation::<DataType>::name(builder.instructions()[0].operation()), TEST_BINARY_OPERATION_NAME);
        assert_eq!(builder.instructions()[0].inputs(), &[left.atom_id().unwrap(), right.atom_id().unwrap()]);
        assert_eq!(builder.instructions()[0].outputs(), &[output.atom_id().unwrap()]);
    }

    #[test]
    fn test_impl_non_differentiable_operation() {
        let inputs = [DifferentiationDual::new(Scalar::from(2.0f32), Scalar::from(1.0f32)).unwrap()];
        let outputs = TestUnaryOperation
            .jvp(&EagerContext::<Scalar, TestUnaryOperation>::new(), &EmptyRegionDriver, &inputs)
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &Scalar::from(-2.0f32));
        assert!(matches!(outputs[0].tangent(), MaybeZero::Zero(DataType::F32)));
        let context = TracingContext::<Scalar, TestUnaryOperation>::new();
        let primal = context.input(DataType::F32);
        let tangent = context.input(DataType::F32);
        let inputs = [DifferentiationDual::new(primal, tangent).unwrap()];
        let outputs = TestUnaryOperation.jvp(&context, &EmptyRegionDriver, &inputs).unwrap();
        assert_eq!(context.builder().borrow().instructions().len(), 1);
        assert!(matches!(outputs[0].tangent(), MaybeZero::Zero(DataType::F32)));
    }

    #[test]
    fn test_impl_non_transposable_operation() {
        let mut context = TracingContext::<Scalar, TestUnaryOperation>::new();
        let inputs: [PartialValue<Tracer<TracingContext<Scalar, TestUnaryOperation>>>; 0] = [];
        let outputs: [MaybeZero<Tracer<TracingContext<Scalar, TestUnaryOperation>>>; 0] = [];
        assert!(matches!(
            <TestUnaryOperation as TransposableOperation<Scalar, TestUnaryOperation>>::transpose(
                &TestUnaryOperation,
                &mut context,
                &EmptyRegionDriver,
                &inputs,
                &outputs,
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "operation `test_unary` is not transposable",
        ));
    }

    #[test]
    fn test_impl_nullary_transposable_operation() {
        let mut context = TracingContext::<Scalar, TestNullaryOperation>::new();
        let inputs: [PartialValue<Tracer<TracingContext<Scalar, TestNullaryOperation>>>; 0] = [];
        let outputs = [MaybeZero::Zero(DataType::F64), MaybeZero::Zero(DataType::F64)];
        let result = <TestNullaryOperation as TransposableOperation<Scalar, TestNullaryOperation>>::transpose(
            &TestNullaryOperation,
            &mut context,
            &EmptyRegionDriver,
            &inputs,
            &outputs,
        )
        .unwrap();
        assert!(result.is_empty());
        let input = context.input(DataType::F64);
        assert!(matches!(
            <TestNullaryOperation as TransposableOperation<Scalar, TestNullaryOperation>>::transpose(
                &TestNullaryOperation,
                &mut context,
                &EmptyRegionDriver,
                &[PartialValue::Known(input)],
                &outputs,
            ),
            Err(DifferentiationError::Program(ProgramError::InvalidInputCount { expected: 0, actual: 1 })),
        ));
        assert!(matches!(
            <TestNullaryOperation as TransposableOperation<Scalar, TestNullaryOperation>>::transpose(
                &TestNullaryOperation,
                &mut context,
                &EmptyRegionDriver,
                &inputs,
                &[],
            ),
            Err(DifferentiationError::Program(ProgramError::InvalidOutputCount { expected: 2, actual: 0 })),
        ));

        fn assert_transposable<O: Operation<DataType> + TransposableOperation<Scalar, O>>() {}

        assert_transposable::<TestWhereNullaryOperation>();
        assert_transposable::<TestGenericNullaryOperation<()>>();
        assert_transposable::<TestBoundedNullaryOperation<()>>();
    }

    #[test]
    fn test_impl_nullary_batchable_operation() {
        let context = BatchingContext::new(EagerContext::<Array, TestNullaryOperation>::new(), 2);
        let outputs = <TestNullaryOperation as BatchableOperation<EagerContext<Array, TestNullaryOperation>>>::batch(
            &TestNullaryOperation,
            &context,
            &EmptyRegionDriver,
            &[],
        )
        .unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].value(), &Array::scalar(3.0));
        assert_eq!(outputs[1].value(), &Array::scalar(4.0));
        assert!(outputs[0].batch_axis().is_replicated());
        assert!(outputs[1].batch_axis().is_replicated());
        assert!(matches!(
            <TestNullaryOperation as BatchableOperation<EagerContext<Array, TestNullaryOperation>>>::batch(
                &TestNullaryOperation,
                &context,
                &EmptyRegionDriver,
                &[ArrayBatch::replicated(Array::scalar(1.0))],
            ),
            Err(crate::BatchingError::Program(ProgramError::InvalidInputCount { expected: 0, actual: 1 })),
        ));

        fn assert_batchable<O>()
        where
            O: Operation<ArrayType>
                + InterpretableOperation<EagerContext<Array, O>>
                + BatchableOperation<EagerContext<Array, O>>,
        {
        }

        assert_batchable::<TestWhereNullaryOperation>();
        assert_batchable::<TestGenericNullaryOperation<()>>();
        assert_batchable::<TestBoundedNullaryOperation<()>>();
    }

    #[test]
    fn test_define_tracer_operator_unary() {
        let context = TracingContext::<Scalar, TestUnaryOperation>::new();
        let input = context.input(DataType::F32);
        let input_id = input.atom_id().unwrap();
        let output = input.apply_unary();
        let builder = output.builder().borrow();
        assert_eq!(builder.instructions().len(), 1);
        assert_eq!(Operation::<DataType>::name(builder.instructions()[0].operation()), TEST_UNARY_OPERATION_NAME);
        assert_eq!(builder.instructions()[0].inputs(), &[input_id]);
        drop(builder);

        let context = PartialEvaluationContext::new(EagerContext::<Scalar, TestUnaryOperation>::new());
        let input = PartialTracer::new(context, PartialEvaluationValue::known(Scalar::from(2.0f32)));
        assert_eq!(input.apply_unary().into_value().unwrap().as_known(), Some(&Scalar::from(-2.0f32)));

        let context = BatchingContext::new(EagerContext::<Array, TestUnaryOperation>::new(), 2);
        let input = BatchingTracer::new(context, ArrayBatch::replicated(Array::scalar(2.0f32)));
        let output = input.apply_unary().into_batch();
        assert_eq!(output.value(), &Array::scalar(-2.0f32));
        assert!(output.batch_axis().is_replicated());

        let context = DifferentiationContext::new(EagerContext::<Scalar, TestUnaryOperation>::new());
        let input = DifferentiationTracer::new(
            DifferentiationDual::new(Scalar::from(2.0f32), Scalar::from(1.0f32)).unwrap(),
            context,
        );
        let output = input.apply_unary().into_dual();
        assert_eq!(output.primal(), &Scalar::from(-2.0f32));
        assert!(matches!(output.tangent(), MaybeZero::Zero(DataType::F32)));
    }

    #[test]
    fn test_define_tracer_operator_binary() {
        let context = TracingContext::<Scalar, TestBinaryOperation>::new();
        let left = context.input(DataType::F32);
        let right = context.input(DataType::F32);
        let input_ids = [left.atom_id().unwrap(), right.atom_id().unwrap()];
        let output = left.apply_binary(right);
        let builder = output.builder().borrow();
        assert_eq!(builder.instructions().len(), 1);
        assert_eq!(Operation::<DataType>::name(builder.instructions()[0].operation()), TEST_BINARY_OPERATION_NAME);
        assert_eq!(builder.instructions()[0].inputs(), &input_ids);
        drop(builder);

        let context = PartialEvaluationContext::new(EagerContext::<Scalar, TestBinaryOperation>::new());
        let left = PartialTracer::new(context.clone(), PartialEvaluationValue::known(Scalar::from(2.0f32)));
        let right = PartialTracer::new(context, PartialEvaluationValue::known(Scalar::from(3.0f32)));
        assert_eq!(left.apply_binary(right).into_value().unwrap().as_known(), Some(&Scalar::from(5.0f32)),);

        let context = BatchingContext::new(EagerContext::<Array, TestBinaryOperation>::new(), 2);
        let left = BatchingTracer::new(context.clone(), ArrayBatch::replicated(Array::scalar(2.0f32)));
        let right = BatchingTracer::new(context, ArrayBatch::replicated(Array::scalar(3.0f32)));
        let output = left.apply_binary(right).into_batch();
        assert_eq!(output.value(), &Array::scalar(5.0f32));
        assert!(output.batch_axis().is_replicated());

        let context = DifferentiationContext::new(EagerContext::<Scalar, TestBinaryOperation>::new());
        let left = DifferentiationTracer::new(
            DifferentiationDual::new(Scalar::from(2.0f32), Scalar::from(1.0f32)).unwrap(),
            context.clone(),
        );
        let right = DifferentiationTracer::new(
            DifferentiationDual::new(Scalar::from(3.0f32), Scalar::from(1.0f32)).unwrap(),
            context,
        );
        let output = left.apply_binary(right).into_dual();
        assert_eq!(output.primal(), &Scalar::from(5.0f32));
        assert!(matches!(output.tangent(), MaybeZero::Zero(DataType::F32)));
    }

    #[test]
    fn test_check_gradient_scalar() {
        fn square<V: Clone + std::ops::Mul<Output = V>>(input: V) -> V {
            input.clone() * input
        }

        check_gradient!(@scalar, square, at = 0.7, step = 1e-6, tolerance = 1e-6);
        check_gradient!(
            @scalar,
            |input| input.abs(),
            at = Complex::new(0.7f64, -0.3),
            step = 1e-6,
            tolerance = 1e-6,
        );
    }

    #[test]
    fn test_check_gradient_array() {
        fn square<V: Clone + std::ops::Mul<Output = V>>(input: V) -> V {
            input.clone() * input
        }

        check_gradient!(
            @array,
            |input| square(input).reduce(&[0], ReductionKind::Sum),
            at = Array::vector(vec![0.7f64, -1.3, 2.1]),
            step = 1e-6,
            tolerance = 1e-6,
        );
        check_gradient!(
            @array,
            |input| input.abs().map(|magnitudes| magnitudes.reduce(&[0], ReductionKind::Sum)),
            at = Array::vector(vec![Complex::new(0.7f64, -0.3), Complex::new(-1.2f64, 0.8)]),
            step = 1e-6,
            tolerance = 1e-6,
        );
    }

    #[test]
    #[should_panic(expected = "finite-difference gradient checking requires an f64 or c128 input but got f32")]
    fn test_check_gradient_scalar_unsupported_input_type() {
        check_gradient!(@scalar, |input| input, at = 0.7f32, step = 1e-3, tolerance = 1e-3);
    }

    #[test]
    #[should_panic(expected = "finite-difference gradient checking requires an f64 or c128 input but got f32")]
    fn test_check_gradient_array_unsupported_input_type() {
        check_gradient!(
            @array,
            |input| input.reduce(&[0], ReductionKind::Sum),
            at = Array::vector(vec![0.7f32, -1.3]),
            step = 1e-3,
            tolerance = 1e-3,
        );
    }
}
