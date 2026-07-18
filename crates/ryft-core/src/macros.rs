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
///   - `check_data_types = [@selector, ...]`: Optional ordered list of [`check_types!`](crate::check_types)
///     selectors applied to scalar input types and array element types before type inference.
///   - `check_array_types = [@selector, ...]`: Optional ordered list of [`check_types!`](crate::check_types) selectors
///     applied to array input types before array broadcasting.
///   - `validate_data_types = $data_type_validator`: Optional hook that validates scalar [`DataType`](crate::DataType)
///     inputs before type inference and array element [`DataType`](crate::DataType)s before array broadcasting when a
///     reusable [`check_types!`](crate::check_types) selector does not express the required contract.
///   - `validate_array_types = $array_type_validator`: Optional hook that validates [`ArrayType`](crate::ArrayType)
///     inputs before array broadcasting when a reusable [`check_types!`](crate::check_types) selector does not express
///     the required contract.
#[macro_export]
macro_rules! define_elementwise_operation {
    (
        @unary
        $(#[$documentation:meta])*
        $operation:ident, $name:ident,
        $capability:ident, $method:ident
        $(, check_data_types = [$(@$data_type_check:ident),* $(,)?])?
        $(, check_array_types = [$(@$array_type_check:ident),* $(,)?])?
        $(, validate_data_types = $data_type_validator:path)?
        $(, validate_array_types = $array_type_validator:path)? $(,)?
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
                $($data_type_validator(input_types, $name)?;)?
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
                $($data_type_validator(&[input_types[0].data_type()], $name)?;)?
                $($($crate::check_types!(@$array_type_check, $name, input_types);)*)?
                $($array_type_validator(input_types, $name)?;)?
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
        $(, check_array_types = [$(@$array_type_check:ident),* $(,)?])?
        $(, validate_data_types = $data_type_validator:path)?
        $(, validate_array_types = $array_type_validator:path)? $(,)?
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
                $($data_type_validator(input_types, $name)?;)?
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
                $($data_type_validator(&[input_types[0].data_type(), input_types[1].data_type()], $name)?;)?
                $($($crate::check_types!(@$array_type_check, $name, input_types);)*)?
                $($array_type_validator(input_types, $name)?;)?
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
    check_builders, check_count, check_gradient, check_sharding, check_types, define_elementwise_capability,
    define_elementwise_operation, define_tracer_operator, impl_non_differentiable_operation,
    impl_non_transposable_operation, impl_nullary_batchable_operation, impl_nullary_transposable_operation,
};

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use crate::programs::ProgramError;
    use crate::programs::types::TypeError;
    use crate::sharding::{Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingError};
    use crate::types::{ArrayType, DataType};

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
}
