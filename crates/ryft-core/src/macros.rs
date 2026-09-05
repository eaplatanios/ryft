/// Checks that `values` contains exactly `expected` entries and, if not, returns an error of the specified type.
#[macro_export]
macro_rules! check_count {
    // This branch reports a program input-count mismatch with the dedicated input error variant.
    ("input", $values:expr, $expected:expr, ProgramError $(,)?) => {{
        let values = &$values;
        let expected = $expected;
        if values.len() != expected {
            return Err($crate::ProgramError::InvalidInputCount { expected, actual: values.len() }.into());
        }
    }};

    // This branch reports a program output-count mismatch with the dedicated output error variant.
    ("output", $values:expr, $expected:expr, ProgramError $(,)?) => {{
        let values = &$values;
        let expected = $expected;
        if values.len() != expected {
            return Err($crate::ProgramError::InvalidOutputCount { expected, actual: values.len() }.into());
        }
    }};

    // This branch reports a type-level count mismatch using the caller's singular descriptor.
    ($descriptor:expr, $values:expr, $expected:expr, TypeError $(,)?) => {{
        let values = &$values;
        let expected = $expected;
        if values.len() != expected {
            let count = values.len();
            let descriptor = $descriptor;
            let noun = if expected == 1 { descriptor.to_string() } else { format!("{descriptor}s") };
            return Err($crate::TypeError::invalid(format!("expected {expected} {noun} but got {count}")));
        }
    }};
}

/// Checks types against a structural or semantic type contract. All forms use an `@` selector and return
/// [`TypeError`](crate::TypeError)s as appropriate, converted into the enclosing function's error type, when the
/// selected contract is not satisfied. Data-type selectors compose by intersection when written next to one another.
/// For example, `@numeric @real` accepts real numeric types, while `@float @real` accepts real floating-point types.
/// The available selectors are:
///
///   - `@same`: Requires the provided expected and actual flat type signatures to be identical.
///   - `@numeric`: Accepts integer, floating-point, and complex [`DataType`](crate::DataType)s.
///   - `@float`: Accepts floating-point and complex [`DataType`](crate::DataType)s.
///   - `@real`: Excludes complex [`DataType`](crate::DataType)s and is intended to refine `@numeric` or `@float`.
///   - `@no_unreduced`: Rejects [`ArrayType`](crate::ArrayType)s carrying any unreduced mesh axes.
///   - `@same_unreduced_axes`: Requires exactly two [`ArrayType`](crate::ArrayType)s with matching unreduced-axis sets.
///   - `@same_reduced_axes`: Requires exactly two [`ArrayType`](crate::ArrayType)s with matching reduced-axis sets.
///
/// # Examples
///
/// Compose selectors to express the intersection of their contracts. Selector order does not affect the accepted types,
/// so both invocations below accept real numeric data types and reject Boolean, token, zero-space, and complex types:
///
/// ```rust,ignore
/// check_types!(@numeric @real, "max", input_types);
/// check_types!(@real @numeric, "max", input_types);
/// ```
///
/// # Parameters
///
///   - `$selectors`: One structural selector, or one or more composable [`DataType`](crate::DataType) selectors
///     identifying the contract to validate.
///   - `$descriptor`: Expression evaluating to a string that identifies the checked operation or signature in errors.
///   - `$types`: Expression evaluating to the data or array types checked by `$selector`.
///   - `$signatures`: Bracketed pair containing the expected and actual flat type signatures checked by `@same`.
#[macro_export]
macro_rules! check_types {
    // This branch checks exact equality between two complete flat type signatures.
    (@same, $descriptor:expr, [$expected:expr, $actual:expr $(,)?] $(,)?) => {{
        let expected = &$expected[..];
        let actual = &$actual[..];
        if expected != actual {
            return Err($crate::TypeError::invalid(format!(
                    "{} type signature mismatch: expected [{}] but got [{}]",
                    $descriptor,
                    expected.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
                    actual.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
                )));
        }
    }};

    // This branch rejects array types that still carry unreduced mesh axes.
    (@no_unreduced, $descriptor:expr, $types:expr $(,)?) => {{
        let descriptor = $descriptor;
        let types = &$types[..];
        if types.iter().any(|r#type| !r#type.unreduced_axes().is_empty()) {
            return Err(
                $crate::TypeError::invalid(format!("`{descriptor}` does not support unreduced operands")).into()
            );
        }
    }};

    // This branch requires a binary operation's operands to carry identical unreduced-axis sets.
    (@same_unreduced_axes, $descriptor:expr, $types:expr $(,)?) => {{
        let descriptor = $descriptor;
        let types = &$types[..];
        if types.len() != 2 {
            return Err($crate::TypeError::invalid(format!("expected 2 inputs but got {}", types.len())).into());
        }
        if types[0].unreduced_axes() != types[1].unreduced_axes() {
            return Err($crate::TypeError::invalid(format!("`{descriptor}` operands must be unreduced over the same axes"))
            .into());
        }
    }};

    // This branch requires a binary operation's operands to carry identical reduced-axis sets.
    (@same_reduced_axes, $descriptor:expr, $types:expr $(,)?) => {{
        let descriptor = $descriptor;
        let types = &$types[..];
        if types.len() != 2 {
            return Err($crate::TypeError::invalid(format!("expected 2 inputs but got {}", types.len())).into());
        }
        if types[0].reduced_axes() != types[1].reduced_axes() {
            return Err($crate::TypeError::invalid(format!("`{descriptor}` operands must be reduced over the same axes"))
            .into());
        }
    }};

    // This branch applies one or more composable data-type predicates to every input type.
    ($(@$selector:ident)+, $descriptor:expr, $types:expr $(,)?) => {{
        let descriptor = $descriptor;
        let types = &$types[..];
        if let Some(input_type) = types.iter().find(|input_type| {
            !$crate::check_types!(@matches_data_type input_type; $(@$selector)+)
        }) {
            return Err($crate::TypeError::invalid(format!("`{descriptor}` does not support input data type {input_type}"))
            .into());
        }
    }};

    // This internal helper terminates a composed data-type contract after every predicate has accepted the candidate.
    // It supplies the `true` identity needed to combine an arbitrary number of selectors with logical conjunction.
    (@matches_data_type $input_type:ident;) => {
        true
    };

    // This internal helper accepts the numeric universe: signed and unsigned integers, real floating-point values,
    // and complex values. It recurses so later selectors can refine that universe without duplicating its variant list.
    (@matches_data_type $input_type:ident; @numeric $($selectors:tt)*) => {
        $input_type.is_numeric() && $crate::check_types!(@matches_data_type $input_type; $($selectors)*)
    };

    // This internal helper accepts real floating-point and complex types as one float-capable universe. Keeping this
    // predicate independent from `@real` lets callers retain complex values with `@float` or exclude them by composing
    // `@float @real`.
    (@matches_data_type $input_type:ident; @float $($selectors:tt)*) => {
        ($input_type.is_floating_point() || $input_type.is_complex())
            && $crate::check_types!(@matches_data_type $input_type; $($selectors)*)
    };

    // This internal helper excludes complex types from the universe established by preceding or following selectors.
    // Its independent predicate makes selector order irrelevant and supports both `@numeric @real` and `@float @real`
    // without compound selector names.
    (@matches_data_type $input_type:ident; @real $($selectors:tt)*) => {
        !$input_type.is_complex()
            && $crate::check_types!(@matches_data_type $input_type; $($selectors)*)
    };
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
    // This branch compares the runtime mesh with the sharding's logical mesh and returns the canonical mismatch error.
    ($mesh:expr, $sharding:expr $(,)?) => {{
        let mesh = &$mesh;
        let sharding = &$sharding;
        if mesh.logical_mesh() != sharding.mesh() {
            return Err($crate::arrays::ShardingError::MeshMismatch {
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
    // This branch checks every builder handle yielded by a caller-provided collection.
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

    // This branch checks one builder handle directly against the reference handle.
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

// TODO(eaplatanios): Review this.
/// Defines a nominal binary dimension-arithmetic operation.
///
/// The generated operation stores the two declared operand types and the name and bounds needed to infer one fresh
/// result identity. The resulting program atom owns that inferred identity; the operation does not duplicate it. The
/// generated type implements [`Operation`](crate::Operation),
/// [`ArithmeticDimensionOperation`](crate::ArithmeticDimensionOperation), [`Display`](std::fmt::Display),
/// identity renaming, capability-based interpretation, and ordinary partial evaluation.
///
/// The caller supplies the operation's public documentation and name, its value-level capability and semantic method,
/// a diagnostic result-name expression, and a bounds-transfer expression. This keeps operation structure and
/// interpretation centralized while leaving concrete value semantics in backend capability implementations.
///
/// # Examples
///
/// ```rust,ignore
/// define_arithmetic_dimension_operation!(
///     /// Checked dimension-addition operation used by [`Add`].
///     DimensionAddOperation,
///     DIMENSION_ADD_OPERATION_NAME,
///     Add,
///     add,
///     result_name = |left: &DimensionType, right: &DimensionType| {
///         format!("{} + {}", left.variable(), right.variable())
///     },
///     infer_bounds = infer_add_bounds,
/// );
/// ```
///
/// # Parameters
///
///   - `$(#[$documentation])*`: Documentation attributes attached to the generated operation type.
///   - `$operation`: Identifier of the generated nominal operation type (e.g., `DimensionAddOperation`).
///   - `$name`: Identifier of an existing operation-name constant (e.g., `DIMENSION_ADD_OPERATION_NAME`).
///   - `$capability`: Value-level capability required by the generated
///     [`InterpretableOperation`](crate::InterpretableOperation) implementation (e.g., `Add`).
///   - `$method`: Semantic capability method used for interpretation (e.g., `add`).
///   - `$result_name`: Expression accepting the left and right [`DimensionType`](crate::DimensionType)s and returning
///     the fresh result identity's diagnostic name.
///   - `$infer_bounds`: Expression accepting the left and right [`DimensionType`](crate::DimensionType)s and returning
///     a `Result<(DimensionBounds, bool), DimensionError>`. The Boolean reports whether their bounds leave a checked
///     runtime failure possible.
#[macro_export]
macro_rules! define_arithmetic_dimension_operation {
    // This public branch defines one nominal binary dimension primitive and its shared operation machinery.
    (
        $(#[$documentation:meta])*
        $operation:ident, $name:ident,
        $capability:ident, $method:ident,
        result_name = $result_name:expr,
        infer_bounds = $infer_bounds:expr $(,)?
    ) => {
        $(#[$documentation])*
        #[derive(Clone, Debug, PartialEq, Eq, Hash, ryft_macros::Parameter)]
        pub struct $operation {
            /// Shared operand contract and result-inference metadata for this arithmetic dimension operation.
            metadata: $crate::operations::dimensions::ArithmeticDimensionOperationMetadata,
        }

        impl $operation {
            /// Creates a new operation and derives its fresh bounded result dimension.
            pub fn new(
                left: &$crate::arrays::DimensionType,
                right: &$crate::arrays::DimensionType,
            ) -> Result<Self, $crate::arrays::DimensionError> {
                let result_name = ($result_name)(left, right);
                let (result_bounds, requires_runtime_assertion) = ($infer_bounds)(left, right)?;
                Ok(Self {
                    metadata: $crate::operations::dimensions::ArithmeticDimensionOperationMetadata::new(
                        left,
                        right,
                        result_name,
                        result_bounds,
                        requires_runtime_assertion,
                    ),
                })
            }

            /// Returns the expected left operand [`DimensionType`](crate::DimensionType).
            #[inline]
            pub fn left_type(&self) -> &$crate::arrays::DimensionType {
                self.metadata.left_type()
            }

            /// Returns the expected right operand [`DimensionType`](crate::DimensionType).
            #[inline]
            pub fn right_type(&self) -> &$crate::arrays::DimensionType {
                self.metadata.right_type()
            }

            /// Returns the diagnostic name used for a freshly inferred result variable.
            #[inline]
            pub fn result_name(&self) -> &str {
                self.metadata.result_name()
            }

            /// Returns the bounds of a freshly inferred result variable.
            #[inline]
            pub fn result_bounds(&self) -> $crate::arrays::DimensionBounds {
                self.metadata.result_bounds()
            }
        }

        impl ::std::fmt::Display for $operation {
            #[inline]
            fn fmt(&self, formatter: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                <$operation as $crate::programs::operations::Operation>::render(
                    self,
                    formatter,
                    0,
                )
            }
        }

        impl $crate::programs::operations::Operation for $operation {
            type Type = $crate::arrays::DimensionType;

            #[inline]
            fn name(&self) -> &'static str {
                $name
            }

            #[inline]
            fn infer_output_types(
                &self,
                input_types: &[$crate::arrays::DimensionType],
                _region_interfaces: &[$crate::programs::regions::RegionInterface<$crate::arrays::DimensionType>],
            ) -> Result<Vec<$crate::arrays::DimensionType>, $crate::programs::types::TypeError> {
                $crate::operations::dimensions::ArithmeticDimensionOperation::infer_output_types(self, input_types)
            }

            #[inline]
            fn effects(&self) -> ::std::borrow::Cow<'_, $crate::programs::effects::Effects> {
                if self.metadata.requires_runtime_assertion() {
                    ::std::borrow::Cow::Owned($crate::programs::effects::Effects::explicit(
                        $crate::programs::effects::EffectClasses::single($crate::programs::effects::EffectClass::OrderedAssertion),
                    ))
                } else {
                    ::std::borrow::Cow::Borrowed($crate::programs::effects::Effects::empty())
                }
            }

            #[inline]
            fn rename_type_identities(
                &self,
                renaming: &$crate::programs::identities::TypeIdentityRenaming<$crate::arrays::DimensionVariable>,
            ) -> Result<Self, $crate::programs::types::TypeError> {
                Ok(Self { metadata: self.metadata.rename_type_identities(renaming)? })
            }

            #[inline]
            fn render(
                &self,
                formatter: &mut ::std::fmt::Formatter<'_>,
                indentation: usize,
            ) -> ::std::fmt::Result {
                // The result name and bounds are recoverable from the instruction's rendered output atom type, and each
                // declared operand type is pinned by the input atom type that must refine it, so the only payload field
                // this rendering must carry is the runtime-assertion classification, which is invisible to the types
                // and decides this operation's effects. It is elided when it is `false`.
                let operation =
                    $crate::programs::operations::OperationFormatter::new(formatter, indentation, $name)?;
                if !self.metadata.requires_runtime_assertion() {
                    return Ok(());
                }
                operation.bracketed(|operation| operation.field("requires_runtime_assertion", true))
            }
        }

        impl $crate::operations::dimensions::ArithmeticDimensionOperation for $operation {
            #[inline]
            fn left_type(&self) -> &$crate::arrays::DimensionType {
                $operation::left_type(self)
            }

            #[inline]
            fn right_type(&self) -> &$crate::arrays::DimensionType {
                $operation::right_type(self)
            }

            #[inline]
            fn result_name(&self) -> &str {
                $operation::result_name(self)
            }

            #[inline]
            fn result_bounds(&self) -> $crate::arrays::DimensionBounds {
                $operation::result_bounds(self)
            }
        }

        impl<
            __C: $crate::contexts::Domain<
                Type = $crate::arrays::DimensionType,
                Value: $capability,
            >,
        > $crate::interpretation::InterpretableOperation<__C> for $operation
        {
            #[inline]
            fn interpret<__D: $crate::interpretation::InterpretationDriver<__C>>(
                &self,
                _context: &__C,
                _driver: &__D,
                inputs: &[__C::Value],
            ) -> Result<Vec<__C::Value>, $crate::programs::ProgramError> {
                $crate::check_count!("input", inputs, 2, ProgramError);
                $crate::programs::operations::Operation::infer_output_types(
                    self,
                    &[
                        $crate::programs::types::Typed::r#type(&inputs[0]).into_owned(),
                        $crate::programs::types::Typed::r#type(&inputs[1]).into_owned(),
                    ],
                    &[],
                )?;
                Ok(vec![inputs[0].$method(&inputs[1])?])
            }
        }

        impl<
            __C: $crate::contexts::Context<
                Type = $crate::arrays::DimensionType,
                Operation: ::std::convert::From<Self>,
            >,
        > $crate::partial::PartiallyEvaluatableOperation<__C> for $operation
        {
        }
    };
}

// TODO(eaplatanios): Review this.
/// Defines a value-level capability for a binary dimension-arithmetic operation.
///
/// The generated trait exposes one semantic binary method. Its blanket implementation constructs and stages the
/// corresponding operation through a context-carrying value's dispatch domain; concrete eager values provide
/// backend-owned implementations.
///
/// # Examples
///
/// ```rust,ignore
/// define_arithmetic_dimension_capability!(
///     /// Returns the maximum of two first-class runtime dimensions.
///     DimensionMax,
///     /// Returns the maximum of `self` and `right`.
///     dimension_max(right),
///     DimensionMaxOperation,
/// );
/// ```
///
/// # Parameters
///
///   - `$(#[$capability_documentation])*`: Documentation attributes attached to the generated capability trait.
///   - `$capability`: Identifier of the generated value-level capability trait (e.g., `DimensionMax`).
///   - `$(#[$method_documentation])*`: Documentation attributes attached to the generated capability method.
///   - `$method`: Identifier of the generated binary capability method (e.g., `dimension_max`).
///   - `$argument`: Name of the capability method's non-receiver argument (e.g., `right`).
///   - `$operation`: Dimension-arithmetic operation constructed and bound by the generated implementation (e.g.,
///     `DimensionMaxOperation`).
#[macro_export]
macro_rules! define_arithmetic_dimension_capability {
    // This branch defines one semantic binary method plus its generic staging implementation.
    (
        $(#[$capability_documentation:meta])*
        $capability:ident,
        $(#[$method_documentation:meta])+
        $method:ident($argument:ident),
        $operation:ident $(,)?
    ) => {
        $(#[$capability_documentation])*
        pub trait $capability: $crate::programs::types::Typed<Type = $crate::arrays::DimensionType> + Sized {
            $(#[$method_documentation])*
            fn $method(&self, $argument: &Self) -> Result<Self, $crate::programs::ProgramError>;
        }

        impl<__V: $crate::programs::values::Value<Type = $crate::arrays::DimensionType>> $capability for __V
        where
            __V::DispatchDomain: $crate::contexts::Context<Type = $crate::arrays::DimensionType>,
            <__V::DispatchDomain as $crate::contexts::Domain>::Operation: ::std::convert::From<$operation>,
        {
            #[inline]
            fn $method(&self, $argument: &Self) -> Result<Self, $crate::programs::ProgramError> {
                let left_type = $crate::programs::types::Typed::r#type(self);
                let right_type = $crate::programs::types::Typed::r#type($argument);
                let operation = $operation::new(left_type.as_ref(), right_type.as_ref())?;
                Ok($crate::contexts::Context::bind(
                    &$crate::programs::values::Value::dispatch_domain(self),
                    operation,
                    Vec::new(),
                    &[self.clone(), $argument.clone()],
                )?
                .remove(0))
            }
        }
    };
}

/// Defines the structural implementations shared by elementwise operations. The generated base includes a zero-sized
/// operation marker parameterized by its [`Type`](crate::Type) universe, its [`Display`](std::fmt::Display),
/// [`Operation`](crate::Operation), [`ElementwiseOperation`](crate::ElementwiseOperation),
/// [`InterpretableOperation`](crate::InterpretableOperation), and
/// [`PartiallyEvaluatableOperation`](crate::PartiallyEvaluatableOperation) implementations.
///
/// # Examples
///
/// An ordinary unary operation declares its type constraints and uses the default type-preserving inference:
///
/// ```rust,ignore
/// define_elementwise_operation!(
///     @unary
///     /// Elementwise sine operation.
///     SinOperation, SIN_OPERATION_NAME,
///     Sin, sin,
///     check_data_types = [@float],
///     check_array_types = [@no_unreduced],
/// );
/// ```
///
/// An operation whose result element type differs from its input can provide a data-type inference closure. The
/// generated array inference broadcasts the array structure and applies that same closure to its element data types.
/// Named functions remain supported because function paths are expressions. Operations with genuinely custom array
/// metadata semantics can additionally provide `infer_array_types`:
///
/// ```rust,ignore
/// define_elementwise_operation!(
///     @unary
///     /// Elementwise absolute-value operation.
///     AbsOperation, ABS_OPERATION_NAME,
///     Abs, abs,
///     infer_data_types = |input_types: &[DataType]| {
///         Ok(vec![match input_types[0] {
///             DataType::C64 => DataType::F32,
///             DataType::C128 => DataType::F64,
///             input_type => input_type,
///         }])
///     },
///     check_array_types = [@no_unreduced],
/// );
/// ```
///
/// # Parameters
///
///   - `@unary` / `@binary`: Selects the operation arity.
///   - `$(#[$documentation])*`: Documentation attributes attached to the generated operation struct.
///   - `$operation`: Identifier of the generated type-parameterized operation marker (e.g., `SinOperation`).
///   - `$name`: Identifier of an existing operation-name constant (e.g., `SIN_OPERATION_NAME`).
///   - `$capability`: Identifier of the value-level capability trait bound by the generated
///     [`InterpretableOperation`](crate::InterpretableOperation) implementation (e.g., `Sin`).
///   - `$method`: Identifier of the capability trait method used for interpretation (e.g., `sin`).
///   - `infer_data_types`: Optional callable expression with signature `Fn(&[DataType]) -> Result<Vec<DataType>,
///     TypeError>`. It receives the operation's validated input data types and returns its single output data type.
///     Both closures and function paths are accepted. When omitted, unary operations preserve their input data type
///     and binary operations use ordinary data-type broadcasting.
///   - `infer_array_types`: Optional callable expression with signature `Fn(&[ArrayType]) -> Result<Vec<ArrayType>,
///     TypeError>`. It receives the validated input array types and returns the single output array type. Both closures
///     and function paths are accepted. When omitted, the macro broadcasts the input array structure and applies
///     `infer_data_types` to the input element data types.
///   - `check_data_types = [@selector ..., ...]`: Optional ordered list of [`check_types!`] data-type contracts applied
///     to scalar input types and array element types before type inference. Space-separated selectors compose one
///     contract, while commas separate independently checked contracts.
///   - `check_array_types = [@selector, ...]`: Optional ordered list of [`check_types!`] selectors applied to array
///     input types before array broadcasting.
#[macro_export]
macro_rules! define_elementwise_operation {
    // This public branch defines a unary elementwise operation and its shared interpretation and inference machinery.
    (
        @unary
        $(#[$documentation:meta])*
        $operation:ident, $name:ident,
        $capability:ident, $method:ident
        $(, infer_data_types = $infer_data_types:expr)?
        $(, infer_array_types = $infer_array_types:expr)?
        $(, check_data_types = [$($(@$data_type_check:ident)+),* $(,)?])?
        $(, check_array_types = [$(@$array_type_check:ident),* $(,)?])? $(,)?
    ) => {
        $crate::define_elementwise_operation!(@marker [$(#[$documentation])*] $operation, $name);

        impl $crate::Operation for $operation<$crate::arrays::DataType> {
            type Type = $crate::arrays::DataType;

            #[inline]
            fn name(&self) -> &'static str {
                $name
            }

            #[inline]
            fn infer_output_types(
                &self,
                input_types: &[$crate::arrays::DataType],
                _region_interfaces: &[$crate::RegionInterface<$crate::arrays::DataType>],
            ) -> Result<Vec<$crate::arrays::DataType>, $crate::TypeError> {
                $crate::check_count!("input", input_types, 1, TypeError);
                $($($crate::check_types!($(@$data_type_check)+, $name, input_types);)*)?
                let output_types: Result<Vec<$crate::arrays::DataType>, $crate::TypeError> =
                    $crate::define_elementwise_operation!(@infer_data_types [$($infer_data_types)?] @unary input_types);
                let output_types = output_types?;
                $crate::check_count!("output", output_types, 1, TypeError);
                Ok(output_types)
            }
        }

        impl $crate::Operation for $operation<$crate::arrays::ArrayType> {
            type Type = $crate::arrays::ArrayType;

            #[inline]
            fn name(&self) -> &'static str {
                $name
            }

            #[inline]
            fn infer_output_types(
                &self,
                input_types: &[$crate::arrays::ArrayType],
                _region_interfaces: &[$crate::RegionInterface<$crate::arrays::ArrayType>],
            ) -> Result<Vec<$crate::arrays::ArrayType>, $crate::TypeError> {
                $crate::ElementwiseOperation::infer_output_types(self, input_types)
            }
        }

        impl $crate::ElementwiseOperation for $operation<$crate::arrays::ArrayType> {
            #[inline]
            fn input_count(&self) -> usize {
                1
            }

            #[inline]
            fn infer_output_types(
                &self,
                input_types: &[$crate::arrays::ArrayType],
            ) -> Result<Vec<$crate::arrays::ArrayType>, $crate::TypeError> {
                $crate::check_count!("input", input_types, 1, TypeError);
                $($($crate::check_types!($(@$data_type_check)+, $name, &[input_types[0].data_type()]);)*)?
                $($($crate::check_types!(@$array_type_check, $name, input_types);)*)?
                let output_types: Result<Vec<$crate::arrays::ArrayType>, $crate::TypeError> =
                    $crate::define_elementwise_operation!(
                        @infer_array_types [$($infer_array_types)?] [$($infer_data_types)?] @unary self,
                        input_types,
                    );
                let output_types = output_types?;
                $crate::check_count!("output", output_types, 1, TypeError);
                Ok(output_types)
            }
        }

        impl<__C: $crate::Domain<Value: $capability>> $crate::InterpretableOperation<__C>
            for $operation<__C::Type>
        where
            $operation<__C::Type>: $crate::Operation<Type = __C::Type>,
        {
            #[inline]
            fn interpret<__D: $crate::InterpretationDriver<__C>>(
                &self,
                _context: &__C,
                _driver: &__D,
                inputs: &[__C::Value],
            ) -> Result<Vec<__C::Value>, $crate::ProgramError> {
                $crate::check_count!("input", inputs, 1, ProgramError);
                Ok(vec![inputs[0].$method()?])
            }
        }

        impl<__C: $crate::Context> $crate::PartiallyEvaluatableOperation<__C> for $operation<__C::Type>
        where
            $operation<__C::Type>: $crate::Operation<Type = __C::Type>,
            __C::Operation: ::std::convert::From<$operation<__C::Type>>,
        {
        }
    };

    // This public branch defines the corresponding binary elementwise operation machinery.
    (
        @binary
        $(#[$documentation:meta])*
        $operation:ident, $name:ident,
        $capability:ident, $method:ident
        $(, infer_data_types = $infer_data_types:expr)?
        $(, infer_array_types = $infer_array_types:expr)?
        $(, check_data_types = [$($(@$data_type_selector:ident)+),* $(,)?])?
        $(, check_array_types = [$(@$array_type_check:ident),* $(,)?])? $(,)?
    ) => {
        $crate::define_elementwise_operation!(@marker [$(#[$documentation])*] $operation, $name);

        impl $crate::Operation for $operation<$crate::arrays::DataType> {
            type Type = $crate::arrays::DataType;

            #[inline]
            fn name(&self) -> &'static str {
                $name
            }

            #[inline]
            fn infer_output_types(
                &self,
                input_types: &[$crate::arrays::DataType],
                _region_interfaces: &[$crate::RegionInterface<$crate::arrays::DataType>],
            ) -> Result<Vec<$crate::arrays::DataType>, $crate::TypeError> {
                $crate::check_count!("input", input_types, 2, TypeError);
                $($($crate::check_types!($(@$data_type_selector)+, $name, input_types);)*)?
                let output_types: Result<Vec<$crate::arrays::DataType>, $crate::TypeError> =
                    $crate::define_elementwise_operation!(
                        @infer_data_types [$($infer_data_types)?] @binary input_types,
                        $name,
                    );
                let output_types = output_types?;
                $crate::check_count!("output", output_types, 1, TypeError);
                Ok(output_types)
            }
        }

        impl $crate::Operation for $operation<$crate::arrays::ArrayType> {
            type Type = $crate::arrays::ArrayType;

            #[inline]
            fn name(&self) -> &'static str {
                $name
            }

            #[inline]
            fn infer_output_types(
                &self,
                input_types: &[$crate::arrays::ArrayType],
                _region_interfaces: &[$crate::RegionInterface<$crate::arrays::ArrayType>],
            ) -> Result<Vec<$crate::arrays::ArrayType>, $crate::TypeError> {
                $crate::ElementwiseOperation::infer_output_types(self, input_types)
            }
        }

        impl $crate::ElementwiseOperation for $operation<$crate::arrays::ArrayType> {
            #[inline]
            fn input_count(&self) -> usize {
                2
            }

            #[inline]
            fn infer_output_types(
                &self,
                input_types: &[$crate::arrays::ArrayType],
            ) -> Result<Vec<$crate::arrays::ArrayType>, $crate::TypeError> {
                $crate::check_count!("input", input_types, 2, TypeError);
                $($($crate::check_types!(
                    $(@$data_type_selector)+,
                    $name,
                    &[input_types[0].data_type(), input_types[1].data_type()],
                );)*)?
                $($($crate::check_types!(@$array_type_check, $name, input_types);)*)?
                let output_types: Result<Vec<$crate::arrays::ArrayType>, $crate::TypeError> =
                    $crate::define_elementwise_operation!(
                        @infer_array_types [$($infer_array_types)?] [$($infer_data_types)?] @binary self,
                        input_types,
                        $name,
                    );
                let output_types = output_types?;
                $crate::check_count!("output", output_types, 1, TypeError);
                Ok(output_types)
            }
        }

        impl<__C: $crate::Domain<Value: $capability>> $crate::InterpretableOperation<__C>
            for $operation<__C::Type>
        where
            $operation<__C::Type>: $crate::Operation<Type = __C::Type>,
        {
            #[inline]
            fn interpret<__D: $crate::InterpretationDriver<__C>>(
                &self,
                _context: &__C,
                _driver: &__D,
                inputs: &[__C::Value],
            ) -> Result<Vec<__C::Value>, $crate::ProgramError> {
                $crate::check_count!("input", inputs, 2, ProgramError);
                Ok(vec![inputs[0].$method(&inputs[1])?])
            }
        }

        impl<__C: $crate::Context> $crate::PartiallyEvaluatableOperation<__C> for $operation<__C::Type>
        where
            $operation<__C::Type>: $crate::Operation<Type = __C::Type>,
            __C::Operation: ::std::convert::From<$operation<__C::Type>>,
        {
        }
    };

    // This internal branch defines the zero-sized, type-indexed marker shared by both public operation arities.
    (@marker [$($documentation:tt)*] $operation:ident, $name:ident) => {
        $($documentation)*
        #[derive(Clone)]
        pub struct $operation<__T: $crate::Type>(::std::marker::PhantomData<fn() -> __T>);

        impl<__T: $crate::Type> $operation<__T> {
            #[doc = ::std::concat!("Creates a new [`", ::std::stringify!($operation), "`].")]
            pub const fn new() -> Self {
                Self(::std::marker::PhantomData)
            }
        }

        impl<__T: $crate::Type> ::std::default::Default for $operation<__T> {
            #[inline]
            fn default() -> Self {
                Self::new()
            }
        }

        impl<__T: $crate::Type> ::std::fmt::Debug for $operation<__T> {
            #[inline]
            fn fmt(&self, formatter: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                formatter.write_str(::std::stringify!($operation))
            }
        }

        impl<__T: $crate::Type> ::std::fmt::Display for $operation<__T> {
            #[inline]
            fn fmt(&self, formatter: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                formatter.write_str($name)
            }
        }
    };

    // This internal branch invokes caller-provided data-type inference for either operation arity.
    (@infer_data_types [$infer_data_types:expr] @$arity:ident $input_types:expr $(, $name:ident)? $(,)?) => {
        ($infer_data_types)($input_types)
    };

    // This internal branch supplies the default type-preserving inference for unary operations.
    (@infer_data_types [] @unary $input_types:expr $(,)?) => {
        Ok::<Vec<$crate::arrays::DataType>, $crate::TypeError>(vec![$input_types[0]])
    };

    // This internal branch supplies the default broadcasting inference for binary data types.
    (@infer_data_types [] @binary $input_types:expr, $name:ident $(,)?) => {
        $crate::arrays::Broadcastable::broadcast(&$input_types[0], &$input_types[1])
            .map(|output| vec![output])
            .map_err(|_| $crate::TypeError::invalid(format!("`{}` input types are not broadcast-compatible", $name)))
    };

    // This internal branch invokes caller-provided array-type inference instead of structural lifting.
    (
        @infer_array_types [$infer_array_types:expr] [$($infer_data_types:expr)?] @$arity:ident
        $operation:expr, $input_types:expr $(, $name:ident)? $(,)?
    ) => {
        ($infer_array_types)($input_types)
    };

    // This internal branch prepares unary element data types for the default array-structure lifting path.
    (
        @infer_array_types [] [$($infer_data_types:expr)?] @unary
        $operation:expr, $input_types:expr $(,)?
    ) => {{
        let input_data_types = [$input_types[0].data_type()];
        $crate::define_elementwise_operation!(
            @infer_default_array_types [$($infer_data_types)?] @unary
            $operation, $input_types, input_data_types.as_slice()
        )
    }};

    // This internal branch prepares both binary element data types for the default array-structure lifting path.
    (
        @infer_array_types [] [$($infer_data_types:expr)?] @binary
        $operation:expr, $input_types:expr, $name:ident $(,)?
    ) => {{
        let input_data_types = [$input_types[0].data_type(), $input_types[1].data_type()];
        $crate::define_elementwise_operation!(
            @infer_default_array_types [$($infer_data_types)?] @binary
            $operation, $input_types, input_data_types.as_slice(), $name
        )
    }};

    // This internal branch combines inferred element types with the operation's broadcast output structure.
    (
        @infer_default_array_types [$($infer_data_types:expr)?] @$arity:ident
        $operation:expr, $input_types:expr, $input_data_types:expr $(, $name:ident)?
    ) => {{
        let output_data_types: Result<Vec<$crate::arrays::DataType>, $crate::TypeError> =
            $crate::define_elementwise_operation!(
            @infer_data_types [$($infer_data_types)?] @$arity $input_data_types $(, $name)?
        );
        let output_data_types = output_data_types?;
        $crate::check_count!("output", output_data_types, 1, TypeError);
        let output_type = $crate::ElementwiseOperation::infer_elementwise_broadcast_type($operation, $input_types)?
            .with_data_type(output_data_types[0]);
        Ok::<Vec<$crate::arrays::ArrayType>, $crate::TypeError>(vec![output_type])
    }};
}

/// Defines a value-level capability trait paired with an elementwise operation and its dispatch-domain implementation.
///
/// # Examples
///
/// Unary capabilities document their receiver-only method directly:
///
/// ```rust,ignore
/// define_elementwise_capability!(
///     @unary
///     /// Value-level sine capability.
///     Sin,
///     /// Computes the elementwise sine of this value.
///     sin,
///     SinOperation,
/// );
/// ```
///
/// Binary capabilities must name their non-receiver argument. That name is used in both the generated trait method and
/// its blanket implementation, allowing operation-specific terminology such as the `x` coordinate of `atan2(y, x)`:
///
/// ```rust,ignore
/// define_elementwise_capability!(
///     @binary
///     /// Value-level two-argument arc-tangent capability.
///     Atan2,
///     /// Computes `atan2(self, x)`, with `self` representing the `y` coordinate.
///     atan2(x),
///     Atan2Operation,
/// );
/// ```
///
/// # Parameters
///
///   - `@unary` / `@binary`: Selects whether the capability consumes only `self` or also one named argument.
///   - `$(#[$capability_documentation])*`: Documentation attributes attached to the generated capability trait.
///   - `$capability`: Identifier of the generated value-level capability trait (e.g., `Sin`).
///   - `$(#[$method_documentation])*`: Documentation attributes attached to the generated capability method.
///   - `$method`: Identifier of the generated capability method (e.g., `sin`).
///   - `$argument`: Required name for the binary capability's non-receiver argument (e.g., `x` in `atan2(x)`).
///   - `$operation`: Stateless operation marker (e.g., `SinOperation`) whose
///     [`OperationProvider`](crate::OperationProvider) implementation selects and constructs the concrete operation
///     staged for each value type family. Stateless operations provide themselves through the blanket implementation,
///     while type families whose operations carry type-derived metadata (e.g., checked dimension arithmetic) override
///     the provider for the marker.
#[macro_export]
macro_rules! define_elementwise_capability {
    // This branch defines a receiver capability whose unary operation is provided by the value type family.
    (
        @unary
        $(#[$capability_documentation:meta])*
        $capability:ident,
        $(#[$method_documentation:meta])+
        $method:ident,
        $operation:ident $(,)?
    ) => {
        $(#[$capability_documentation])*
        pub trait $capability: Sized {
            $(#[$method_documentation])+
            fn $method(&self) -> Result<Self, $crate::ProgramError>;
        }

        impl<__V: $crate::Value> $capability for __V
        where
            $operation<__V::Type>: $crate::OperationProvider<__V::Type>,
            __V::DispatchDomain: $crate::Context<
                    Type = __V::Type,
                    Value = __V,
                    Operation: ::std::convert::From<
                        <$operation<__V::Type> as $crate::OperationProvider<__V::Type>>::Operation,
                    >,
                >,
        {
            #[inline]
            fn $method(&self) -> Result<Self, $crate::ProgramError> {
                let input_type = $crate::Typed::r#type(self);
                let operation =
                    <$operation<__V::Type> as $crate::OperationProvider<__V::Type>>::provide(&[input_type.as_ref()])?;
                Ok($crate::Context::bind(
                    &$crate::Value::dispatch_domain(self),
                    operation,
                    Vec::new(),
                    ::std::slice::from_ref(self),
                )?
                .remove(0))
            }
        }
    };

    // This branch defines a two-operand capability whose binary operation is provided by the value type family,
    // using the caller-provided name for the right operand.
    (
        @binary
        $(#[$capability_documentation:meta])*
        $capability:ident,
        $(#[$method_documentation:meta])+
        $method:ident($argument:ident),
        $operation:ident $(,)?
    ) => {
        $(#[$capability_documentation])*
        pub trait $capability: Sized {
            $(#[$method_documentation])+
            fn $method(&self, $argument: &Self) -> Result<Self, $crate::ProgramError>;
        }

        impl<__V: $crate::Value> $capability for __V
        where
            $operation<__V::Type>: $crate::OperationProvider<__V::Type>,
            __V::DispatchDomain: $crate::Context<
                    Type = __V::Type,
                    Value = __V,
                    Operation: ::std::convert::From<
                        <$operation<__V::Type> as $crate::OperationProvider<__V::Type>>::Operation,
                    >,
                >,
        {
            #[inline]
            fn $method(&self, $argument: &Self) -> Result<Self, $crate::ProgramError> {
                let left_type = $crate::Typed::r#type(self);
                let right_type = $crate::Typed::r#type($argument);
                let operation = <$operation<__V::Type> as $crate::OperationProvider<__V::Type>>::provide(&[
                    left_type.as_ref(),
                    right_type.as_ref(),
                ])?;
                Ok($crate::Context::bind(
                    &$crate::Value::dispatch_domain(self),
                    operation,
                    Vec::new(),
                    &[self.clone(), $argument.clone()],
                )?
                .remove(0))
            }
        }
    };
}

/// Implements the [`ReferenceDischargeableOperation`](crate::ReferenceDischargeableOperation) trait
/// for an [`Operation`](crate::Operation) as a verbatim replay/interpretation, by delegating to
/// [`discharge_reference_free_operation`](crate::discharge_reference_free_operation). The generated rule replays the
/// application over its rewritten operands, so an eager context executes it and a staging context records it. That is
/// the complete implementation for an operation that touches no reference, and a checked rejecting placeholder for one
/// that does, until that operation gets a discharge implementation of its own.
///
/// Note that the precondition this macro states is _reference freedom_ and not effect purity. An operation with ordered
/// or other effects replays here perfectly well, because replaying it reproduces those effects in the destination
/// exactly as the source performed them. Only a reference makes the rewrite the operation's own business.
///
/// An application that carries regions still replays verbatim when nothing in their closure touches a reference: the
/// regions are copied into the destination as they stand. The generated implementation is a rejection rather than a
/// rewrite in the two cases it cannot serve: a region closure that does reach a reference, because how a reference
/// boundary widens is knowledge that belongs to the operation, and an operand that is a live reference handle, because
/// a reference-touching operation owns its own rewrite. Both diagnostics name the operation.
///
/// The optional leading generic list declares operation-specific type parameters, and an optional `where` clause can
/// provide any bounds needed to make the operation type well-formed.
///
/// # Parameters
///
///   - `$generic`: Optional operation-specific type parameters used by `$operation`.
///   - `$operation`: The operation type for which the implementation is generated.
///   - `$bounds`: Optional bounds required to make `$operation` well-formed.
#[macro_export]
macro_rules! impl_reference_free_dischargeable_operation {
    // This branch accepts a generic operation with additional well-formedness bounds.
    (<$($generic:ident),+> $operation:ty where $($bounds:tt)+) => {
        $crate::impl_reference_free_dischargeable_operation!(@impl [$($generic),+] ($operation) { $($bounds)+ });
    };

    // This branch accepts a generic operation whose `Operation` implementation supplies all required bounds.
    (<$($generic:ident),+> $operation:ty $(,)?) => {
        $crate::impl_reference_free_dischargeable_operation!(@impl [$($generic),+] ($operation) {});
    };

    // This branch accepts the common non-generic operation form.
    ($operation:ty $(,)?) => {
        $crate::impl_reference_free_dischargeable_operation!(@impl [] ($operation) {});
    };

    // This internal helper emits the shared reference-free replay rule for every public invocation form. The
    // destination is bounded by `Context` rather than `Domain` because replaying the application binds it.
    (@impl [$($generic:ident),*] ($operation:ty) { $($bounds:tt)* }) => {
        impl<
            __C: $crate::Context,
            __P: $crate::ReferenceDischargePolicy<__C>
            $(, $generic)*
        >
            $crate::ReferenceDischargeableOperation<__C, __P> for $operation
        where
            __C::Operation: ::std::convert::From<$operation>,
            $operation: $crate::Operation<Type = __C::Type>,
            $($bounds)*
        {
            #[inline]
            fn discharge_references<__D: $crate::ReferenceDischargeDriver<__C, __P>>(
                &self,
                context: &$crate::ReferenceDischargeContext<__C, __P>,
                driver: &__D,
                inputs: &[$crate::ReferenceDischargeValue<__C, __P>],
            ) -> Result<Vec<$crate::ReferenceDischargeValue<__C, __P>>, $crate::ProgramError> {
                $crate::discharge_reference_free_operation(self, context, driver, inputs)
            }
        }
    };
}

/// Implements complete forward-mode differentiation (i.e., the Jacobian-Vector Product, or JVP, tranform) and primitive
/// transposition rules for an [`Operation`](crate::Operation). The caller supplies the operation-specific algorithms
/// while this macro generates the common [`DifferentiableOperation`](crate::DifferentiableOperation) and
/// [`TransposableOperation`](crate::TransposableOperation) implementation shells. Reverse-mode differentiation needs no
/// separate rule because it is derived by linearizing and then transposing the staged tangent program.
///
/// Use this macro when a rule needs direct access to the complete input list, an operation's structural metadata, or a
/// transformation driver. Elementwise operations should generally use [`impl_differentiable_elementwise_operation!`]
/// instead because it provides lazy tangent contributions, primal alignment, and structured transposition cases. The
/// closure-like syntax only names the generated method arguments; it does not allocate or dynamically dispatch a
/// runtime closure.
///
/// An optional leading generic list declares type parameters owned by the operation payload. Those parameters are
/// available to both generated implementations, so each rule can tie the payload's type universe to the abstraction
/// that owns it.
///
/// # Examples
///
/// A structural linear operation can provide both algorithms directly. The JVP body receives differentiation duals,
/// while the transposition body receives partial primal inputs and output cotangents:
///
/// ```rust,ignore
/// impl_differentiable_operation! {
///     BroadcastOperation,
///     jvp<C> where C: Context<Type = ArrayType, Value: Broadcast> {
///         |operation, _context, _driver, inputs| {
///             broadcast_jvp(operation, inputs)
///         }
///     },
///     transpose<V, O>
///     where
///         V: Value<Type = ArrayType>,
///         O: Operation<Type = ArrayType> + From<BroadcastOperation>,
///     {
///         |operation, context, driver, inputs, outputs| {
///             broadcast_transpose(operation, context, driver, inputs, outputs)
///         }
///     },
/// }
/// ```
///
/// A nonlinear primitive can provide its JVP and request the standard rejecting primitive-transposition rule.
/// Reverse-mode differentiation remains available when the JVP stages transposable linear primitives:
///
/// ```rust,ignore
/// impl_differentiable_operation! {
///     Atan2Operation,
///     jvp<C> where C::Value: Atan2 {
///         |operation, context, driver, inputs| {
///             atan2_jvp(operation, context, driver, inputs)
///         }
///     },
///     transpose = @nonlinear,
/// }
/// ```
///
/// # Parameters
///
///   - `@nonlinear`: Selects the standard unsupported primitive transposition rule.
///   - `$generic`: Optional operation-specific type parameters used by `$operation` and shared by the generated
///     Jacobian-Vector Product (JVP) and transposition implementations.
///   - `$operation`: Operation type for which the rules are generated.
///   - `$context`: Context parameter declared by `jvp<$context>` and used by the generated differentiation
///     implementation and its bounds.
///   - `$value`: Value parameter declared by `transpose<$value, $operations>` and used by the generated transposition
///     implementation and its bounds.
///   - `$operations`: Operation-family parameter declared by `transpose<$value, $operations>`.
///   - `$bounds`: Additional predicates required by a rule, written directly after its `where` using ordinary Rust
///     `where`-predicate syntax without an additional delimiter.
///   - `$operation_binding`: Name bound to `self` inside a rule body.
///   - `$context_binding`: Name bound to the active context inside a rule body.
///   - `$driver_binding`: Name bound to the instruction-scoped region driver inside a rule body.
///   - `$inputs`: Name bound to the complete input slice inside a rule body.
///   - `$outputs`: Name bound to the complete output-cotangent slice inside a transposition body.
#[macro_export]
macro_rules! impl_differentiable_operation {
    // This branch normalizes an operation-generic invocation before parsing its JVP and transposition rules.
    (
        <$($generic:ident),+> $operation:ty,
        $($rules:tt)*
    ) => {
        $crate::impl_differentiable_operation! {
            @start [$($generic),+] [$operation] $($rules)*
        }
    };

    // This branch normalizes the common non-generic invocation into the same parser state.
    (
        $operation:ty,
        $($rules:tt)*
    ) => {
        $crate::impl_differentiable_operation! {
            @start [] [$operation] $($rules)*
        }
    };

    // This internal branch starts parsing a JVP with an unbraced `where` clause. Token-by-token collection is
    // necessary because `macro_rules!` cannot otherwise distinguish the final predicate from the rule body.
    (
        @start [$($generic:ident),*] [$operation:ty]
        jvp<$context:ident>
        where
        $($tail:tt)*
    ) => {
        $crate::impl_differentiable_operation! {
            @collect_jvp_where [$($generic),*] [$context] [$operation] [] $($tail)*
        }
    };

    // This internal branch accepts a boundless JVP form and forwards it directly to normalized rule parsing.
    (
        @start [$($generic:ident),*] [$operation:ty]
        jvp<$context:ident> { $($jvp:tt)* },
        $($tail:tt)*
    ) => {
        $crate::impl_differentiable_operation! {
            @jvp_ready [$($generic),*] [$context] [$operation] [] { $($jvp)* } $($tail)*
        }
    };

    // This internal helper ends JVP-bound collection when it reaches the rule body and forwards the normalized rule.
    (
        @collect_jvp_where
        [$($generic:ident),*] [$context:ident] [$operation:ty] [$($bounds:tt)*]
        { $($jvp:tt)* },
        $($tail:tt)*
    ) => {
        $crate::impl_differentiable_operation! {
            @jvp_ready [$($generic),*] [$context] [$operation] [$($bounds)*] { $($jvp)* } $($tail)*
        }
    };

    // This internal helper consumes one token tree from a JVP `where` clause while preserving the remaining rule.
    (
        @collect_jvp_where
        [$($generic:ident),*] [$context:ident] [$operation:ty] [$($bounds:tt)*]
        $next:tt $($rest:tt)+
    ) => {
        $crate::impl_differentiable_operation! {
            @collect_jvp_where
            [$($generic),*] [$context] [$operation] [$($bounds)* $next] $($rest)*
        }
    };

    // This internal helper emits a generic operation's complete JVP followed by its rejecting transposition rule.
    (
        @jvp_ready [$generic:ident $(, $remaining_generic:ident)*]
        [$context:ident] [$operation:ty] [$($bounds:tt)*]
        { |$self:ident, $jvp_context:ident, $jvp_driver:ident, $inputs:ident| $jvp_body:block }
        transpose = @nonlinear $(,)?
    ) => {
        $crate::impl_differentiable_operation! {
            @impl_jvp
            impl<$context, $generic $(, $remaining_generic)*> $operation
            where { $($bounds)* }
            |$self, $jvp_context, $jvp_driver, $inputs| $jvp_body
        }

        $crate::impl_non_transposable_operation!(
            <$generic $(, $remaining_generic)*> $operation
            where $generic: $crate::Type $(, $remaining_generic: $crate::Type)*
        );
    };

    // This internal helper emits a non-generic operation's complete JVP followed by the standard rejecting
    // primitive-transposition rule.
    (
        @jvp_ready [] [$context:ident] [$operation:ty] [$($bounds:tt)*]
        { |$self:ident, $jvp_context:ident, $jvp_driver:ident, $inputs:ident| $jvp_body:block }
        transpose = @nonlinear $(,)?
    ) => {
        $crate::impl_differentiable_operation! {
            @impl_jvp
            impl<$context> $operation
            where { $($bounds)* }
            |$self, $jvp_context, $jvp_driver, $inputs| $jvp_body
        }

        $crate::impl_non_transposable_operation!($operation);
    };

    // This internal helper begins collecting an explicitly bounded transposition rule while retaining the complete
    // normalized JVP. The transposition body supplies the unambiguous end marker for its predicates.
    (
        @jvp_ready [$($generic:ident),*] [$context:ident] [$operation:ty] [$($jvp_bounds:tt)*]
        { |$self:ident, $jvp_context:ident, $jvp_driver:ident, $inputs:ident| $jvp_body:block }
        transpose<$value:ident, $operations:ident>
        where
        $($tail:tt)*
    ) => {
        $crate::impl_differentiable_operation! {
            @collect_transpose_where
            [$($generic),*] [$context] [$operation] [$($jvp_bounds)*]
            [$self] [$jvp_context] [$jvp_driver] [$inputs] [$jvp_body]
            [$value] [$operations] [] $($tail)*
        }
    };

    // This internal helper accepts a boundless transposition rule and emits both normalized implementations directly.
    (
        @jvp_ready [$($generic:ident),*] [$context:ident] [$operation:ty] [$($jvp_bounds:tt)*]
        { |$self:ident, $jvp_context:ident, $jvp_driver:ident, $inputs:ident| $jvp_body:block }
        transpose<$value:ident, $operations:ident>
        { |$transpose_self:ident, $transpose_context:ident, $transpose_driver:ident, $transpose_inputs:ident,
            $outputs:ident| $transpose_body:block } $(,)?
    ) => {
        $crate::impl_differentiable_operation! {
            @impl_jvp
            impl<$context $(, $generic)*> $operation
            where { $($jvp_bounds)* }
            |$self, $jvp_context, $jvp_driver, $inputs| $jvp_body
        }

        $crate::impl_differentiable_operation! {
            @impl_transpose
            impl<$value, $operations $(, $generic)*> $operation
            where {}
            |$transpose_self, $transpose_context, $transpose_driver, $transpose_inputs, $outputs| $transpose_body
        }
    };

    // This internal helper ends transposition-bound collection and emits the retained JVP and transposition rules.
    (
        @collect_transpose_where
        [$($generic:ident),*] [$context:ident] [$operation:ty] [$($jvp_bounds:tt)*]
        [$self:ident] [$jvp_context:ident] [$jvp_driver:ident] [$inputs:ident] [$jvp_body:block]
        [$value:ident] [$operations:ident] [$($transpose_bounds:tt)*]
        { |$transpose_self:ident, $transpose_context:ident, $transpose_driver:ident, $transpose_inputs:ident,
            $outputs:ident| $transpose_body:block } $(,)?
    ) => {
        $crate::impl_differentiable_operation! {
            @impl_jvp
            impl<$context $(, $generic)*> $operation
            where { $($jvp_bounds)* }
            |$self, $jvp_context, $jvp_driver, $inputs| $jvp_body
        }

        $crate::impl_differentiable_operation! {
            @impl_transpose
            impl<$value, $operations $(, $generic)*> $operation
            where { $($transpose_bounds)* }
            |$transpose_self, $transpose_context, $transpose_driver, $transpose_inputs, $outputs| $transpose_body
        }
    };

    // This internal helper consumes one token tree from a transposition `where` clause while retaining the JVP state.
    (
        @collect_transpose_where
        [$($generic:ident),*] [$context:ident] [$operation:ty] [$($jvp_bounds:tt)*]
        [$self:ident] [$jvp_context:ident] [$jvp_driver:ident] [$inputs:ident] [$jvp_body:block]
        [$value:ident] [$operations:ident] [$($transpose_bounds:tt)*]
        $next:tt $($rest:tt)+
    ) => {
        $crate::impl_differentiable_operation! {
            @collect_transpose_where
            [$($generic),*] [$context] [$operation] [$($jvp_bounds)*]
            [$self] [$jvp_context] [$jvp_driver] [$inputs] [$jvp_body]
            [$value] [$operations] [$($transpose_bounds)* $next] $($rest)*
        }
    };

    // This internal helper emits the shared `DifferentiableOperation` shell around a complete caller-provided JVP.
    (
        @impl_jvp
        impl<$context:ident $(, $generic:ident)*> $operation:ty
        where { $($bounds:tt)* }
        |$self:ident, $jvp_context:ident, $jvp_driver:ident, $inputs:ident| $body:block
    ) => {
        impl<$context: $crate::Context $(, $generic)*> $crate::DifferentiableOperation<$context> for $operation
        where
            $operation: $crate::Operation<Type = <$context as $crate::Domain>::Type>,
            $($bounds)*
        {
            fn jvp<__D: $crate::DifferentiationDriver<$context>>(
                &self,
                $jvp_context: &$context,
                $jvp_driver: &__D,
                $inputs: &[$crate::DifferentiationDual<<$context as $crate::Domain>::Value>],
            ) -> Result<
                Vec<$crate::DifferentiationDual<<$context as $crate::Domain>::Value>>,
                $crate::DifferentiationError,
            > {
                let $self = self;
                $body
            }
        }
    };

    // This internal helper emits the shared `TransposableOperation` shell around a complete caller-provided pullback.
    (
        @impl_transpose
        impl<$value:ident, $operations:ident $(, $generic:ident)*> $operation:ty
        where { $($bounds:tt)* }
        |$self:ident, $context:ident, $driver:ident, $inputs:ident, $outputs:ident| $body:block
    ) => {
        impl<
            $value: $crate::Value,
            $operations: $crate::Operation<Type = <$value as $crate::Typed>::Type>,
            $($generic,)*
        > $crate::TransposableOperation<$value, $operations> for $operation
        where
            $operation: $crate::Operation<Type = <$value as $crate::Typed>::Type>,
            $($bounds)*
        {
            fn transpose<__D: $crate::TranspositionDriver<$value, $operations>>(
                &self,
                $context: &mut $crate::TranspositionContext<'_, $value, $operations>,
                $driver: &__D,
                $inputs: &[$crate::PartialValue<$crate::Tracer<$crate::TracingContext<$value, $operations>>>],
                $outputs: &[$crate::MaybeZero<$crate::Tracer<$crate::TracingContext<$value, $operations>>>],
            ) -> Result<
                Vec<$crate::MaybeZero<$crate::Tracer<$crate::TracingContext<$value, $operations>>>>,
                $crate::DifferentiationError,
            > {
                // Hand-written rule bodies stage ordinary linear operations only, so the transposition context is
                // narrowed to its tracing context through `DerefMut` before the body runs. The body consequently
                // cannot reach the reference accumulators, which only the hand-implemented reference-aware rules use.
                let $context: &mut $crate::TracingContext<$value, $operations> = $context;
                let $self = self;
                $body
            }
        }
    };
}

/// Implements the forward-mode differentiation (Jacobian-Vector Product; JVP) and primitive transposition rules for
/// an elementwise [`Operation`](crate::Operation). The macro keeps the operation-specific mathematical rule at the
/// invocation site while generating the common [`DifferentiableOperation`](crate::DifferentiableOperation) and
/// [`TransposableOperation`](crate::TransposableOperation) implementation shells. Reverse-mode differentiation
/// needs no separate rule because it is derived by linearizing and then transposing the staged tangent program.
///
/// Unary and binary JVP forms replay the primal operation through the active [`Context`](crate::Context), preserve
/// structural-zero tangents, and delegate type promotion, broadcasting, and sharding alignment to the shared
/// elementwise differentiation helpers. Each JVP contribution is written as a closure over `(primal, tangent)` pairs.
/// An `_` pattern declares that a contribution does not consume that value, so the macro does not evaluate or align it.
/// Contributions are invoked independently and lazily: a structural-zero tangent neither evaluates its contribution nor
/// stages conversions for the primals that contribution would consume. Each JVP rule declares the context parameter and
/// its operation-specific bounds in a `jvp<C> where ... { ... }` block. Unary rules contain one contribution and may
/// additionally bind the primal output after `->`. That value is evaluated at the output tangent type only when the
/// contribution is live. Binary rules list the left-tangent contribution first and the right-tangent contribution
/// second. The tangent slot for the other operand is `_` in each contribution, making the contribution's dependency
/// explicit at the call site.
///
/// `@linear` implements both JVP and transposition from signed coefficients: unary rules take one `@positive` or
/// `@negative` coefficient, binary rules take two, and every sign combination is supported. Binary rules combine live
/// tangents with the operation's natural signed combination (e.g., a single staged `sub` for a `[@positive, @negative]`
/// rule) so that the staged tangent program mirrors the primal operation. `@non_differentiable` replays the primal and
/// assigns structural-zero output tangents, while rejecting primitive transposition. `@constant` represents a
/// unary, single-output operation whose result is constant with respect to its exemplar input. Its transpose therefore
/// returns a structural-zero exemplar cotangent.
///
/// Binary rules may describe transposition as knownness cases. Each supported case marks one operand `@linear` and the
/// other `@known`, then gives the contribution to the linear operand as an ordinary Rust expression. The macro aligns
/// the known value to the live output cotangent, unaligns the contribution to the linear operand's cotangent type, and
/// returns structural zeros for known operands. Symmetric bilinear rules provide both operand orderings. One-sided
/// linear rules provide only the supported ordering. Unsupported patterns and missing cotangent spaces receive
/// diagnostics derived from the operation and operand names.
///
/// `@nonlinear` implements the standard erroring primitive transpose rule. Reverse-mode differentiation remains
/// available by transposing the linear operations produced by the JVP. Operations whose rules cannot be expressed as
/// elementwise tangent contributions should use [`impl_differentiable_operation!`] instead.
///
/// # Examples
///
/// Linear operations need only declare the sign with which each input contributes.
/// The macro generates both the JVP and the transposition rules:
///
/// ```rust,ignore
/// impl_differentiable_elementwise_operation! {
///     @linear
///     AddOperation,
///     rule = [@positive, @positive]
/// }
/// ```
///
/// A nonlinear unary rule expresses its one tangent contribution directly. This example uses `-> output` to bind
/// the primal output at the tangent type, reusing the original output when its type already matches and otherwise
/// re-evaluating it only when the input tangent is live:
///
/// ```rust,ignore
/// impl_differentiable_elementwise_operation! {
///     @unary
///     ExpOperation,
///     jvp<C> where C::Value: std::ops::Mul<Output = C::Value> {
///         |(_, input_tangent) -> output| output * input_tangent
///     },
///     transpose = @nonlinear,
/// }
/// ```
///
/// A binary rule provides one independently lazy contribution per input tangent. It can additionally describe the
/// supported primitive-transposition knownness cases. The macro derives diagnostics for omitted knownness patterns
/// and linear inputs without cotangent spaces from the operation and operand names:
///
/// ```rust,ignore
/// impl_differentiable_elementwise_operation! {
///     @binary
///     MulOperation,
///     jvp<C> where C::Value: std::ops::Mul<Output = C::Value> {
///         |(_, left_tangent), (right, _)| right * left_tangent;
///         |(left, _), (_, right_tangent)| left * right_tangent;
///     },
///     transpose<V, O>
///     where
///         V::Type: DifferentiableType,
///         O: From<MulOperation>,
///         Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<V::Type>,
///     {
///         [left = @linear, right = @known] =>
///             |output_cotangent| right.binary(output_cotangent, MulOperation::new());
///         [left = @known, right = @linear] =>
///             |output_cotangent| left.binary(output_cotangent, MulOperation::new());
///     },
/// }
/// ```
///
/// Finally, operations with no differential dependence use the compact selectors. `@non_differentiable` gives every
/// output a structural-zero tangent, while `@constant` also gives its exemplar input a structural-zero cotangent:
///
/// ```rust,ignore
/// impl_differentiable_elementwise_operation!(@non_differentiable CompareOperation);
/// impl_differentiable_elementwise_operation!(@constant SignOperation);
/// impl_differentiable_elementwise_operation!(@constant <T> ZeroLikeOperation<T>);
/// impl_differentiable_elementwise_operation! {
///     @linear<T>
///     TagOperation<T>,
///     rule = [@positive]
/// }
/// ```
///
/// # Parameters
///
///   - `@non_differentiable`: Selects a structural-zero JVP and unsupported primitive transposition rule.
///   - `@constant`: Selects a unary constant-in-its-input JVP and structural-zero transposition rule.
///   - `@linear`: Selects a fully linear unary or binary rule with the provided signed coefficients.
///   - `@unary`: Selects a unary JVP with one lazily evaluated tangent term.
///   - `@binary`: Selects a binary JVP with one lazily evaluated term per input tangent.
///   - `@nonlinear`: Selects the standard unsupported primitive transposition rule.
///   - `$context`: Context parameter declared by `jvp<$context>` and used by the generated differentiation
///     implementation and its bounds.
///   - `$operation`: Elementwise operation type for which the rules are generated.
///   - `$type`: Optional payload type parameter for `@constant <T>` and positive unary `@linear <T>` rules.
///     The generated implementations tie it to the active context's type universe.
///   - `$bounds`: Additional bounds required by a JVP or transposition formula, written directly after that rule's
///     `where` using ordinary Rust `where`-predicate syntax without an additional delimiter.
///   - `$input_primal`: Name bound to an aligned input primal. `_` omits that value without evaluating it.
///   - `$input_tangent`: Name bound to the live, aligned tangent whose contribution is being evaluated.
///   - `$output_primal`: Optional name following `->`, bound to the primal output evaluated at its tangent type.
///   - `$left_primal`, `$right_primal`: Names bound to aligned binary input primals. `_` omits a primal without
///     evaluating it.
///   - `$left_tangent`, `$right_tangent`: Names bound to the live, aligned tangent for the corresponding binary
///     contribution. The other contribution's tangent slot must be `_`.
///   - `$term`: Ordinary Rust expression computing one tangent contribution.
///   - `@linear`: Marks an operand that is unknown because it belongs to the linear program being transposed.
///   - `@known`: Marks an operand that is available as a known primal value during transposition.
///   - `$output_cotangent`: Name bound to the live output cotangent in a transposition case.
#[macro_export]
macro_rules! impl_differentiable_elementwise_operation {
    // This branch implements a type-parameterized unary result that is constant with respect to its exemplar input.
    // The payload's type parameter is the same type universe used by the differentiation and transposition contexts.
    (@constant <$type:ident> $operation:ty $(,)?) => {
        $crate::impl_non_differentiable_operation!(<$type> $operation where $type: $crate::Type);

        impl<
            $type: $crate::DifferentiableType,
            __V: $crate::Value<Type = $type>,
            __O: $crate::Operation<Type = $type>,
        > $crate::TransposableOperation<__V, __O> for $operation
        where
            $operation: $crate::Operation<Type = $type>,
        {
            fn transpose<__D: $crate::TranspositionDriver<__V, __O>>(
                &self,
                _context: &mut $crate::TranspositionContext<'_, __V, __O>,
                _driver: &__D,
                inputs: &[$crate::PartialValue<$crate::Tracer<$crate::TracingContext<__V, __O>>>],
                outputs: &[$crate::MaybeZero<$crate::Tracer<$crate::TracingContext<__V, __O>>>],
            ) -> Result<
                Vec<$crate::MaybeZero<$crate::Tracer<$crate::TracingContext<__V, __O>>>>,
                $crate::DifferentiationError,
            > {
                $crate::check_count!("input", inputs, 1, ProgramError);
                $crate::check_count!("output", outputs, 1, ProgramError);
                Ok(vec![$crate::MaybeZero::Zero($crate::DifferentiableType::cotangent(
                    $crate::Typed::r#type(&inputs[0]).as_ref(),
                )?)])
            }
        }
    };

    // This branch implements a type-parameterized positive unary linear operation. Its payload type parameter is tied
    // to the active context's type universe, while both its tangent and cotangent pass through unchanged.
    (
        @linear <$type:ident>
        $operation:ty,
        rule = [@positive $(,)?] $(,)?
    ) => {
        impl<__C: $crate::Context<Type = $type>, $type: $crate::DifferentiableType>
            $crate::DifferentiableOperation<__C> for $operation
        where
            __C::Operation: ::std::convert::From<$operation>,
        {
            fn jvp<__D: $crate::DifferentiationDriver<__C>>(
                &self,
                context: &__C,
                _driver: &__D,
                inputs: &[$crate::DifferentiationDual<__C::Value>],
            ) -> Result<Vec<$crate::DifferentiationDual<__C::Value>>, $crate::DifferentiationError> {
                $crate::check_count!("input", inputs, 1, ProgramError);
                let mut primals = $crate::Context::bind(
                    context,
                    self.clone(),
                    Vec::new(),
                    ::std::slice::from_ref(inputs[0].primal()),
                )?;
                $crate::check_count!("output", primals, 1, ProgramError);
                Ok(vec![$crate::DifferentiationDual::new(primals.remove(0), inputs[0].tangent().clone())?])
            }
        }

        impl<
            $type: $crate::Type,
            __V: $crate::Value<Type = $type>,
            __O: $crate::Operation<Type = $type>,
        > $crate::TransposableOperation<__V, __O> for $operation
        where
            $operation: $crate::Operation<Type = $type>,
        {
            fn transpose<__D: $crate::TranspositionDriver<__V, __O>>(
                &self,
                _context: &mut $crate::TranspositionContext<'_, __V, __O>,
                _driver: &__D,
                inputs: &[$crate::PartialValue<$crate::Tracer<$crate::TracingContext<__V, __O>>>],
                outputs: &[$crate::MaybeZero<$crate::Tracer<$crate::TracingContext<__V, __O>>>],
            ) -> Result<
                Vec<$crate::MaybeZero<$crate::Tracer<$crate::TracingContext<__V, __O>>>>,
                $crate::DifferentiationError,
            > {
                $crate::check_count!("input", inputs, 1, ProgramError);
                $crate::check_count!("output", outputs, 1, ProgramError);
                Ok(vec![outputs[0].clone()])
            }
        }
    };

    // This branch implements an operation with no differential dependence by replaying its primal outputs with
    // structural-zero tangents and generating the standard rejecting primitive-transposition rule.
    (@non_differentiable $operation:ident $(,)?) => {
        $crate::impl_non_differentiable_operation!(
            <__P> $operation<__P> where __P: $crate::Type
        );
        $crate::impl_non_transposable_operation!(
            <__P> $operation<__P> where __P: $crate::Type
        );
    };

    // This branch implements a unary result that is constant with respect to its exemplar input. Its JVP has a
    // structural-zero tangent, while transposition returns a structural-zero cotangent shaped by that input.
    (@constant $operation:ident $(,)?) => {
        $crate::impl_non_differentiable_operation!(<__P> $operation<__P>
            where __P: $crate::Type);

        impl<
            __T: $crate::DifferentiableType,
            __P: $crate::Type,
            __V: $crate::Value<Type = __T>,
            __O: $crate::Operation<Type = __T>,
        > $crate::TransposableOperation<__V, __O> for $operation<__P>
        where
            $operation<__P>: $crate::Operation<Type = __T>,
        {
            fn transpose<__D: $crate::TranspositionDriver<__V, __O>>(
                &self,
                _context: &mut $crate::TranspositionContext<'_, __V, __O>,
                _driver: &__D,
                inputs: &[$crate::PartialValue<$crate::Tracer<$crate::TracingContext<__V, __O>>>],
                outputs: &[$crate::MaybeZero<$crate::Tracer<$crate::TracingContext<__V, __O>>>],
            ) -> Result<
                Vec<$crate::MaybeZero<$crate::Tracer<$crate::TracingContext<__V, __O>>>>,
                $crate::DifferentiationError,
            > {
                $crate::check_count!("input", inputs, 1, ProgramError);
                $crate::check_count!("output", outputs, 1, ProgramError);
                Ok(vec![$crate::MaybeZero::Zero($crate::DifferentiableType::cotangent(
                    $crate::Typed::r#type(&inputs[0]).as_ref(),
                )?)])
            }
        }
    };

    // This branch starts parsing a unary rule, whose single independently lazy tangent contribution may bind
    // its input primal and output primal. The shared parser handles its optional unbraced JVP bounds.
    (@unary $operation:ident, $($tail:tt)*) => {
        $crate::impl_differentiable_elementwise_operation!(
            @public_jvp [unary] [__P] $operation<__P>, $($tail)*
        );
    };

    // This branch starts parsing a binary rule with one independently lazy contribution per input tangent
    // and either a structured or custom primitive-transposition rule.
    (@binary $operation:ident, $($tail:tt)*) => {
        $crate::impl_differentiable_elementwise_operation!(
            @public_jvp [binary] [__P] $operation<__P>, $($tail)*
        );
    };

    // This branch implements a positive unary linear rule. Both its tangent and cotangent pass through unchanged,
    // so the generated implementations need no arithmetic capabilities beyond the operation itself.
    (
        @linear
        $operation:ident,
        rule = [@positive $(,)?] $(,)?
    ) => {
        $crate::impl_differentiable_elementwise_operation! {
            @linear_unary [positive]
            impl<__C, __P> $operation<__P>
            where {}
            transpose_type_bound { $crate::Type }
            transpose_operation_bounds {}
        }
    };

    // This branch implements a negative unary linear rule. Both its tangent and cotangent are negated, so it supplies
    // the shared generator with the `Neg` value capability and `NegOperation` operation-family conversion.
    (
        @linear
        $operation:ident,
        rule = [@negative $(,)?] $(,)?
    ) => {
        $crate::impl_differentiable_elementwise_operation! {
            @linear_unary [negative]
            impl<__C, __P> $operation<__P>
            where {
                <__C as $crate::Domain>::Value: ::std::ops::Neg<Output = <__C as $crate::Domain>::Value>,
            }
            transpose_type_bound { $crate::DifferentiableType }
            transpose_operation_bounds { + ::std::convert::From<$crate::NegOperation<__T>> }
        }
    };

    // This branch implements a binary linear rule with two positive coefficients. It combines live tangents
    // with addition and forwards the output cotangent positively to each linear input.
    (
        @linear
        $operation:ident,
        rule = [@positive, @positive $(,)?] $(,)?
    ) => {
        $crate::impl_differentiable_elementwise_operation! {
            @linear_binary [positive, positive]
            impl<__C, __P> $operation<__P>
            where { ::std::ops::Add<Output = <__C as $crate::Domain>::Value> }
            transpose_operation_bounds {}
        }
    };

    // This branch implements a binary linear rule with a positive left coefficient and negative right coefficient.
    // It stages subtraction for the tangent and negates only the right cotangent contribution.
    (
        @linear
        $operation:ident,
        rule = [@positive, @negative $(,)?] $(,)?
    ) => {
        $crate::impl_differentiable_elementwise_operation! {
            @linear_binary [positive, negative]
            impl<__C, __P> $operation<__P>
            where {
                ::std::ops::Neg<Output = <__C as $crate::Domain>::Value>
                    + ::std::ops::Sub<Output = <__C as $crate::Domain>::Value>
            }
            transpose_operation_bounds { + ::std::convert::From<$crate::NegOperation<__T>> }
        }
    };

    // This branch implements a binary linear rule with a negative left coefficient and positive right coefficient.
    // It stages the reversed subtraction needed for the tangent and negates only the left cotangent.
    (
        @linear
        $operation:ident,
        rule = [@negative, @positive $(,)?] $(,)?
    ) => {
        $crate::impl_differentiable_elementwise_operation! {
            @linear_binary [negative, positive]
            impl<__C, __P> $operation<__P>
            where {
                ::std::ops::Neg<Output = <__C as $crate::Domain>::Value>
                    + ::std::ops::Sub<Output = <__C as $crate::Domain>::Value>
            }
            transpose_operation_bounds { + ::std::convert::From<$crate::NegOperation<__T>> }
        }
    };

    // This branch implements a binary linear rule with two negative coefficients. It negates the sum of live
    // tangents and negates both input cotangent contributions.
    (
        @linear
        $operation:ident,
        rule = [@negative, @negative $(,)?] $(,)?
    ) => {
        $crate::impl_differentiable_elementwise_operation! {
            @linear_binary [negative, negative]
            impl<__C, __P> $operation<__P>
            where {
                ::std::ops::Add<Output = <__C as $crate::Domain>::Value>
                    + ::std::ops::Neg<Output = <__C as $crate::Domain>::Value>
            }
            transpose_operation_bounds { + ::std::convert::From<$crate::NegOperation<__T>> }
        }
    };

    // This internal helper branch recognizes a public JVP with an unbraced `where` clause and initializes
    // token-by-token bound collection. Collection is necessary because `macro_rules!` has no fragment that
    // matches a complete Rust `where` clause while also identifying where the following JVP body begins.
    (
        @public_jvp [$kind:ident] [$($generic:ident),*]
        $operation:ty,
        jvp<$context:ident>
        where
        $($tail:tt)*
    ) => {
        $crate::impl_differentiable_elementwise_operation! {
            @collect_jvp_where [$kind] [$($generic),*] [$context] [$operation] [] $($tail)*
        }
    };

    // This internal helper branch recognizes a public JVP without additional bounds and forwards it directly to
    // normalized rule dispatch. It bypasses the collector so the common boundless form does not take an unnecessary
    // recursive parsing path.
    (
        @public_jvp [$kind:ident] [$($generic:ident),*]
        $operation:ty,
        jvp<$context:ident> { $($jvp:tt)* },
        $($tail:tt)*
    ) => {
        $crate::impl_differentiable_elementwise_operation! {
            @jvp_ready [$kind] [$($generic),*] [$context] [$operation] []
            { $($jvp)* }
            $($tail)*
        }
    };

    // This internal helper branch terminates JVP-bound collection when it reaches the brace-delimited JVP expression
    // and forwards the accumulated predicates to normalized rule dispatch. A terminal arm is required to distinguish
    // the body delimiter from ordinary token trees inside the preceding `where` clause.
    (
        @collect_jvp_where
        [$kind:ident] [$($generic:ident),*] [$context:ident] [$operation:ty] [$($bounds:tt)*]
        { $($jvp:tt)* },
        $($tail:tt)*
    ) => {
        $crate::impl_differentiable_elementwise_operation! {
            @jvp_ready [$kind] [$($generic),*] [$context] [$operation] [$($bounds)*]
            { $($jvp)* }
            $($tail)*
        }
    };

    // This internal helper branch consumes one token tree from an unbraced JVP `where` clause and recurses with that
    // token appended to the bound accumulator. It exists because arbitrary Rust predicates cannot be captured as one
    // macro fragment without also consuming the JVP body that follows them.
    (
        @collect_jvp_where
        [$kind:ident] [$($generic:ident),*] [$context:ident] [$operation:ty] [$($bounds:tt)*]
        $next:tt $($rest:tt)+
    ) => {
        $crate::impl_differentiable_elementwise_operation! {
            @collect_jvp_where
            [$kind] [$($generic),*] [$context] [$operation] [$($bounds)* $next] $($rest)*
        }
    };

    // This internal helper branch emits a unary elementwise JVP after the public parser has normalized its optional
    // bounds. It remains a dedicated branch because unary rules have one tangent contribution and always reject
    // transposition, unlike binary rules with structured knownness cases.
    (
        @jvp_ready [unary] [$($generic:ident),*] [$context:ident] [$operation:ty] [$($bounds:tt)*]
        { |($input_primal:tt, $input_tangent:ident) $(-> $output_primal:ident)?| $term:expr }
        transpose = @nonlinear $(,)?
    ) => {
        $crate::impl_differentiable_operation! {
            @impl_jvp
            impl<$context $(, $generic)*> $operation
            where {
                $($generic: $crate::Type,)*
                <$context as $crate::Domain>::Type: $crate::DifferentiableType,
                <$context as $crate::Domain>::Operation: ::std::convert::From<$operation>,
                <$context as $crate::Domain>::Value:
                    $crate::ElementwiseDerivativeAlignment<<$context as $crate::Domain>::Type>,
                $($bounds)*
            }
            |operation, context, _driver, inputs| {
                $crate::unary_elementwise_jvp(
                    operation,
                    inputs,
                    |input| {
                        let mut outputs = $crate::Context::bind(
                            context,
                            operation.clone(),
                            Vec::new(),
                            ::std::slice::from_ref(input),
                        )?;
                        $crate::check_count!("output", outputs, 1, ProgramError);
                        Ok(outputs.remove(0))
                    },
                    |operands| {
                        $crate::impl_differentiable_elementwise_operation! {
                            @bind_unary_input_primal operands, $input_primal
                        }
                        $($crate::impl_differentiable_elementwise_operation! {
                            @bind_unary_output_primal operands, $output_primal
                        })?
                        let $input_tangent = operands.input_tangent()?;
                        Ok($term)
                    },
                )
            }
        }

        $crate::impl_non_transposable_operation!(<$($generic),*> $operation where $($generic: $crate::Type),*);
    };

    // This internal helper branch emits a binary JVP whose operation is explicitly nonlinear under transposition, then
    // adds the standard rejecting transposition implementation. Separating this terminal form avoids forcing nonlinear
    // operations through the transposition-bound parser.
    (
        @jvp_ready [binary] [$($generic:ident),*] [$context:ident] [$operation:ty] [$($bounds:tt)*]
        { $($jvp:tt)* }
        transpose = @nonlinear $(,)?
    ) => {
        $crate::impl_differentiable_elementwise_operation! {
            @binary_jvp
            impl<$context $(, $generic)*> $operation
            where { $($bounds)* }
            jvp { $($jvp)* }
        }

        $crate::impl_non_transposable_operation!(<$($generic),*> $operation where $($generic: $crate::Type),*);
    };

    // This internal helper branch recognizes a binary rule with a transposition implementation and starts collecting
    // its unbraced transposition bounds. The JVP is carried along unchanged so both implementations can be emitted
    // once the transposition body supplies an unambiguous end marker.
    (
        @jvp_ready [binary] [$($generic:ident),*] [$context:ident] [$operation:ty] [$($jvp_bounds:tt)*]
        { $($jvp:tt)* }
        transpose<$value:ident, $operations:ident>
        where
        $($tail:tt)*
    ) => {
        $crate::impl_differentiable_elementwise_operation! {
            @collect_public_transpose_where
            [$($generic),*] [$context] [$operation] [$($jvp_bounds)*] [$($jvp)*]
            [$value] [$operations] [] $($tail)*
        }
    };

    // This internal helper branch ends binary transposition-bound collection at the brace-delimited rule body and
    // forwards a normalized representation to code generation. One terminal arm handles both structured cases and
    // the custom closure escape hatch because the braces already provide an unambiguous boundary after an arbitrary
    // `where`.
    (
        @collect_public_transpose_where
        [$($generic:ident),*] [$context:ident] [$operation:ty] [$($jvp_bounds:tt)*] [$($jvp:tt)*]
        [$value:ident] [$operations:ident] [$($transpose_bounds:tt)*]
        { $($transpose_body:tt)* } $(,)?
    ) => {
        $crate::impl_differentiable_elementwise_operation! {
            @binary_ready
            impl<$context $(, $generic)*> $operation
            where { $($jvp_bounds)* }
            {
                jvp { $($jvp)* }
                transpose<$value, $operations>
                where { $($transpose_bounds)* }
                { $($transpose_body)* }
            }
        }
    };

    // This internal helper branch consumes one token tree from a binary transposition `where` clause and recurses with
    // it in the bound accumulator. It is the recursive counterpart to the brace-delimited terminal arm above and is
    // needed solely because `macro_rules!` cannot parse a complete unbraced `where` clause.
    (
        @collect_public_transpose_where
        [$($generic:ident),*] [$context:ident] [$operation:ty] [$($jvp_bounds:tt)*] [$($jvp:tt)*]
        [$value:ident] [$operations:ident] [$($transpose_bounds:tt)*]
        $next:tt $($rest:tt)+
    ) => {
        $crate::impl_differentiable_elementwise_operation! {
            @collect_public_transpose_where
            [$($generic),*] [$context] [$operation] [$($jvp_bounds)*] [$($jvp)*]
            [$value] [$operations] [$($transpose_bounds)* $next] $($rest)*
        }
    };

    // This internal helper branch emits the shared unary linear JVP and transposition algorithms after the public
    // sign-specific arms have supplied their minimal bounds. Keeping the sign as a token lets one shell preserve
    // positive tangents and cotangents or negate negative ones without duplicating both trait implementations.
    (
        @linear_unary [$sign:ident]
        impl<$context:ident $(, $generic:ident)*> $operation:ty
        where { $($jvp_bounds:tt)* }
        transpose_type_bound { $($transpose_type_bound:tt)+ }
        transpose_operation_bounds { $($transpose_operation_bounds:tt)* }
    ) => {
        $crate::impl_differentiable_operation! {
            @impl_jvp
            impl<$context $(, $generic)*> $operation
            where {
                $($generic: $crate::Type,)*
                <$context as $crate::Domain>::Type: $crate::DifferentiableType,
                <$context as $crate::Domain>::Operation: ::std::convert::From<$operation>,
                $($jvp_bounds)*
            }
            |operation, context, _driver, inputs| {
                $crate::check_count!("input", inputs, 1, ProgramError);
                let mut primals = $crate::Context::bind(
                    context,
                    operation.clone(),
                    Vec::new(),
                    ::std::slice::from_ref(inputs[0].primal()),
                )?;
                $crate::check_count!("output", primals, 1, ProgramError);
                let tangent = inputs[0].tangent().clone().map(|tangent| {
                    $crate::impl_differentiable_elementwise_operation!(@apply_tangent_sign $sign, tangent)
                });
                Ok(vec![$crate::DifferentiationDual::new(primals.remove(0), tangent)?])
            }
        }

        impl<
            __T: $($transpose_type_bound)+,
            $($generic: $crate::Type,)*
            __V: $crate::Value<Type = __T>,
            __O: $crate::Operation<Type = __T> $($transpose_operation_bounds)*,
        > $crate::TransposableOperation<__V, __O> for $operation
        where
            $operation: $crate::Operation<Type = __T>,
        {
            fn transpose<__D: $crate::TranspositionDriver<__V, __O>>(
                &self,
                _context: &mut $crate::TranspositionContext<'_, __V, __O>,
                _driver: &__D,
                inputs: &[$crate::PartialValue<$crate::Tracer<$crate::TracingContext<__V, __O>>>],
                outputs: &[$crate::MaybeZero<$crate::Tracer<$crate::TracingContext<__V, __O>>>],
            ) -> Result<
                Vec<$crate::MaybeZero<$crate::Tracer<$crate::TracingContext<__V, __O>>>>,
                $crate::DifferentiationError,
            > {
                $crate::check_count!("input", inputs, 1, ProgramError);
                $crate::check_count!("output", outputs, 1, ProgramError);
                // Unary elementwise linear operations preserve their operand type, so applying the declared sign is
                // sufficient; the output cotangent needs no unalignment before becoming the input contribution.
                Ok(vec![outputs[0].clone().map(|cotangent| {
                    $crate::impl_differentiable_elementwise_operation!(@apply_tangent_sign $sign, cotangent)
                })])
            }
        }
    };

    // This internal helper branch emits the shared binary linear JVP and transposition algorithms after a public sign
    // rule has selected the minimal arithmetic bounds. The two signs remain explicit because they determine both the
    // natural staged tangent expression and each operand's cotangent contribution.
    (
        @linear_binary [$left_sign:ident, $right_sign:ident]
        impl<$context:ident $(, $generic:ident)*> $operation:ty
        where { $($value_bounds:tt)* }
        transpose_operation_bounds { $($transpose_operation_bounds:tt)* }
    ) => {
        $crate::impl_differentiable_operation! {
            @impl_jvp
            impl<$context $(, $generic)*> $operation
            where {
                $($generic: $crate::Type,)*
                <$context as $crate::Domain>::Type: $crate::DifferentiableType,
                <$context as $crate::Domain>::Operation: ::std::convert::From<$operation>,
                <$context as $crate::Domain>::Value: $($value_bounds)*
                    + $crate::ElementwiseDerivativeAlignment<<$context as $crate::Domain>::Type>,
            }
            |operation, context, _driver, inputs| {
                $crate::check_count!("input", inputs, 2, ProgramError);
                let mut primals = $crate::Context::bind(
                    context,
                    operation.clone(),
                    Vec::new(),
                    &[inputs[0].primal().clone(), inputs[1].primal().clone()],
                )?;
                $crate::check_count!("output", primals, 1, ProgramError);
                let primal = primals.remove(0);
                let target = $crate::DifferentiableType::tangent($crate::Typed::r#type(&primal).as_ref())?;
                let left = inputs[0].tangent().as_value();
                let right = inputs[1].tangent().as_value();
                if $crate::DifferentiableType::is_zero_space(&target) && (left.is_some() || right.is_some()) {
                    return Err($crate::ProgramError::UnsupportedOperation {
                        message: format!(
                            "`{}` output type {} has no tangent space",
                            $crate::Operation::name(operation),
                            $crate::Typed::r#type(&primal),
                        ),
                    }
                    .into());
                }

                // This rule combines live tangents with the operation's natural signed combination (e.g., a single
                // staged `sub` for a `[@positive, @negative]` rule) instead of delegating to `binary_elementwise_jvp`,
                // which always sums its per-side contributions and would therefore stage `add(left, neg(right))` here.
                // Staged tangent program shapes are part of an operation's differentiation contract, so this difference
                // is load-bearing and not a consolidation candidate.
                let tangent = match (left, right) {
                    (Some(left), Some(right)) => {
                        let left = $crate::ElementwiseDerivativeAlignment::align_tangent(left, &target, &primal)?;
                        let right = $crate::ElementwiseDerivativeAlignment::align_tangent(right, &target, &primal)?;
                        $crate::MaybeZero::Value($crate::impl_differentiable_elementwise_operation!(
                                    @combine_linear_tangents [$left_sign, $right_sign], left, right
                        ))
                    }
                    (Some(tangent), None) => {
                        let tangent =
                            $crate::ElementwiseDerivativeAlignment::align_tangent(tangent, &target, &primal)?;
                        $crate::MaybeZero::Value($crate::impl_differentiable_elementwise_operation!(
                            @apply_tangent_sign $left_sign, tangent
                        ))
                    }
                    (None, Some(tangent)) => {
                        let tangent =
                            $crate::ElementwiseDerivativeAlignment::align_tangent(tangent, &target, &primal)?;
                        $crate::MaybeZero::Value($crate::impl_differentiable_elementwise_operation!(
                            @apply_tangent_sign $right_sign, tangent
                        ))
                    }
                    (None, None) => $crate::MaybeZero::Zero(target),
                };
                Ok(vec![$crate::DifferentiationDual::new(primal, tangent)?])
            }
        }

        impl<
            __T: $crate::DifferentiableType,
            $($generic: $crate::Type,)*
            __V: $crate::Value<Type = __T>,
            __O: $crate::Operation<Type = __T> $($transpose_operation_bounds)*,
        > $crate::TransposableOperation<__V, __O> for $operation
        where
            $crate::Tracer<$crate::TracingContext<__V, __O>>: $crate::ElementwiseDerivativeAlignment<__T>,
            $operation: $crate::Operation<Type = __T>,
        {
            fn transpose<__D: $crate::TranspositionDriver<__V, __O>>(
                &self,
                _context: &mut $crate::TranspositionContext<'_, __V, __O>,
                _driver: &__D,
                inputs: &[$crate::PartialValue<$crate::Tracer<$crate::TracingContext<__V, __O>>>],
                outputs: &[$crate::MaybeZero<$crate::Tracer<$crate::TracingContext<__V, __O>>>],
            ) -> Result<
                Vec<$crate::MaybeZero<$crate::Tracer<$crate::TracingContext<__V, __O>>>>,
                $crate::DifferentiationError,
            > {
                $crate::check_count!("input", inputs, 2, ProgramError);
                $crate::check_count!("output", outputs, 1, ProgramError);
                match &outputs[0] {
                    $crate::MaybeZero::Zero(_) =>
                        inputs
                            .iter()
                            .map(|input| {
                                Ok($crate::MaybeZero::Zero($crate::DifferentiableType::cotangent(
                                    $crate::Typed::r#type(input).as_ref(),
                                )?))
                            })
                            .collect(),
                    $crate::MaybeZero::Value(cotangent) => {
                        let operation_name = $crate::Operation::name(self);
                        Ok(vec![
                            $crate::impl_differentiable_elementwise_operation!(
                        @linear_transpose_contribution $left_sign, operation_name, &inputs[0], cotangent
                            ),
                            $crate::impl_differentiable_elementwise_operation!(
                        @linear_transpose_contribution $right_sign, operation_name, &inputs[1], cotangent
                            ),
                        ])
                    }
                }
            }
        }
    };

    // This internal helper branch combines two positive tangent contributions with addition. A sign-specific expansion
    // keeps the generated program in the operation's natural linear form and requires only `Add` from the value type.
    (@combine_linear_tangents [positive, positive], $left:expr, $right:expr) => { $left + $right };

    // This internal helper branch subtracts a negative right contribution from a positive left contribution. Emitting
    // `Sub` directly preserves the expected staged program instead of rewriting the rule as addition plus negation.
    (@combine_linear_tangents [positive, negative], $left:expr, $right:expr) => { $left - $right };

    // This internal helper branch subtracts a negative left contribution from a positive right contribution. The
    // reversed operand order implements `-left + right` directly while retaining the minimal `Sub` requirement.
    (@combine_linear_tangents [negative, positive], $left:expr, $right:expr) => { $right - $left };

    // This internal helper branch adds two magnitudes and negates the result when both tangent contributions are
    // negative. Keeping this case explicit avoids imposing subtraction bounds that its formula does not use.
    (@combine_linear_tangents [negative, negative], $left:expr, $right:expr) => { -($left + $right) };

    // This internal helper branch applies a positive derivative sign as the identity. It pairs with the negative arm so
    // the shared unary and binary generators can select sign behavior without duplicating their surrounding algorithms.
    (@apply_tangent_sign positive, $tangent:expr) => { $tangent };

    // This internal helper branch applies a negative derivative sign with one negation. It is isolated from the
    // positive arm so positive linear rules do not acquire an unnecessary `Neg` bound or staged negation operation.
    (@apply_tangent_sign negative, $tangent:expr) => { -$tangent };

    // This internal helper branch converts one live output cotangent into a signed input contribution for a binary
    // linear rule. It centralizes zero-space validation and broadcast unalignment because both operands require exactly
    // that boundary handling even though their signs can differ.
    (@linear_transpose_contribution $sign:ident, $operation_name:ident, $input:expr, $cotangent:ident) => {{
        let target = $crate::DifferentiableType::cotangent($crate::Typed::r#type($input).as_ref())?;
        if $crate::DifferentiableType::is_zero_space(&target) {
            return Err($crate::ProgramError::UnsupportedOperation {
                message: format!("`{}` input has no cotangent space", $operation_name),
            }
            .into());
        }
        let contribution = $crate::impl_differentiable_elementwise_operation!(
            @apply_tangent_sign $sign, $cotangent.clone()
        );
        $crate::MaybeZero::Value($crate::ElementwiseDerivativeAlignment::unalign_cotangent(&contribution, &target)?)
    }};

    // This internal helper branch generates a binary rule whose transposition supports either operand being linear
    // while the other is known. It must remain distinct from one-sided rules because it selects between two
    // user-provided cotangent formulas at runtime and reports the actual unsupported knownness pattern.
    (
        @binary_ready
        impl<$context:ident $(, $generic:ident)*> $operation:ty
        where { $($jvp_bounds:tt)* }
        {
            jvp { $($jvp:tt)* }

            transpose<$value:ident, $operations:ident>
            where { $($transpose_bounds:tt)* }
            {
                [$transpose_left:ident = @linear, $transpose_right:ident = @known] =>
                    |$left_output_cotangent:ident| $left_contribution:expr;
                [$transpose_left_again:ident = @known, $transpose_right_again:ident = @linear] =>
                    |$right_output_cotangent:ident| $right_contribution:expr;
            }
        }
    ) => {
        $crate::impl_differentiable_elementwise_operation! {
            @binary_jvp
            impl<$context $(, $generic)*> $operation
            where { $($jvp_bounds)* }
            jvp { $($jvp)* }
        }

        $crate::impl_differentiable_operation! {
            @impl_transpose
            impl<$value, $operations $(, $generic)*> $operation
            where {
                $($generic: $crate::Type,)*
                $($transpose_bounds)*
            }
            |operation, _context, _driver, inputs, outputs| {
                $crate::check_count!("input", inputs, 2, ProgramError);
                $crate::check_count!("output", outputs, 1, ProgramError);
                let (linear_index, contribution) = match (inputs[0].is_unknown(), inputs[1].is_unknown()) {
                    (true, false) => {
                        let target = $crate::DifferentiableType::cotangent(
                            $crate::Typed::r#type(&inputs[0]).as_ref(),
                        )?;
                        let contribution = match &outputs[0] {
                            $crate::MaybeZero::Zero(_) => $crate::MaybeZero::Zero(target),
                            $crate::MaybeZero::Value($left_output_cotangent) => {
                                if $crate::DifferentiableType::is_zero_space(&target) {
                                    return Err($crate::ProgramError::UnsupportedOperation {
                                        message: format!(
                                            "linear input `{}` of operation `{}` has no cotangent space",
                                            stringify!($transpose_left),
                                            $crate::Operation::name(operation),
                                        ),
                                    }
                                    .into());
                                }
                                // The surrounding knownness match guarantees that this operand is known.
                                let $transpose_right = $crate::ElementwiseDerivativeAlignment::align_tangent(
                                    inputs[1].as_known().unwrap(),
                                    $crate::Typed::r#type($left_output_cotangent).as_ref(),
                                    $left_output_cotangent,
                                )?;
                                let contribution = $left_contribution;
                                $crate::MaybeZero::Value(
                                    $crate::ElementwiseDerivativeAlignment::unalign_cotangent(
                                        &contribution,
                                        &target,
                                    )?,
                                )
                            }
                        };
                        (0, contribution)
                    }
                    (false, true) => {
                        let target = $crate::DifferentiableType::cotangent(
                            $crate::Typed::r#type(&inputs[1]).as_ref(),
                        )?;
                        let contribution = match &outputs[0] {
                            $crate::MaybeZero::Zero(_) => $crate::MaybeZero::Zero(target),
                            $crate::MaybeZero::Value($right_output_cotangent) => {
                                if $crate::DifferentiableType::is_zero_space(&target) {
                                    return Err($crate::ProgramError::UnsupportedOperation {
                                        message: format!(
                                            "linear input `{}` of operation `{}` has no cotangent space",
                                            stringify!($transpose_right_again),
                                            $crate::Operation::name(operation),
                                        ),
                                    }
                                    .into());
                                }
                                // The surrounding knownness match guarantees that this operand is known.
                                let $transpose_left_again = $crate::ElementwiseDerivativeAlignment::align_tangent(
                                    inputs[0].as_known().unwrap(),
                                    $crate::Typed::r#type($right_output_cotangent).as_ref(),
                                    $right_output_cotangent,
                                )?;
                                let contribution = $right_contribution;
                                $crate::MaybeZero::Value(
                                    $crate::ElementwiseDerivativeAlignment::unalign_cotangent(
                                        &contribution,
                                        &target,
                                    )?,
                                )
                            }
                        };
                        (1, contribution)
                    }
                    (left_is_linear, right_is_linear) => {
                        return Err($crate::ProgramError::UnsupportedOperation {
                            message: format!(
                                "operation `{}` does not support transposition for input pattern [{} = {}, {} = {}]",
                                $crate::Operation::name(operation),
                                stringify!($transpose_left),
                                if left_is_linear { "linear" } else { "known" },
                                stringify!($transpose_right),
                                if right_is_linear { "linear" } else { "known" },
                            ),
                        }
                        .into());
                    }
                };
                let mut contributions = inputs
                    .iter()
                    .map(|input| {
                        Ok($crate::MaybeZero::Zero($crate::DifferentiableType::cotangent(
                            $crate::Typed::r#type(input).as_ref(),
                        )?))
                    })
                    .collect::<Result<Vec<_>, $crate::DifferentiationError>>()?;
                contributions[linear_index] = contribution;
                Ok(contributions)
            }
        }
    };

    // This internal helper branch generates a binary rule that can transpose only a linear left operand with a known
    // right operand. A dedicated branch keeps one-sided operations from pretending to support the mirrored case and
    // avoids requiring a second formula that is mathematically invalid or unavailable.
    (
        @binary_ready
        impl<$context:ident $(, $generic:ident)*> $operation:ty
        where { $($jvp_bounds:tt)* }
        {
            jvp { $($jvp:tt)* }

            transpose<$value:ident, $operations:ident>
            where { $($transpose_bounds:tt)* }
            {
                [$transpose_left:ident = @linear, $transpose_right:ident = @known] =>
                    |$output_cotangent:ident| $contribution:expr;
            }
        }
    ) => {
        $crate::impl_differentiable_elementwise_operation! {
            @binary_jvp
            impl<$context $(, $generic)*> $operation
            where { $($jvp_bounds)* }
            jvp { $($jvp)* }
        }

        $crate::impl_differentiable_operation! {
            @impl_transpose
            impl<$value, $operations $(, $generic)*> $operation
            where {
                $($generic: $crate::Type,)*
                $($transpose_bounds)*
            }
            |operation, _context, _driver, inputs, outputs| {
                $crate::check_count!("input", inputs, 2, ProgramError);
                $crate::check_count!("output", outputs, 1, ProgramError);
                let left_is_linear = inputs[0].is_unknown();
                let right_is_linear = inputs[1].is_unknown();
                if !left_is_linear || right_is_linear {
                    return Err($crate::ProgramError::UnsupportedOperation {
                        message: format!(
                            "operation `{}` does not support transposition for input pattern [{} = {}, {} = {}]",
                            $crate::Operation::name(operation),
                            stringify!($transpose_left),
                            if left_is_linear { "linear" } else { "known" },
                            stringify!($transpose_right),
                            if right_is_linear { "linear" } else { "known" },
                        ),
                    }
                    .into());
                }
                let target = $crate::DifferentiableType::cotangent(
                    $crate::Typed::r#type(&inputs[0]).as_ref(),
                )?;
                let contribution = match &outputs[0] {
                    $crate::MaybeZero::Zero(_) => $crate::MaybeZero::Zero(target),
                    $crate::MaybeZero::Value($output_cotangent) => {
                        if $crate::DifferentiableType::is_zero_space(&target) {
                            return Err($crate::ProgramError::UnsupportedOperation {
                                message: format!(
                                    "linear input `{}` of operation `{}` has no cotangent space",
                                    stringify!($transpose_left),
                                    $crate::Operation::name(operation),
                                ),
                            }
                            .into());
                        }
                        // The checks above guarantee that the right operand is known.
                        let $transpose_right = $crate::ElementwiseDerivativeAlignment::align_tangent(
                            inputs[1].as_known().unwrap(),
                            $crate::Typed::r#type($output_cotangent).as_ref(),
                            $output_cotangent,
                        )?;
                        let contribution = $contribution;
                        $crate::MaybeZero::Value(
                            $crate::ElementwiseDerivativeAlignment::unalign_cotangent(&contribution, &target)?,
                        )
                    }
                };
                Ok(vec![
                    contribution,
                    $crate::MaybeZero::Zero($crate::DifferentiableType::cotangent(
                        $crate::Typed::r#type(&inputs[1]).as_ref(),
                    )?),
                ])
            }
        }
    };

    // This internal helper branch pairs the shared binary JVP generator with a caller-supplied transposition closure.
    // It is the low-level escape hatch for rules whose knownness handling cannot be described by the structured
    // symmetric or one-sided forms above.
    (
        @binary_ready
        impl<$context:ident $(, $generic:ident)*> $operation:ty
        where { $($jvp_bounds:tt)* }
        {
            jvp { $($jvp:tt)* }

            transpose<$value:ident, $operations:ident>
            where { $($transpose_bounds:tt)* }
            {
                |$transpose_self:ident, $transpose_context:ident, $transpose_driver:ident, $transpose_inputs:ident,
                    $outputs:ident| $transpose_body:block
            }
        }
    ) => {
        $crate::impl_differentiable_elementwise_operation! {
            @binary_jvp
            impl<$context $(, $generic)*> $operation
            where { $($jvp_bounds)* }
            jvp { $($jvp)* }
        }

        $crate::impl_differentiable_operation! {
            @impl_transpose
            impl<$value, $operations $(, $generic)*> $operation
            where {
                $($generic: $crate::Type,)*
                $($transpose_bounds)*
            }
            |$transpose_self, $transpose_context, $transpose_driver, $transpose_inputs, $outputs| $transpose_body
        }
    };

    // This internal helper branch emits the common binary JVP implementation from two per-operand tangent formulas. It
    // owns primal replay, derivative alignment, structural-zero handling, and contribution summation so operation rules
    // only state the mathematics unique to each operand.
    (
        @binary_jvp
        impl<$context:ident $(, $generic:ident)*> $operation:ty
        where { $($bounds:tt)* }
        jvp {
            |($left_primal_for_left:tt, $left_tangent:ident), ($right_primal_for_left:tt, _)|
                $left_term:expr;
            |($left_primal_for_right:tt, _), ($right_primal_for_right:tt, $right_tangent:ident)|
                $right_term:expr;
        }
    ) => {
        $crate::impl_differentiable_operation! {
            @impl_jvp
            impl<$context $(, $generic)*> $operation
            where {
                $($generic: $crate::Type,)*
                <$context as $crate::Domain>::Type: $crate::DifferentiableType,
                <$context as $crate::Domain>::Operation: ::std::convert::From<$operation>,
                <$context as $crate::Domain>::Value: ::std::ops::Add<Output = <$context as $crate::Domain>::Value>
                    + $crate::ElementwiseDerivativeAlignment<<$context as $crate::Domain>::Type>,
                $($bounds)*
            }
            |operation, context, _driver, inputs| {
                $crate::binary_elementwise_jvp(
                    operation,
                    inputs,
                    |left, right| {
                        let mut outputs = $crate::Context::bind(
                            context,
                            operation.clone(),
                            Vec::new(),
                            &[left.clone(), right.clone()],
                        )?;
                        $crate::check_count!("output", outputs, 1, ProgramError);
                        Ok(outputs.remove(0))
                    },
                    |_operands, $left_tangent| {
                        $crate::impl_differentiable_elementwise_operation! {
                            @bind_binary_left_primal _operands, $left_primal_for_left
                        }
                        $crate::impl_differentiable_elementwise_operation! {
                            @bind_binary_right_primal _operands, $right_primal_for_left
                        }
                        Ok($left_term)
                    },
                    |_operands, $right_tangent| {
                        $crate::impl_differentiable_elementwise_operation! {
                            @bind_binary_left_primal _operands, $left_primal_for_right
                        }
                        $crate::impl_differentiable_elementwise_operation! {
                            @bind_binary_right_primal _operands, $right_primal_for_right
                        }
                        Ok($right_term)
                    },
                )
            }
        }
    };

    // This internal helper branch handles `_` for a unary input primal by emitting no binding and, importantly,
    // no accessor call. Avoiding the call preserves the DSL's promise that omitted primals are not evaluated
    // unnecessarily.
    (@bind_unary_input_primal $operands:ident, _) => {};

    // This internal helper branch binds a named unary input primal through the lazy operand accessor. It is separate
    // from the `_` arm so only formulas that reference the primal pay for alignment and possible replay.
    (@bind_unary_input_primal $operands:ident, $input_primal:ident) => {
        let $input_primal = $operands.input_primal()?;
    };

    // This internal helper branch binds the optional unary output primal at the tangent target type. The whole
    // invocation is conditionally expanded by the caller, so rules without an `-> output` binding never recompute
    // that value.
    (@bind_unary_output_primal $operands:ident, $output_primal:ident) => {
        let $output_primal = $operands.output_primal_at_tangent_type()?;
    };

    // This internal helper branch handles `_` for a binary left primal by emitting neither a binding nor an accessor
    // call. The explicit arm preserves lazy primal evaluation for tangent formulas that depend only on the right
    // operand.
    (@bind_binary_left_primal $operands:ident, _) => {};

    // This internal helper branch binds a named binary left primal through the lazy operand accessor. It complements
    // the `_` arm so the generated rule evaluates and aligns the left primal only when its formula references it.
    (@bind_binary_left_primal $operands:ident, $left_primal:ident) => {
        let $left_primal = $operands.left_primal()?;
    };

    // This internal helper branch handles `_` for a binary right primal by emitting neither a binding nor an accessor
    // call. The explicit arm preserves lazy primal evaluation for tangent formulas that depend only on the left
    // operand.
    (@bind_binary_right_primal $operands:ident, _) => {};

    // This internal helper branch binds a named binary right primal through the lazy operand accessor. It complements
    // the `_` arm so the generated rule evaluates and aligns the right primal only when its formula references it.
    (@bind_binary_right_primal $operands:ident, $right_primal:ident) => {
        let $right_primal = $operands.right_primal()?;
    };

}

/// Implements the [`DifferentiableOperation`](crate::DifferentiableOperation) rule for an operation whose outputs carry
/// no tangent, such as a Boolean-codomain predicate or an explicit gradient barrier. The primal operation is replayed
/// on the input primals, and each output is paired with a structural zero tangent, which stays symbolic and stages
/// nothing. Because such a rule stages no live tangent, an input-bearing operation can never appear on a linear operand
/// in a valid tangent program and is typically paired with [`impl_non_transposable_operation!`]. A regionless nullary
/// operation should instead use [`impl_nullary_transposable_operation!`] so that transposition accepts its outputs and
/// returns its empty operand-cotangent list. The optional leading generic list declares operation-specific type
/// parameters, and an optional `where` clause can provide any bounds needed to make the operation type well-formed.
///
/// # Parameters
///
///   - `$generic`: Optional operation-specific type parameters used by `$operation`.
///   - `$operation`: The operation type for which the implementation is generated.
///   - `$bounds`: Optional bounds required to make `$operation` well-formed.
#[macro_export]
macro_rules! impl_non_differentiable_operation {
    // This branch accepts a generic operation with additional well-formedness bounds.
    (<$($generic:ident),+> $operation:ty where $($bounds:tt)+) => {
        $crate::impl_non_differentiable_operation!(@impl [$($generic),+] ($operation) { $($bounds)+ });
    };

    // This branch accepts a generic operation whose `Operation` implementation supplies all required bounds.
    (<$($generic:ident),+> $operation:ty $(,)?) => {
        $crate::impl_non_differentiable_operation!(@impl [$($generic),+] ($operation) {});
    };

    // This branch accepts the common non-generic operation form.
    ($operation:ty $(,)?) => {
        $crate::impl_non_differentiable_operation!(@impl [] ($operation) {});
    };

    // This internal helper emits the implementation shared by every public invocation form.
    (@impl [$($generic:ident),*] ($operation:ty) { $($bounds:tt)* }) => {
        impl<__C: $crate::Context $(, $generic)*> $crate::DifferentiableOperation<__C> for $operation
        where
            __C::Type: $crate::DifferentiableType,
            __C::Operation: ::std::convert::From<$operation>,
            $operation: $crate::Operation<Type = __C::Type>,
            $($bounds)*
        {
            #[inline]
            fn jvp<__D: $crate::DifferentiationDriver<__C>>(
                &self,
                context: &__C,
                _driver: &__D,
                inputs: &[$crate::DifferentiationDual<__C::Value>],
            ) -> Result<Vec<$crate::DifferentiationDual<__C::Value>>, $crate::DifferentiationError> {
                // The outputs carry no tangent. We replay the primal operation on the input primals and pair each
                // output with a structural zero tangent, which stays symbolic and stages nothing.
                $crate::Context::bind(
                    context,
                    self.clone(),
                    ::std::vec::Vec::new(),
                    &inputs.iter().map(|dual| dual.primal().clone()).collect::<Vec<_>>(),
                )?
                .into_iter()
                .map($crate::DifferentiationDual::new_with_zero_tangent)
                .collect()
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
/// families were unified. The optional leading generic list declares operation-specific type parameters, and an
/// optional `where` clause can provide bounds needed to make the operation type well-formed.
///
/// # Parameters
///
///   - `$generic`: Optional operation-specific type parameters used by `$operation`.
///   - `$operation`: The operation type for which the implementation is generated.
///   - `$bounds`: Optional bounds required to make `$operation` well-formed.
#[macro_export]
macro_rules! impl_non_transposable_operation {
    // This branch accepts a generic operation with additional well-formedness bounds.
    (<$($generic:ident),+> $operation:ty where $($bounds:tt)+) => {
        $crate::impl_non_transposable_operation!(@impl [$($generic),+] ($operation) { $($bounds)+ });
    };

    // This branch accepts a generic operation whose `Operation` implementation supplies all required bounds.
    (<$($generic:ident),+> $operation:ty $(,)?) => {
        $crate::impl_non_transposable_operation!(@impl [$($generic),+] ($operation) {});
    };

    // This branch accepts the common non-generic operation form.
    ($operation:ty $(,)?) => {
        $crate::impl_non_transposable_operation!(@impl [] ($operation) {});
    };

    // This internal helper generates the standard unsupported-transposition diagnostic.
    (@impl [$($generic:ident),*] ($operation:ty) { $($bounds:tt)* }) => {
        impl<
            __T: $crate::Type,
            __V: $crate::Value<Type = __T>,
            __O: $crate::Operation<Type = __T>
            $(, $generic)*
        >
            $crate::TransposableOperation<__V, __O> for $operation
        where
            $operation: $crate::Operation<Type = __T>,
            $($bounds)*
        {
            #[inline]
            fn transpose<__D: $crate::TranspositionDriver<__V, __O>>(
                &self,
                _context: &mut $crate::TranspositionContext<'_, __V, __O>,
                _driver: &__D,
                _inputs: &[$crate::PartialValue<$crate::Tracer<$crate::TracingContext<__V, __O>>>],
                _outputs: &[$crate::MaybeZero<$crate::Tracer<$crate::TracingContext<__V, __O>>>],
            ) -> Result<
                Vec<$crate::MaybeZero<$crate::Tracer<$crate::TracingContext<__V, __O>>>>,
                $crate::DifferentiationError,
            > {
                Err($crate::ProgramError::UnsupportedOperation {
                    message: format!("operation `{}` is not transposable", $crate::Operation::name(self)),
                }
                .into())
            }
        }
    };
}

/// Implements the [`TransposableOperation`](crate::TransposableOperation) trait for a [`Region`](crate::Region)-less
/// nullary [`Operation`](crate::Operation). The generated implementation validates that the operation application has
/// no inputs, infers and validates its output count, and returns no operand cotangents. The optional leading generic
/// list declares operation-specific type parameters; the macro supplies its internal transposition type, value,
/// operation-family, and driver parameters and derives behavioral bounds from [`Operation`](crate::Operation). An
/// optional `where` clause can provide bounds required to make the operation type itself well-formed.
///
/// # Parameters
///
///   - `$generic`: Optional operation-specific type parameters used by `$operation`.
///   - `$operation`: Regionless nullary operation type for which the implementation is generated.
///   - `$bounds`: Optional bounds required to make `$operation` well-formed.
#[macro_export]
macro_rules! impl_nullary_transposable_operation {
    // This branch accepts a generic nullary operation with additional well-formedness bounds.
    (<$($generic:ident),+> $operation:ty where $($bounds:tt)+) => {
        $crate::impl_nullary_transposable_operation!(@impl [$($generic),+] ($operation) { $($bounds)+ });
    };

    // This branch accepts a generic nullary operation whose `Operation` implementation supplies all required bounds.
    (<$($generic:ident),+> $operation:ty $(,)?) => {
        $crate::impl_nullary_transposable_operation!(@impl [$($generic),+] ($operation) {});
    };

    // This branch accepts the common non-generic nullary operation form.
    ($operation:ty $(,)?) => {
        $crate::impl_nullary_transposable_operation!(@impl [] ($operation) {});
    };

    // This internal helper emits the transposition implementation shared by every public invocation form.
    (@impl [$($generic:ident),*] ($operation:ty) { $($bounds:tt)* }) => {
        impl<__T: $crate::Type, __V: $crate::Value<Type = __T>, __O: $crate::Operation<Type = __T> $(, $generic)*>
            $crate::TransposableOperation<__V, __O> for $operation
        where
            $operation: $crate::Operation<Type = __T>,
            $($bounds)*
        {
            #[inline]
            fn transpose<__D: $crate::TranspositionDriver<__V, __O>>(
                &self,
                _context: &mut $crate::TranspositionContext<'_, __V, __O>,
                _driver: &__D,
                inputs: &[$crate::PartialValue<$crate::Tracer<$crate::TracingContext<__V, __O>>>],
                outputs: &[$crate::MaybeZero<$crate::Tracer<$crate::TracingContext<__V, __O>>>],
            ) -> Result<
                Vec<$crate::MaybeZero<$crate::Tracer<$crate::TracingContext<__V, __O>>>>,
                $crate::DifferentiationError,
            > {
                $crate::check_count!("input", inputs, 0, ProgramError);
                let output_count = $crate::Operation::infer_output_types(self, &[], &[])?.len();
                $crate::check_count!("output", outputs, output_count, ProgramError);
                Ok(Vec::new())
            }
        }
    };
}

/// Implements the [`BatchableOperation`](crate::BatchableOperation) trait for a [`Region`](crate::Region)-less
/// nullary [`Operation`](crate::Operation) according to the selected batching policy.
///
/// The `@replicated` policy interprets the operation once through the parent [`Context`](crate::Context) and marks
/// every output as replicated because the operation is invariant across the mapped axis. It is generic over every
/// [`Context`](crate::Context) whose type matches the operation's native [`Operation::Type`](crate::Operation::Type)
/// and every [`BatchingPolicy`](crate::BatchingPolicy) for that context.
///
/// The `@member<U, P>` policy supports a nullary operation embedded as a member of a parent type universe `U` under
/// batching policy `P`. Although the member operation is nullary in its native type universe, its parent instruction
/// may consume representation operands. The generated rule requires every such operand to be replicated, binds the
/// member operation once in the parent context, and marks every output as replicated. Neither policy assumes a
/// concrete type, value, batch carrier, or batching implementation. Nullary operations whose result depends on the
/// mapped axis, such as [`AxisIndexOperation`](crate::AxisIndexOperation), require a custom batching rule instead.
///
/// The optional leading generic list declares operation-specific type parameters. Behavioral bounds for `@replicated`
/// are derived from [`InterpretableOperation<C>`](crate::InterpretableOperation). An optional `where` clause can
/// provide bounds required to make the operation type itself well-formed.
///
/// # Parameters
///
///   - `@replicated`: Selects batching that evaluates the operation once and marks every output as replicated.
///   - `@member<U, P>`: Selects parent-universe member batching under batching policy `P`, requiring all
///     representation operands to be replicated.
///   - `$generic`: Optional operation-specific type parameters used by `$operation`.
///   - `$operation`: Regionless nullary operation type for which the implementation is generated.
///   - `$bounds`: Optional bounds required to make `$operation` well-formed.
#[macro_export]
macro_rules! impl_nullary_batchable_operation {
    // This branch accepts a generic member operation with additional well-formedness bounds.
    (@member<$parent:ty, $policy:ty> <$($generic:ident),+> $operation:ty where $($bounds:tt)+) => {
        $crate::impl_nullary_batchable_operation!(
            @impl_member [$($generic),+] ($parent) ($policy) ($operation) { $($bounds)+ }
        );
    };

    // This branch accepts a generic member operation whose type supplies all required bounds.
    (@member<$parent:ty, $policy:ty> <$($generic:ident),+> $operation:ty $(,)?) => {
        $crate::impl_nullary_batchable_operation!(
            @impl_member [$($generic),+] ($parent) ($policy) ($operation) {}
        );
    };

    // This branch accepts the common non-generic member operation form.
    (@member<$parent:ty, $policy:ty> $operation:ty $(,)?) => {
        $crate::impl_nullary_batchable_operation!(@impl_member [] ($parent) ($policy) ($operation) {});
    };

    // This internal helper emits the parent-universe member implementation shared by every public invocation form.
    (@impl_member [$($generic:ident),*] ($parent:ty) ($policy:ty) ($operation:ty) { $($bounds:tt)* }) => {
        impl<__C: $crate::Context<Type = $parent> $(, $generic)*>
            $crate::MemberBatchableOperation<__C, $policy>
            for $operation
        where
            $policy: $crate::BatchingPolicy<__C>,
            __C::Operation: From<$operation>,
            $($bounds)*
        {
            #[inline]
            fn batch_in_parent<__D: $crate::BatchingDriver<__C, $policy>>(
                &self,
                context: &$crate::BatchingContext<__C, $policy>,
                _driver: &__D,
                inputs: &[<$policy as $crate::BatchingPolicy<__C>>::Batch],
            ) -> Result<$crate::BatchedOutputs<__C, $policy>, $crate::BatchingError> {
                for (index, input) in inputs.iter().enumerate() {
                    let axis = <$policy as $crate::BatchingPolicy<__C>>::batch_axis(input);
                    if !axis.is_replicated() {
                        return Err($crate::BatchingError::UnsupportedOperation {
                            message: format!(
                                "member operand {} of type {} must be replicated but is mapped at {}",
                                index,
                                <$policy as $crate::BatchingPolicy<__C>>::unbatched_type(input),
                                axis,
                            ),
                        });
                    }
                }
                let inputs = inputs
                    .iter()
                    .map(|input| <$policy as $crate::BatchingPolicy<__C>>::value(input).clone())
                    .collect::<Vec<_>>();
                Ok(context
                    .parent()
                    .bind(self.clone(), Vec::new(), inputs.as_slice())?
                    .into_iter()
                    .map(<$policy as $crate::BatchingPolicy<__C>>::replicated)
                    .collect::<Vec<_>>()
                    .into())
            }
        }
    };

    // This branch accepts a generic replicated operation with additional well-formedness bounds.
    (@replicated <$($generic:ident),+> $operation:ty where $($bounds:tt)+) => {
        $crate::impl_nullary_batchable_operation!(@impl_replicated [$($generic),+] ($operation) { $($bounds)+ });
    };

    // This branch accepts a generic replicated operation whose interpretation supplies all required bounds.
    (@replicated <$($generic:ident),+> $operation:ty $(,)?) => {
        $crate::impl_nullary_batchable_operation!(@impl_replicated [$($generic),+] ($operation) {});
    };

    // This branch accepts the common non-generic replicated operation form.
    (@replicated $operation:ty $(,)?) => {
        $crate::impl_nullary_batchable_operation!(@impl_replicated [] ($operation) {});
    };

    // This internal helper emits the replicated batching implementation shared by every public invocation form.
    (@impl_replicated [$($generic:ident),*] ($operation:ty) { $($bounds:tt)* }) => {
        impl<__C: $crate::Context, __P: $crate::BatchingPolicy<__C> $(, $generic)*>
            $crate::BatchableOperation<__C, __P>
            for $operation
        where
            $operation: $crate::Operation<Type = __C::Type> + $crate::InterpretableOperation<__C>,
            $($bounds)*
        {
            #[inline]
            fn batch<__D: $crate::BatchingDriver<__C, __P>>(
                &self,
                context: &$crate::BatchingContext<__C, __P>,
                _driver: &__D,
                inputs: &[__P::Batch],
            ) -> Result<$crate::BatchedOutputs<__C, __P>, $crate::BatchingError> {
                $crate::check_count!("input", inputs, 0, ProgramError);
                Ok($crate::InterpretableOperation::interpret(
                    self,
                    context.parent(),
                    &$crate::EmptyRegionDriver,
                    &[],
                )?
                .into_iter()
                .map(__P::replicated)
                .collect::<Vec<_>>()
                .into())
            }
        }
    };
}

/// Implements a foreign `std::ops` operator trait as panicking sugar for the four core transform tracer types (i.e.,
/// [`Tracer`](crate::Tracer), [`PartialTracer`](crate::PartialTracer), [`BatchingTracer`](crate::BatchingTracer), and
/// [`DifferentiationTracer`](crate::DifferentiationTracer)) and for [`ProjectedValue`](crate::ProjectedValue) member
/// views by binding the operation through each value's own context.
///
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
/// Binary invocations use a fallible capability whose provider selects the operation for the value's
/// [`Type`](crate::Type) family. Provider-construction failures poison ordinary staged tracers and fail immediately
/// for direct-binding transform tracers and projected member views, preserving each value family's established error
/// behavior.
///
/// # Parameters
///
///   - `@unary` / `@binary`: Selects the operator shape to stamp out.
///     `@unary` produces `fn(self) -> Self` operators and `@binary` produces `fn(self, Self) -> Self` operators.
///   - `$trait`: Path to the foreign `std::ops` operator trait to implement (e.g., `std::ops::Add`).
///   - `$method`: Identifier of the operator trait method to define (e.g., `add`).
///   - `capability = $capability, method = $capability_method`: Fallible value capability and method that implement
///     the binary operator. Operator sugar delegates to this pair so operation selection and error behavior remain
///     defined in one place.
///   - `$message`: Panic message used when a unary tracer bind fails.
#[macro_export]
macro_rules! define_tracer_operator {
    // This branch implements receiver-only operator syntax for every transform tracer family.
    (@unary $trait:path, $method:ident, $operation:ident, $message:literal $(,)?) => {
        impl<__T: $crate::Type, __V> $trait for $crate::ProjectedValue<__T, __V>
        where
            $crate::ProjectedValue<__T, __V>: $crate::Value<Type = __T>,
            <$crate::ProjectedValue<__T, __V> as $crate::Value>::DispatchDomain: $crate::Context<
                    Type = __T,
                    Value = $crate::ProjectedValue<__T, __V>,
                    Operation: ::std::convert::From<$operation<__T>>,
                >,
        {
            type Output = Self;

            #[inline]
            fn $method(self) -> Self {
                $crate::Context::bind(
                    &$crate::Value::dispatch_domain(&self),
                    $operation::<__T>::new(),
                    Vec::new(),
                    ::std::slice::from_ref(&self),
                )
                .expect($message)
                .remove(0)
            }
        }

        impl<__C: $crate::StagingContext<Operation: ::std::convert::From<$operation<__C::Type>>>> $trait
            for $crate::Tracer<__C>
        {
            type Output = Self;

            #[inline]
            fn $method(self) -> Self {
                self.unary($operation::new())
            }
        }

        impl<__C: $crate::Context> $trait for $crate::PartialTracer<__C>
        where
            $crate::PartialEvaluationContext<__C>: $crate::Context<
                    Value = $crate::PartialTracer<__C>,
                    Operation: ::std::convert::From<$operation<__C::Type>>,
                >,
        {
            type Output = Self;

            #[inline]
            fn $method(self) -> Self {
                $crate::Context::bind(self.context(), $operation::new(), Vec::new(), ::std::slice::from_ref(&self))
                    .expect($message)
                    .remove(0)
            }
        }

        impl<__C: $crate::Context<Type = $crate::arrays::ArrayType>> $trait
            for $crate::BatchingTracer<__C, $crate::ArrayBatching>
        where
            $crate::BatchingContext<__C, $crate::ArrayBatching>: $crate::Context<
                    Value = $crate::BatchingTracer<__C, $crate::ArrayBatching>,
                    Operation: ::std::convert::From<$operation<$crate::arrays::ArrayType>>,
                >,
        {
            type Output = Self;

            #[inline]
            fn $method(self) -> Self {
                $crate::Context::bind(self.context(), $operation::new(), Vec::new(), ::std::slice::from_ref(&self))
                    .expect($message)
                    .remove(0)
            }
        }

        impl<__C: $crate::Context> $trait for $crate::DifferentiationTracer<__C>
        where
            $crate::DifferentiationContext<__C>: $crate::Context<
                    Value = $crate::DifferentiationTracer<__C>,
                    Operation: ::std::convert::From<$operation<__C::Type>>,
                >,
        {
            type Output = Self;

            #[inline]
            fn $method(self) -> Self {
                $crate::Context::bind(self.context(), $operation::new(), Vec::new(), ::std::slice::from_ref(&self))
                    .expect($message)
                    .remove(0)
            }
        }
    };

    // This branch layers panicking binary operator sugar over one fallible value capability for every tracer family.
    (
        @binary $trait:path,
        $method:ident,
        capability = $capability:path,
        method = $capability_method:ident $(,)?
    ) => {
        impl<__T: $crate::Type, __V> $trait for $crate::ProjectedValue<__T, __V>
        where
            $crate::ProjectedValue<__T, __V>: $crate::Value<Type = __T> + $capability,
        {
            type Output = Self;

            #[inline]
            fn $method(self, right: Self) -> Self {
                <Self as $capability>::$capability_method(&self, &right).unwrap_or_else(|error| panic!("{error}"))
            }
        }

        impl<__C: $crate::StagingContext> $trait for $crate::Tracer<__C>
        where
            Self: $capability,
        {
            type Output = Self;

            #[inline]
            fn $method(self, right: Self) -> Self {
                let left_type = $crate::Typed::r#type(&self);
                match <Self as $capability>::$capability_method(&self, &right) {
                    Ok(output) => output,
                    Err(error) => {
                        $crate::StagingContext::error(self.context(), error);
                        $crate::Tracer::new(self.context().clone(), $crate::TracerState::Poison, left_type.into_owned())
                    }
                }
            }
        }

        impl<__C: $crate::Context> $trait for $crate::PartialTracer<__C>
        where
            Self: $capability,
        {
            type Output = Self;

            #[inline]
            fn $method(self, right: Self) -> Self {
                <Self as $capability>::$capability_method(&self, &right).unwrap_or_else(|error| panic!("{error}"))
            }
        }

        impl<__C: $crate::Context<Type = $crate::arrays::ArrayType>> $trait
            for $crate::BatchingTracer<__C, $crate::ArrayBatching>
        where
            Self: $capability,
        {
            type Output = Self;

            #[inline]
            fn $method(self, right: Self) -> Self {
                <Self as $capability>::$capability_method(&self, &right).unwrap_or_else(|error| panic!("{error}"))
            }
        }

        impl<__C: $crate::Context> $trait for $crate::DifferentiationTracer<__C>
        where
            Self: $capability,
        {
            type Output = Self;

            #[inline]
            fn $method(self, right: Self) -> Self {
                <Self as $capability>::$capability_method(&self, &right).unwrap_or_else(|error| panic!("{error}"))
            }
        }
    };
}

/// Checks exact type-inference behavior for a concrete, regionless [`Operation`](crate::Operation). Each ordinary case
/// declares `input_types` and either the expected `output_types` or exact [`TypeError`](crate::TypeError) message.
/// Cases are expanded independently, so one invocation may cover different [`Type`](crate::Type) families such as
/// [`DataType`](crate::DataType) and [`ArrayType`](crate::ArrayType). A case whose input and output types do not
/// identify the family, most commonly an empty-input error case, may declare it explicitly with `type = ...`. The
/// `@elementwise @unary` and `@elementwise @binary` selectors check an elementwise operation's data-type inference
/// together with its lifting to representative, metadata-bearing [`ArrayType`](crate::ArrayType)s. Their successful
/// cases declare input element data types and the expected output element data type; rejected cases declare the exact
/// data-type inference diagnostic. The macro also provides selectors for shared sharding rejection contracts whose
/// generated mesh fixtures would otherwise obscure the behavior under test:
///
///   - `@elementwise @unary`: Checks unary element-type inference and successful array-type lifting.
///   - `@elementwise @binary`: Checks binary element-type inference and successful array-type lifting.
///   - `@reject @unreduced`: Checks that array inputs carrying an unreduced mesh axis are rejected.
///   - `@reject @mismatched_reduced`: Checks both operand orders for a binary operation whose operands
///     must carry the same reduced-axis markers.
///
/// # Examples
///
/// This example checks both successful and rejected type inference for the elementwise
/// [`AddOperation`](crate::AddOperation):
///
/// ```rust
/// # use ryft_core::arrays::DataType;
/// # use ryft_core::{AddOperation, check_operation_type_inference};
///
/// check_operation_type_inference!(
///     @elementwise @binary,
///     operation = AddOperation,
///     cases = [
///         {
///             input_data_types = [DataType::F32, DataType::F64],
///             output_data_types = [DataType::F64],
///         },
///         {
///             input_data_types = [DataType::Boolean, DataType::Boolean],
///             error = "`add` does not support input data type bool",
///         },
///     ],
/// );
/// ```
///
/// This example uses a generated sharding fixture to check the unreduced-input rejection contract
/// of [`SinOperation`](crate::SinOperation):
///
/// ```rust
/// # use ryft_core::arrays::{ArrayType, DataType};
/// # use ryft_core::{SinOperation, check_operation_type_inference};
///
/// check_operation_type_inference!(
///     @reject @unreduced,
///     operation = SinOperation::<ArrayType>::new(),
///     input_types = [ArrayType::scalar(DataType::F64)],
/// );
/// ```
///
/// # Parameters
///
///   - `$selector`: Optional generated type-inference contract: `@elementwise @unary`, `@elementwise @binary`,
///     `@reject @unreduced`, or `@reject @mismatched_reduced`.
///   - `operation = $operation`: [`Operation`](crate::Operation) expression evaluated once per macro invocation.
///   - `cases = $cases`: Type-inference test cases. Ordinary cases use `input_types` and either `output_types` or
///     `error`; elementwise cases use the corresponding element-data-type fields documented below.
///   - `type = $type`: Optional explicit [`Type`](crate::Type) family for a case whose other expressions cannot
///     determine the operation implementation.
///   - `input_types = $input_types`: Input types used by one ordinary case or by the selected generated rejection
///     check, in operation input order.
///   - `output_types = $output_types`: Expected output types for one successful ordinary case, in result order.
///   - `error = $error`: Exact [`TypeError`](crate::TypeError) message expected from one rejected ordinary case.
///   - `input_data_types = $input_data_types`: Input element data types used by an elementwise case. Unary cases
///     contain one data type, while binary cases contain two.
///   - `output_data_types = $output_data_types`: Expected output element data types for a successful elementwise
///     case, in result order.
#[macro_export]
macro_rules! check_operation_type_inference {
    // This branch checks unary element-type cases and lifts every successful case through a representative array type.
    // It leaves rejected cases in the data-type universe because custom array inference may own a more specific error.
    (
        @elementwise @unary,
        operation = $operation:ident,
        cases = [$( { $($case:tt)* } ),+ $(,)?] $(,)?
    ) => {{
        let data_operation = $operation::<$crate::arrays::DataType>::new();
        $($crate::check_operation_type_inference!(
            @elementwise_unary_case data_operation, $operation, { $($case)* }
        );)+
    }};

    // This branch checks binary element-type cases and lifts every successful case through matching representative
    // array structures. Using matching structures keeps the contract valid for elementwise operations that deliberately
    // impose stricter array relationships than ordinary broadcasting.
    (
        @elementwise @binary,
        operation = $operation:ident,
        cases = [$( { $($case:tt)* } ),+ $(,)?] $(,)?
    ) => {{
        let data_operation = $operation::<$crate::arrays::DataType>::new();
        $($crate::check_operation_type_inference!(
            @elementwise_binary_case data_operation, $operation, { $($case)* }
        );)+
    }};

    // This branch checks one or more exact regionless type-inference cases. It evaluates the operation once and
    // delegates each heterogeneous success or error case to an internal assertion branch.
    (
        operation = $operation:expr,
        cases = [$( { $($case:tt)* } ),+ $(,)?] $(,)?
    ) => {{
        let operation = $operation;
        $($crate::check_operation_type_inference!(@case operation, { $($case)* });)+
    }};

    // This branch generates an unreduced sharding fixture for every declared input and checks the standard
    // rejection used by operations that cannot consume partial sums.
    (
        @reject @unreduced,
        operation = $operation:expr,
        input_types = [$($input_type:expr),+ $(,)?] $(,)?
    ) => {{
        let operation = $operation;
        let descriptor = $crate::programs::operations::Operation::name(&operation);
        let mesh = $crate::arrays::sharding::LogicalMesh::new(vec![
            $crate::arrays::sharding::MeshAxis::new("x", 2, $crate::arrays::sharding::MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let input_types = vec![$($input_type),+]
            .into_iter()
            .map(|input_type| {
                let dimensions = vec![$crate::arrays::sharding::ShardingDimension::Replicated; input_type.rank()];
                let sharding = $crate::arrays::sharding::Sharding::new(mesh.clone(), dimensions)
                    .unwrap()
                    .with_unreduced_axes(["x"])
                    .unwrap();
                input_type.with_sharding(sharding).unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            $crate::programs::operations::Operation::infer_output_types(&operation, input_types.as_slice(), &[]),
            Err($crate::programs::types::TypeError::invalid(format!(
                "`{descriptor}` does not support unreduced operands",
            ))),
        );
    }};

    // This branch generates both operand orderings with mismatched reduced-axis markers and checks that
    // a binary operation rejects each ordering with the same diagnostic.
    (
        @reject @mismatched_reduced,
        operation = $operation:expr,
        input_types = [$left_type:expr, $right_type:expr $(,)?] $(,)?
    ) => {{
        let operation = $operation;
        let descriptor = $crate::programs::operations::Operation::name(&operation);
        let mesh = $crate::arrays::sharding::LogicalMesh::new(vec![
            $crate::arrays::sharding::MeshAxis::new("x", 2, $crate::arrays::sharding::MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let plain = |input_type: $crate::arrays::ArrayType| {
            let dimensions = vec![$crate::arrays::sharding::ShardingDimension::Replicated; input_type.rank()];
            input_type
                .with_sharding($crate::arrays::sharding::Sharding::new(mesh.clone(), dimensions).unwrap())
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
        let expected = Err($crate::programs::types::TypeError::invalid(format!(
            "`{descriptor}` operands must be reduced over the same axes",
        )));
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

    // This internal branch checks a successful explicitly typed case. The type annotation disambiguates applications
    // with no input or output expressions from which Rust could infer the operation's `Type` family.
    (
        @case $operation:ident,
        {
            type = $type:ty,
            input_types = [$($input_type:expr),* $(,)?],
            output_types = [$($output_type:expr),* $(,)?] $(,)?
        }
    ) => {{
        let input_types = [$($input_type),*];
        assert_eq!(
            $crate::programs::operations::Operation::infer_output_types(
                &$operation,
                input_types.as_slice(),
                &[],
            ),
            Ok(vec![$($output_type),*]),
        );
    }};

    // This internal branch checks a successful unary element-type case in both the data-type and array-type universes.
    // The array fixture carries shape, layout, and memory metadata so the assertion also verifies structural lifting.
    (
        @elementwise_unary_case $data_operation:ident, $operation:ident,
        {
            input_data_types = [$input_data_type:expr $(,)?],
            output_data_types = [$($output_data_type:expr),* $(,)?] $(,)?
        }
    ) => {{
        let array_operation = $operation::<$crate::arrays::ArrayType>::new();
        let input_data_type = $input_data_type;
        let output_data_types: ::std::vec::Vec<$crate::arrays::DataType> = ::std::vec![$($output_data_type),*];
        assert_eq!(
            $crate::programs::operations::Operation::infer_output_types(
                &$data_operation,
                &[input_data_type],
                &[],
            ),
            Ok(output_data_types.clone()),
        );
        let input_type = $crate::arrays::ArrayType::new(
            input_data_type,
            $crate::arrays::Shape::new(vec![
                $crate::arrays::Dimension::Static(2),
                $crate::arrays::Dimension::Static(3),
            ]),
        )
        .with_layout($crate::arrays::Layout::Strided($crate::arrays::StridedLayout::new(vec![3, 1])))
        .with_memory($crate::arrays::Memory::Host { pinned: true });
        assert_eq!(
            $crate::programs::operations::Operation::infer_output_types(
                &array_operation,
                ::std::slice::from_ref(&input_type),
                &[],
            ),
            Ok(output_data_types
                .into_iter()
                .map(|output_data_type| input_type.clone().with_data_type(output_data_type))
                .collect::<::std::vec::Vec<_>>()),
        );
    }};

    // This internal branch checks a rejected unary element-type case and compares the exact operation-owned diagnostic.
    (
        @elementwise_unary_case $data_operation:ident, $operation:ident,
        {
            input_data_types = [$input_data_type:expr $(,)?],
            error = $message:expr $(,)?
        }
    ) => {
        $crate::check_operation_type_inference!(@case $data_operation, {
            input_types = [$input_data_type],
            error = $message,
        });
    };

    // This internal branch checks a successful binary element-type case in both universes. Matching array structures
    // isolate element-type lifting from the separately tested shared broadcasting contract.
    (
        @elementwise_binary_case $data_operation:ident, $operation:ident,
        {
            input_data_types = [$left_data_type:expr, $right_data_type:expr $(,)?],
            output_data_types = [$($output_data_type:expr),* $(,)?] $(,)?
        }
    ) => {{
        let array_operation = $operation::<$crate::arrays::ArrayType>::new();
        let left_data_type = $left_data_type;
        let right_data_type = $right_data_type;
        let output_data_types: ::std::vec::Vec<$crate::arrays::DataType> = ::std::vec![$($output_data_type),*];
        assert_eq!(
            $crate::programs::operations::Operation::infer_output_types(
                &$data_operation,
                &[left_data_type, right_data_type],
                &[],
            ),
            Ok(output_data_types.clone()),
        );
        let left_type = $crate::arrays::ArrayType::new(
            left_data_type,
            $crate::arrays::Shape::new(vec![
                $crate::arrays::Dimension::Static(2),
                $crate::arrays::Dimension::Static(3),
            ]),
        )
        .with_layout($crate::arrays::Layout::Strided($crate::arrays::StridedLayout::new(vec![3, 1])))
        .with_memory($crate::arrays::Memory::Host { pinned: true });
        let right_type = left_type.clone().with_data_type(right_data_type);
        assert_eq!(
            $crate::programs::operations::Operation::infer_output_types(
                &array_operation,
                &[left_type.clone(), right_type],
                &[],
            ),
            Ok(output_data_types
                .into_iter()
                .map(|output_data_type| left_type.clone().with_data_type(output_data_type))
                .collect::<::std::vec::Vec<_>>()),
        );
    }};

    // This internal branch checks a rejected binary element-type case and compares the exact operation-owned diagnostic.
    (
        @elementwise_binary_case $data_operation:ident, $operation:ident,
        {
            input_data_types = [$left_data_type:expr, $right_data_type:expr $(,)?],
            error = $message:expr $(,)?
        }
    ) => {
        $crate::check_operation_type_inference!(@case $data_operation, {
            input_types = [$left_data_type, $right_data_type],
            error = $message,
        });
    };

    // This internal branch checks a rejected explicitly typed case. It provides the same disambiguation as the typed
    // success branch while retaining exact comparison of the operation-owned diagnostic.
    (
        @case $operation:ident,
        {
            type = $type:ty,
            input_types = [$($input_type:expr),* $(,)?],
            error = $message:expr $(,)?
        }
    ) => {{
        let input_types = [$($input_type),*];
        assert_eq!(
            $crate::programs::operations::Operation::infer_output_types(
                &$operation,
                input_types.as_slice(),
                &[],
            ),
            Err($crate::programs::types::TypeError::invalid(
                ::core::convert::Into::<::std::string::String>::into($message),
            )),
        );
    }};

    // This internal branch checks a successful ordinary case by inferring with no attached regions and comparing the
    // complete ordered output-type vector. Keeping success separate lets callers use natural `output_types` syntax.
    (
        @case $operation:ident,
        {
            input_types = [$($input_type:expr),* $(,)?],
            output_types = [$($output_type:expr),* $(,)?] $(,)?
        }
    ) => {{
        let input_types = [$($input_type),*];
        assert_eq!(
            $crate::programs::operations::Operation::infer_output_types(
                &$operation,
                input_types.as_slice(),
                &[],
            ),
            Ok(vec![$($output_type),*]),
        );
    }};

    // This internal branch checks a rejected ordinary case against its complete `TypeError` message. It is distinct
    // from the success branch so diagnostics remain exact without making callers construct the error wrapper.
    (
        @case $operation:ident,
        {
            input_types = [$($input_type:expr),* $(,)?],
            error = $message:expr $(,)?
        }
    ) => {{
        let input_types = [$($input_type),*];
        assert_eq!(
            $crate::programs::operations::Operation::infer_output_types(
                &$operation,
                input_types.as_slice(),
                &[],
            ),
            Err($crate::programs::types::TypeError::invalid(
                ::core::convert::Into::<::std::string::String>::into($message),
            )),
        );
    }};
}

/// Checks how a concrete [`Operation`](crate::Operation) behaves under partial evaluation. The concise `inputs`
/// and `expected` form checks the default rule for a single-output operation under which all-known inputs fold the
/// operation, every individual unknown-input position residualizes it, and an all-unknown input set does the same.
/// The explicit form builds a one-instruction program for each case and checks its output classification, residual
/// instruction count, and replayed values. Inputs use `(@known, value)` or `(@unknown(type = ..., replay = ...))`.
/// Outputs use `(@known, value)` or `(@residual, value)`. The default form uses the eager [`Array`](crate::Array)
/// reference backend, while `backend = (Value, Operation)` supports other value and operation families.
///
/// # Examples
///
/// This is an example of how to use this macro to check the elementwise [`NegOperation`](crate::NegOperation):
///
/// ```rust
/// # use ryft_core::{Array, NegOperation, check_operation_partial_evaluation};
/// check_operation_partial_evaluation!(
///     operation = NegOperation::new(),
///     inputs = [Array::scalar(2.0)],
///     expected = Array::scalar(-2.0),
/// );
/// ```
///
/// For more control, explicit cases specify the partial-evaluation state of every input and expected output. In the
/// following example, the left input is known while the right input is unknown. Partial evaluation must therefore leave
/// one residual instruction and classify its output as residual. The `replay` value is then supplied for the unknown
/// input when interpreting that residual program, and the resulting output is compared with the declared value:
///
/// ```rust
/// # use ryft_core::arrays::{ArrayType, DataType};
/// # use ryft_core::{AddOperation, Array, check_operation_partial_evaluation};
///
/// check_operation_partial_evaluation!(
///     operation = AddOperation::new(),
///     cases = [{
///         inputs = [
///             (@known, Array::scalar(2.0)),
///             (@unknown(type = ArrayType::scalar(DataType::F64), replay = Array::scalar(3.0))),
///         ],
///         outputs = [(@residual, Array::scalar(5.0))],
///         residual_instructions = 1,
///     }],
/// );
/// ```
///
/// # Parameters
///
///   - `operation = $operation`: [`Operation`](crate::Operation) expression evaluated once per macro invocation.
///   - `inputs = $inputs`: Concrete inputs checked by the concise default-rule form.
///   - `expected = $expected`: Expected concrete output of the concise default-rule form.
///   - `cases = $cases`: Explicit partial-evaluation cases, including expected output classifications and residual
///     instruction counts.
///   - `backend = ($value, $operation_family)`: Optional value and operation-family types used to construct programs
///     for downstream operation families.
#[macro_export]
macro_rules! check_operation_partial_evaluation {
    // This branch checks the standard fold-or-residualize contract using the array reference backend.
    (
        operation = $operation:expr,
        inputs = [$($input:expr),+ $(,)?],
        expected = $expected:expr $(,)?
    ) => {{
        let operation = $operation;
        let inputs: Vec<$crate::Array> = vec![$($input),+];
        let expected: $crate::Array = $expected;
        let mut builder = $crate::programs::builders::ProgramBuilder::<
            $crate::Array,
            $crate::ArrayOperation<$crate::Array>,
        >::new();
        let input_ids = inputs
            .iter()
            .map(|input| builder.add_input($crate::programs::types::Typed::r#type(input).into_owned()))
            .collect::<Vec<_>>();
        let operation: $crate::ArrayOperation<$crate::Array> =
            ::core::convert::Into::into(operation);
        let output_ids = builder.add_instruction(operation, Vec::new(), input_ids, None).unwrap().to_vec();
        assert_eq!(output_ids.len(), 1);
        let program = builder
            .build::<
                Vec<$crate::Array>,
                Vec<$crate::Array>,
            >(
                output_ids,
                vec![$crate::parameters::Placeholder; inputs.len()],
                vec![$crate::parameters::Placeholder],
            )
            .unwrap();
        let context = $crate::contexts::EagerContext::<
            $crate::Array,
            $crate::ArrayOperation<$crate::Array>,
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

    // This branch forwards explicit cases to the array reference backend form.
    (
        operation = $operation:expr,
        cases = $cases:tt $(,)?
    ) => {
        $crate::check_operation_partial_evaluation!(
            backend = (
                $crate::Array,
                $crate::ArrayOperation<$crate::Array>
            ),
            operation = $operation,
            cases = $cases,
        )
    };

    // This branch executes explicit known/residual cases against a caller-selected backend family.
    (
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
                vec![$($crate::check_operation_partial_evaluation!(@partial_input $value, $input)),*];
            let expected_outputs: Vec<(bool, $value)> =
                vec![$($crate::check_operation_partial_evaluation!(@partial_output $output)),*];
            let mut builder = $crate::programs::builders::ProgramBuilder::<$value, $operation_family>::new();
            let input_ids = inputs
                .iter()
                .map(|(input, _)| {
                    builder.add_input($crate::programs::types::Typed::r#type(input).into_owned())
                })
                .collect::<Vec<_>>();
            let operation: $operation_family = ::core::convert::Into::into(operation.clone());
            let output_ids = builder.add_instruction(operation, Vec::new(), input_ids, None).unwrap().to_vec();
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

    // This internal branch converts one known input declaration into its partial value and no replay input.
    (@partial_input $value:ty, (@known, $input:expr)) => {{
        let input: $value = ::core::convert::Into::into($input);
        ($crate::partial::PartialValue::Known(input), Option::<$value>::None)
    }};

    // This internal branch converts one unknown input declaration and retains its concrete replay value.
    (@partial_input $value:ty, (@unknown(type = $r#type:expr, replay = $input:expr))) => {{
        let input: $value = ::core::convert::Into::into($input);
        ($crate::partial::PartialValue::Unknown($r#type), Some(input))
    }};

    // This internal branch records that an expected output should fold to a known value.
    (@partial_output (@known, $output:expr)) => {
        (true, ::core::convert::Into::into($output))
    };

    // This internal branch records that an expected output should remain in the residual program.
    (@partial_output (@residual, $output:expr)) => {
        (false, ::core::convert::Into::into($output))
    };
}

/// Checks how a concrete [`Operation`](crate::Operation) behaves under batching. Each input and expected output
/// is written as `(@mapped(axis = ...), value)` or `(@replicated, value)`, so the lists encode operation arity
/// and batch placement directly. Use `@exact` to compare complete [`ArrayBatch`](crate::ArrayBatch) values or
/// `@approx(epsilon = ...)` to compare `f64` payloads approximately while still checking output types
/// and batch axes exactly. The default form uses the eager [`Array`](crate::Array) reference context, an
/// [`EmptyRegionDriver`](crate::EmptyRegionDriver), and replicated mapped-axis sharding. The extended form accepts a
/// custom `context`, `driver`, and `axis_sharding`. [`Region`](crate::Region)-ful operations may use that form with
/// their [`Instruction`](crate::Instruction)-scoped driver, but tests whose main subject is nested-region
/// transformation should generally keep that setup explicit.
///
/// # Example
///
/// This is an example of how to use this macro to check the elementwise [`AddOperation`](crate::AddOperation):
///
/// ```rust
/// # use ryft_core::{Array, AddOperation, check_operation_batching};
/// check_operation_batching!(
///     @exact,
///     operation = AddOperation::new(),
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
///   - `$selector`: Output comparison to perform: `@exact` or `@approx(epsilon = ...)`.
///   - `context = $context`: Optional parent [`Context`](crate::Context) for the extended batching form.
///   - `driver = $driver`: Optional [`BatchingDriver`](crate::BatchingDriver) for the extended batching form.
///   - `operation = $operation`: [`Operation`](crate::Operation) expression evaluated once per macro invocation.
///   - `axis_size = $axis_size`: Dimension of the mapped batching axis. It remains explicit because no mapped input exists
///     from which to infer it in an all-replicated case.
///   - `axis_sharding = $axis_sharding`: Optional [`ShardingDimension`](crate::ShardingDimension) assigned to the
///     mapped axis by the extended form. It appears immediately after `axis_size` because both arguments describe the
///     transform-owned mapped axis.
///   - `cases = $cases`: Batching cases declaring their inputs and expected outputs.
#[macro_export]
macro_rules! check_operation_batching {
    // This branch runs exact comparisons with the default eager array context and replicated axis sharding.
    (
        @exact,
        operation = $operation:expr,
        axis_size = $axis_size:expr,
        cases = $cases:tt $(,)?
    ) => {
        $crate::check_operation_batching!(
            @run (@exact),
            context = $crate::contexts::EagerContext::<
                $crate::Array,
                $crate::ArrayOperation<$crate::Array>,
            >::new(),
            driver = &$crate::programs::regions::EmptyRegionDriver,
            operation = $operation,
            axis_size = $axis_size,
            axis_sharding = $crate::arrays::sharding::ShardingDimension::Replicated,
            cases = $cases,
        )
    };

    // This branch runs approximate comparisons with the default eager array context and replicated axis sharding.
    (
        @approx(epsilon = $epsilon:expr),
        operation = $operation:expr,
        axis_size = $axis_size:expr,
        cases = $cases:tt $(,)?
    ) => {
        $crate::check_operation_batching!(
            @run (@approx($epsilon)),
            context = $crate::contexts::EagerContext::<
                $crate::Array,
                $crate::ArrayOperation<$crate::Array>,
            >::new(),
            driver = &$crate::programs::regions::EmptyRegionDriver,
            operation = $operation,
            axis_size = $axis_size,
            axis_sharding = $crate::arrays::sharding::ShardingDimension::Replicated,
            cases = $cases,
        )
    };

    // This branch accepts an explicit context, driver, and axis sharding for exact comparisons.
    (
        @exact,
        context = $context:expr,
        driver = $driver:expr,
        operation = $operation:expr,
        axis_size = $axis_size:expr,
        axis_sharding = $axis_sharding:expr,
        cases = $cases:tt $(,)?
    ) => {
        $crate::check_operation_batching!(
            @run (@exact),
            context = $context,
            driver = $driver,
            operation = $operation,
            axis_size = $axis_size,
            axis_sharding = $axis_sharding,
            cases = $cases,
        )
    };

    // This branch accepts an explicit context, driver, and axis sharding for approximate comparisons.
    (
        @approx(epsilon = $epsilon:expr),
        context = $context:expr,
        driver = $driver:expr,
        operation = $operation:expr,
        axis_size = $axis_size:expr,
        axis_sharding = $axis_sharding:expr,
        cases = $cases:tt $(,)?
    ) => {
        $crate::check_operation_batching!(
            @run (@approx($epsilon)),
            context = $context,
            driver = $driver,
            operation = $operation,
            axis_size = $axis_size,
            axis_sharding = $axis_sharding,
            cases = $cases,
        )
    };

    // This internal branch constructs the batching context, executes every case, and dispatches output comparison.
    (
        @run $comparison:tt,
        context = $context:expr,
        driver = $driver:expr,
        operation = $operation:expr,
        axis_size = $axis_size:expr,
        axis_sharding = $axis_sharding:expr,
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
        let context =
            $crate::batching::BatchingContext::<_, $crate::ArrayBatching>::new($context, axis_size)
            .with_axis_sharding(axis_sharding);
        $(
            let inputs = vec![$($crate::check_operation_batching!(@batch_value $input)),*];
            let expected_outputs = vec![$($crate::check_operation_batching!(@batch_value $output)),*];
            let (actual_outputs, _) = $crate::batching::BatchableOperation::batch(
                &operation,
                &context,
                driver,
                inputs.as_slice(),
            )
            .unwrap()
            .into_parts();
            $crate::check_operation_batching!(@compare_batches $comparison, actual_outputs, expected_outputs);
        )*
    }};

    // This internal branch converts a mapped value declaration into an `ArrayBatch` at the requested physical axis.
    (@batch_value (@mapped(axis = $axis:expr), $value:expr)) => {{
        let value = $value;
        $crate::ArrayBatch::new(value, $crate::batching::BatchAxis::new($axis)).unwrap()
    }};

    // This internal branch converts a replicated value declaration into a replicated `ArrayBatch`.
    (@batch_value (@replicated, $value:expr)) => {
        $crate::ArrayBatch::replicated($value)
    };

    // This internal branch compares complete batched values exactly.
    (@compare_batches (@exact), $actual:expr, $expected:expr) => {{
        assert_eq!($actual, $expected);
    }};

    // This internal branch compares batch metadata exactly and numerical payloads within the requested tolerance.
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
}

/// Checks how a concrete [`Operation`](crate::Operation) behaves under forward-mode differentiation. Each
/// case builds a single [`Instruction`](crate::Instruction) [`Program`](crate::Program), transforms it with
/// [`Program::jvp`](crate::Program::jvp), interprets the fused primal-and-tangent program, and checks the declared
/// primal and tangent outputs. The macro independently checks the tangent outputs against the central directional
/// finite difference `(f(x + h·ẋ) - f(x - h·ẋ)) / (2h)`, so the numerical oracle never uses the differentiation rule
/// being tested. An optional `jvp` string checks the transformed program's symbolic form. The default form uses the
/// eager [`Array`](crate::Array) reference backend, while `backend = (Value, Operation)` supports downstream value and
/// operation families whose values implement the arithmetic and approximate-equality operations used by the check.
///
/// # Example
///
/// This is an example of how to use this macro to check the elementwise [`MulOperation`](crate::MulOperation):
///
/// ```rust
/// # use indoc::indoc;
/// # use ryft_core::{Array, MulOperation, check_operation_differentiation};
/// check_operation_differentiation!(
///     @approx(step = 1e-6, epsilon = 1e-6),
///     operation = MulOperation::new(),
///     cases = [{
///         primals = [Array::scalar(2.0), Array::scalar(5.0)],
///         tangents = [Array::scalar(3.0), Array::scalar(-1.0)],
///         primal_outputs = [Array::scalar(10.0)],
///         tangent_outputs = [Array::scalar(13.0)],
///         jvp = indoc! {"
///             lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
///             let %4:f64[] = mul %0 %1
///                 %5:f64[] = mul %1 %2
///                 %6:f64[] = mul %0 %3
///                 %7:f64[] = add %5 %6
///             in (%4, %7)
///         "},
///     }],
/// );
/// ```
///
/// # Parameters
///
///   - `$selector`: Numerical check configuration, written as `@approx(step = ..., epsilon = ...)`, where `step` is
///     the central finite-difference spacing and `epsilon` is the absolute comparison tolerance.
///   - `operation = $operation`: [`Operation`](crate::Operation) expression evaluated once per macro invocation.
///   - `cases = $cases`: Differentiation cases declaring primal inputs, input tangents, primal outputs, tangent outputs,
///     and an optional `jvp` rendering.
///   - `backend = ($value, $operation_family)`: Optional value and operation-family types used to construct programs
///     for a downstream operation family.
#[macro_export]
macro_rules! check_operation_differentiation {
    // This branch selects the eager array reference backend for a finite-difference differentiation check.
    (
        @approx(step = $step:expr, epsilon = $epsilon:expr),
        operation = $operation:expr,
        cases = $cases:tt $(,)?
    ) => {
        $crate::check_operation_differentiation!(
            @approx(step = $step, epsilon = $epsilon),
            backend = ($crate::Array, $crate::ArrayOperation<$crate::Array>),
            operation = $operation,
            cases = $cases,
        )
    };

    // This branch builds, differentiates, interprets, and numerically checks every case for the selected backend.
    (
        @approx(step = $step:expr, epsilon = $epsilon:expr),
        backend = ($value:ty, $operation_family:ty),
        operation = $operation:expr,
        cases = [
            $(
                {
                    primals = [$($primal:expr),+ $(,)?],
                    tangents = [$($tangent:expr),+ $(,)?],
                    primal_outputs = [$($primal_output:expr),+ $(,)?],
                    tangent_outputs = [$($tangent_output:expr),+ $(,)?]
                    $(, jvp = $jvp:expr)? $(,)?
                }
            ),+ $(,)?
        ] $(,)?) => {{
        let operation = $operation;
        let step: f64 = $step;
        let epsilon: f64 = $epsilon;
        assert!(step > 0.0, "finite-difference step must be positive");
        assert!(epsilon >= 0.0, "comparison epsilon must be nonnegative");
        $(
        {
            let primals: Vec<$value> = vec![$(::core::convert::Into::into($primal)),+];
            let tangents: Vec<$value> = vec![$(::core::convert::Into::into($tangent)),+];
            assert_eq!(primals.len(), tangents.len(), "primal and tangent input counts differ");
            let expected_primals: Vec<$value> = vec![$(::core::convert::Into::into($primal_output)),+];
            let expected_tangents: Vec<$value> = vec![$(::core::convert::Into::into($tangent_output)),+];
            assert_eq!(
                expected_primals.len(),
                expected_tangents.len(),
                "primal and tangent output counts differ",
            );

            let mut builder = $crate::programs::builders::ProgramBuilder::<$value, $operation_family>::new();
            let input_ids = primals
                .iter()
                .map(|input| builder.add_input($crate::programs::types::Typed::r#type(input).into_owned()))
                .collect::<Vec<_>>();
            let operation: $operation_family = ::core::convert::Into::into(operation.clone());
            let output_ids = builder.add_instruction(operation, Vec::new(), input_ids, None).unwrap().to_vec();
            assert_eq!(output_ids.len(), expected_primals.len(), "declared output count is incorrect");
            let output_count = output_ids.len();
            let program = builder
                .build::<Vec<$value>, Vec<$value>>(
                    output_ids,
                    vec![$crate::parameters::Placeholder; primals.len()],
                    vec![$crate::parameters::Placeholder; output_count],
                )
                .unwrap();
            let jvp = program.jvp().unwrap();
            $(assert_eq!(jvp.to_string(), $jvp.trim_end());)?

            let jvp_inputs = primals.iter().cloned().chain(tangents.iter().cloned()).collect::<Vec<_>>();
            let actual = jvp.interpret(jvp_inputs).unwrap();
            assert_eq!(actual.len(), 2 * output_count);
            let (actual_primals, actual_tangents) = actual.split_at(output_count);
            for (actual, expected) in actual_primals.iter().zip(expected_primals.iter()) {
                ::approx::assert_abs_diff_eq!(actual, expected, epsilon = epsilon);
            }
            for (actual, expected) in actual_tangents.iter().zip(expected_tangents.iter()) {
                ::approx::assert_abs_diff_eq!(actual, expected, epsilon = epsilon);
            }

            let plus_inputs = primals
                .iter()
                .cloned()
                .zip(tangents.iter().cloned())
                .map(|(primal, tangent)| primal + tangent * step)
                .collect::<Vec<_>>();
            let minus_inputs = primals
                .iter()
                .cloned()
                .zip(tangents.iter().cloned())
                .map(|(primal, tangent)| primal - tangent * step)
                .collect::<Vec<_>>();
            let plus_outputs = program.interpret(plus_inputs).unwrap();
            let minus_outputs = program.interpret(minus_inputs).unwrap();
            assert_eq!(plus_outputs.len(), output_count);
            assert_eq!(minus_outputs.len(), output_count);
            for ((actual, plus), minus) in actual_tangents
                .iter()
                .zip(plus_outputs.into_iter())
                .zip(minus_outputs.into_iter())
            {
                let estimate = (plus - minus) * (1.0 / (2.0 * step));
                ::approx::assert_abs_diff_eq!(actual, &estimate, epsilon = epsilon);
            }
        }
        )+
    }};
}

/// Checks how a concrete [`Operation`](crate::Operation) behaves under transposition. Supported cases build a single
/// [`Instruction`](crate::Instruction) [`Program`](crate::Program), classify each input as `@linear` or `@known`,
/// transpose with respect to the linear inputs, and check the interpreted input cotangents. The pullback receives
/// output cotangents followed by the concrete known inputs in source-input order, and returns cotangents in
/// linear-input order. An optional `pullback` string checks the transformed program's symbolic form. Use `@exact`
/// for complete value equality or `@approx(epsilon = ...)` for approximate equality. The `@rejected` selector checks
/// that transposition reaches the operation's unsupported-transposition error. The default forms use the eager
/// [`Array`](crate::Array) reference backend, while `backend = (Value, Operation)` supports downstream value
/// and operation families.
///
/// # Example
///
/// This is an example of how to use this macro to check the elementwise [`MulOperation`](crate::MulOperation):
///
/// ```rust
/// # use indoc::indoc;
/// # use ryft_core::arrays::{ArrayType, DataType};
/// # use ryft_core::{Array, MulOperation, check_operation_transposition};
///
/// check_operation_transposition!(
///     @exact,
///     operation = MulOperation::new(),
///     cases = [{
///         inputs = [
///             (@known, Array::scalar(4.0)),
///             (@linear(type = ArrayType::scalar(DataType::F64))),
///         ],
///         output_cotangents = [Array::scalar(3.0)],
///         input_cotangents = [Array::scalar(12.0)],
///         pullback = indoc! {"
///             lambda %0:f64[], %1:f64[] .
///             let %2:f64[] = mul %1 %0
///             in (%2)
///         "},
///     }],
/// );
/// ```
///
/// # Parameters
///
///   - `$selector`: Output comparison to perform for supported cases (`@exact` or `@approx(epsilon = ...)`), or the
///     `@rejected` unsupported-operation contract.
///   - `operation = $operation`: [`Operation`](crate::Operation) expression evaluated once per macro invocation.
///   - `cases = $cases`: Supported transposition cases declaring known and linear inputs, output cotangents, expected
///     input cotangents, and an optional `pullback` rendering.
///   - `input_types = $input_types`: Input types used to build the transposed program, in operation input order.
///   - `backend = ($value, $operation_family)`: Optional value and operation-family types used to construct a
///     transposition program for a downstream operation family.
#[macro_export]
macro_rules! check_operation_transposition {
    // This branch runs exact supported-transposition cases against the eager array reference backend.
    (
        @exact,
        operation = $operation:expr,
        cases = $cases:tt $(,)?
    ) => {
        $crate::check_operation_transposition!(
            @run (@exact),
            backend = (
                $crate::Array,
                $crate::ArrayOperation<$crate::Array>
            ),
            operation = $operation,
            cases = $cases,
        )
    };

    // This branch runs approximate supported-transposition cases against the eager array reference backend.
    (
        @approx(epsilon = $epsilon:expr),
        operation = $operation:expr,
        cases = $cases:tt $(,)?
    ) => {
        $crate::check_operation_transposition!(
            @run (@approx($epsilon)),
            backend = (
                $crate::Array,
                $crate::ArrayOperation<$crate::Array>
            ),
            operation = $operation,
            cases = $cases,
        )
    };

    // This branch accepts a caller-selected backend for exact supported-transposition cases.
    (
        @exact,
        backend = ($value:ty, $operation_family:ty),
        operation = $operation:expr,
        cases = $cases:tt $(,)?) => {
        $crate::check_operation_transposition!(
            @run (@exact),
            backend = ($value, $operation_family),
            operation = $operation,
            cases = $cases,
        )
    };

    // This branch accepts a caller-selected backend for approximate supported-transposition cases.
    (
        @approx(epsilon = $epsilon:expr),
        backend = ($value:ty, $operation_family:ty),
        operation = $operation:expr,
        cases = $cases:tt $(,)?) => {
        $crate::check_operation_transposition!(
            @run (@approx($epsilon)),
            backend = ($value, $operation_family),
            operation = $operation,
            cases = $cases,
        )
    };

    // This internal branch builds and executes each pullback, then dispatches the selected cotangent comparison.
    (
        @run $comparison:tt,
        backend = ($value:ty, $operation_family:ty),
        operation = $operation:expr,
        cases = [
            $(
                {
                    inputs = [$($input:tt),+ $(,)?],
                    output_cotangents = [$($output_cotangent:expr),+ $(,)?],
                    input_cotangents = [$($input_cotangent:expr),+ $(,)?]
                    $(, pullback = $pullback:expr)? $(,)?
                }
            ),+ $(,)?
        ] $(,)?) => {{
        let operation = $operation;
        $(
        {
            let inputs: Vec<(_, bool, Option<$value>)> =
                vec![$($crate::check_operation_transposition!(@input $value, $input)),+];
            let linear_indices = inputs
                .iter()
                .enumerate()
                .filter_map(|(index, (_, linear, _))| linear.then_some(index))
                .collect::<Vec<_>>();
            assert!(!linear_indices.is_empty(), "a supported transposition case needs at least one linear input");
            let mut builder = $crate::programs::builders::ProgramBuilder::<$value, $operation_family>::new();
            let input_ids = inputs
                .iter()
                .map(|(r#type, _, _)| builder.add_input(r#type.clone()))
                .collect::<Vec<_>>();
            let operation: $operation_family = ::core::convert::Into::into(operation.clone());
            let output_ids = builder.add_instruction(operation, Vec::new(), input_ids, None).unwrap().to_vec();
            let output_count = output_ids.len();
            let program = builder
                .build::<Vec<$value>, Vec<$value>>(
                    output_ids,
                    vec![$crate::parameters::Placeholder; inputs.len()],
                    vec![$crate::parameters::Placeholder; output_count],
                )
                .unwrap();
            let pullback = program.transpose_with_respect_to(linear_indices.as_slice()).unwrap();
            $(assert_eq!(pullback.to_string(), $pullback.trim_end());)?

            let output_cotangents: Vec<$value> =
                vec![$(::core::convert::Into::into($output_cotangent)),+];
            assert_eq!(output_cotangents.len(), output_count, "declared output cotangent count is incorrect");
            let expected: Vec<$value> = vec![$(::core::convert::Into::into($input_cotangent)),+];
            assert_eq!(expected.len(), linear_indices.len(), "declared input cotangent count is incorrect");
            let pullback_inputs = output_cotangents
                .into_iter()
                .chain(inputs.into_iter().filter_map(|(_, _, value)| value))
                .collect::<Vec<_>>();
            let actual = pullback.interpret(pullback_inputs).unwrap();
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected.iter()) {
                $crate::check_operation_transposition!(@assert $comparison, actual, expected);
            }
        }
        )+
    }};

    // This branch checks the standard unsupported-transposition diagnostic using the eager array reference backend.
    (
        @rejected,
        operation = $operation:expr,
        input_types = $input_types:tt $(,)?
    ) => {
        $crate::check_operation_transposition!(
            @rejected,
            backend = (
                $crate::Array,
                $crate::ArrayOperation<$crate::Array>
            ),
            operation = $operation,
            input_types = $input_types,
        )
    };

    // This branch checks unsupported transposition for a caller-selected backend family.
    (
        @rejected,
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
        let output_ids = builder.add_instruction(operation, Vec::new(), input_ids, None).unwrap().to_vec();
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

    // This internal branch represents one linear input by type without supplying a known primal value.
    (@input $value:ty, (@linear(type = $r#type:expr))) => {
        ($r#type, true, Option::<$value>::None)
    };

    // This internal branch retains one known primal input for pullback interpretation.
    (@input $value:ty, (@known, $input:expr)) => {{
        let input: $value = ::core::convert::Into::into($input);
        ($crate::programs::types::Typed::r#type(&input).into_owned(), false, Some(input))
    }};

    // This internal branch compares one interpreted cotangent exactly.
    (@assert (@exact), $actual:expr, $expected:expr) => {
        assert_eq!($actual, $expected)
    };

    // This internal branch compares one interpreted cotangent within the requested tolerance.
    (@assert (@approx($epsilon:expr)), $actual:expr, $expected:expr) => {
        ::approx::assert_abs_diff_eq!($actual, $expected, epsilon = $epsilon)
    };
}

/// Asserts that the reverse-mode gradient of a function at an input matches a central finite-difference estimate of its
/// derivative within an absolute tolerance. This is the standard oracle for testing operation gradient rules without
/// hand-deriving the expected derivative and without trusting the machinery under test (i.e., the gradient side runs
/// the function through [`DifferentiationBuilder::gradient`](crate::DifferentiationBuilder::gradient), while the
/// finite-difference side evaluates the function directly on concrete values at the perturbed points, never touching
/// the differentiation machinery that it is checking). That double instantiation is why this is a macro: the function
/// must be a closure literal (or a generic function), and the macro instantiates it once over
/// [`LinearizationTracer`](crate::LinearizationTracer) inputs and once over concrete [`Array`](crate::Array) inputs.
///
/// An `f64`-typed input estimates each ordinary partial derivative with `(f(x + h) - f(x - h)) / (2h)`, holding all
/// other elements fixed. Rank-zero arrays therefore cover scalar functions without requiring a separate scalar value
/// universe. A `c128`-typed input requires a ℂⁿ → ℝ function and estimates both real partials per element with central
/// differences along the real and imaginary axes, assembling `complex(∂f/∂re, -∂f/∂im)`, the conjugate steepest-ascent
/// gradient returned by the bilinear transposition pairing (e.g., `2z̄` for `f(z) = |z|²`). Other input data types
/// (including `c64`, whose `f32` precision cannot support a meaningful central difference) panic!
///
/// # Parameters
///
///   - `$function`: Closure literal (or generic function) to differentiate. The function may return its output value
///     either directly or wrapped in a [`Result`] whose error type converts into the differentiation machinery's error
///     types (which holds for the [`ProgramError`](crate::ProgramError) that the value capability traits return), so
///     fallible capability calls like `x.sin()` need no `.unwrap()`. Refer to [`MaybeFallible`](crate::MaybeFallible)
///     for the exact contract.
///   - `at = $input`: Expression convertible into the selected universe's value, at which the gradient is checked.
///   - `with = $capture`: Optional single non-differentiated runtime capture, convertible into an
///     [`Array`](crate::Array), that is passed as the function's second argument. The capture participates in the
///     concrete and differentiated evaluations but is excluded from the gradient parameter tree. Structured capture
///     trees are not accepted here and go through the builder API (i.e., `differentiate_at(...).with_captures(...)`)
///     directly.
///   - `step = $step`: Central finite-difference spacing `h`.
///   - `tolerance = $tolerance`: Absolute tolerance for the comparison. Pick one compatible with the `O($step²)`
///     truncation error of the central difference.
#[macro_export]
macro_rules! check_gradient {
    // This public branch evaluates an array function through reverse-mode differentiation and concrete finite
    // differences. Rank-zero arrays represent scalar inputs and outputs.
    ($function:expr, at = $input:expr, step = $step:expr, tolerance = $tolerance:expr $(,)?) => {{
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
            __F: Fn(
                $crate::differentiation::LinearizationTracer<
                    $crate::contexts::EagerContext<
                        $crate::Array,
                        $crate::ArrayOperation<$crate::Array>,
                    >,
                >,
            ) -> __Output,
            __Output,
        >(function: __F) -> __F {
            function
        }

        fn pin_eager<
            __F: Fn($crate::Array) -> __Output,
            __Output: $crate::MaybeFallible<$crate::Array, $crate::ProgramError>,
        >(
            function: __F,
        ) -> impl Fn($crate::Array) -> $crate::Array {
            move |input| {
                $crate::MaybeFallible::into_result(function(input)).unwrap_or_else(|error| panic!("{error}"))
            }
        }

        let input: $crate::Array = ::core::convert::Into::into($input);
        let step: f64 = $step;
        let tolerance: f64 = $tolerance;
        let gradient = $crate::differentiation::differentiate_at(input.clone())
            .gradient(pin_traced($function))
            .unwrap();

        $crate::check_gradient!(@assert(gradient, pin_eager($function), input, step, tolerance))
    }};

    // This public branch evaluates an array function with nondifferentiated runtime captures through reverse-mode
    // differentiation and concrete finite differences.
    (
        $function:expr,
        at = $input:expr,
        with = $capture:expr,
        step = $step:expr,
        tolerance = $tolerance:expr $(,)?
    ) => {{
        fn pin_traced<
            __F: Fn(
                $crate::differentiation::LinearizationTracer<
                    $crate::contexts::EagerContext<
                        $crate::Array,
                        $crate::ArrayOperation<$crate::Array>,
                    >,
                >,
                $crate::differentiation::LinearizationTracer<
                    $crate::contexts::EagerContext<
                        $crate::Array,
                        $crate::ArrayOperation<$crate::Array>,
                    >,
                >,
            ) -> __Output,
            __Output,
        >(function: __F) -> __F {
            function
        }

        fn pin_eager<
            __F: Fn($crate::Array, $crate::Array) -> __Output,
            __Output: $crate::MaybeFallible<$crate::Array, $crate::ProgramError>,
        >(
            function: __F,
        ) -> impl Fn($crate::Array, $crate::Array) -> $crate::Array {
            move |input, captures| {
                $crate::MaybeFallible::into_result(function(input, captures))
                    .unwrap_or_else(|error| panic!("{error}"))
            }
        }

        let input: $crate::Array = ::core::convert::Into::into($input);
        let captures: $crate::Array = ::core::convert::Into::into($capture);
        let step: f64 = $step;
        let tolerance: f64 = $tolerance;
        let gradient = $crate::differentiation::differentiate_at(input.clone())
            .with_captures(captures.clone())
            .gradient(pin_traced($function))
            .unwrap();

        $crate::check_gradient!(
            @assert(
                gradient,
                move |input| pin_eager($function)(input, captures.clone()),
                input,
                step,
                tolerance,
            )
        )
    }};

    // This internal branch checks a reverse-mode `$gradient` of the ℝⁿ → ℝ or ℂⁿ → ℝ function `$evaluate` at `$input`
    // (an array of any shape whose output is a rank-0 real `f64` array) against the central finite-difference estimates
    // of its partials, perturbing one input element at a time with all others held fixed.
    (@assert($gradient:expr, $evaluate:expr, $input:expr, $step:expr, $tolerance:expr $(,)?)) => {{
        let gradient = $gradient;
        let evaluate = $evaluate;
        let input = $input;
        let step = $step;
        let tolerance = $tolerance;

        // The function output is a rank-0 real array, so the central difference reads its single `f64` element.
        let central_difference = |plus: $crate::Array, minus: $crate::Array| {
            (evaluate(plus).to_f64s()[0] - evaluate(minus).to_f64s()[0]) / (2.0 * step)
        };

        let input_type = $crate::programs::types::Typed::r#type(&input).into_owned();
        let element_count = $crate::Array::materialized_element_count(&input_type).unwrap();
        match input_type.data_type() {
            $crate::arrays::DataType::F64 => {
                let perturbed = |index: usize, delta: f64| {
                    let mut values = input.to_f64s();
                    values[index] += delta;
                    $crate::Array::from_f64s(input_type.clone(), values)
                };
                let estimates = (0..element_count)
                    .map(|index| central_difference(perturbed(index, step), perturbed(index, -step)))
                    .collect::<Vec<_>>();
                let estimate = $crate::Array::from_f64s(input_type.clone(), estimates);
                ::approx::assert_abs_diff_eq!(gradient, estimate, epsilon = tolerance);
            }
            $crate::arrays::DataType::C128 => {
                // Per input element, the two central differences estimate the real partials that assemble the
                // conjugate steepest-ascent gradient `complex(∂f/∂re, -∂f/∂im)`.
                let part_type = input_type.clone().with_data_type($crate::arrays::DataType::F64);
                let real_values = $crate::operations::complex::Real::real(&input).unwrap().to_f64s();
                let imaginary_values = $crate::operations::complex::Imaginary::imaginary(&input).unwrap().to_f64s();
                let perturbed = |index: usize, real_delta: f64, imaginary_delta: f64| {
                    let mut real_values = real_values.clone();
                    let mut imaginary_values = imaginary_values.clone();
                    real_values[index] += real_delta;
                    imaginary_values[index] += imaginary_delta;
                    $crate::operations::complex::Complex::complex(
                        &$crate::Array::from_f64s(part_type.clone(), real_values),
                        &$crate::Array::from_f64s(part_type.clone(), imaginary_values),
                    )
                    .unwrap()
                };
                let mut real_estimates = Vec::with_capacity(element_count);
                let mut imaginary_estimates = Vec::with_capacity(element_count);
                for index in 0..element_count {
                    real_estimates.push(central_difference(
                        perturbed(index, step, 0.0),
                        perturbed(index, -step, 0.0),
                    ));
                    imaginary_estimates.push(-central_difference(
                        perturbed(index, 0.0, step),
                        perturbed(index, 0.0, -step),
                    ));
                }
                let estimate = $crate::operations::complex::Complex::complex(
                    &$crate::Array::from_f64s(part_type.clone(), real_estimates),
                    &$crate::Array::from_f64s(part_type, imaginary_estimates),
                )
                .unwrap();
                ::approx::assert_abs_diff_eq!(gradient, estimate, epsilon = tolerance);
            }
            other => panic!("finite-difference gradient checking requires an f64 or c128 input but got {other}"),
        }
    }};
}

pub use crate::{
    check_builders, check_count, check_gradient, check_operation_batching, check_operation_differentiation,
    check_operation_partial_evaluation, check_operation_transposition, check_operation_type_inference, check_sharding,
    check_types, define_arithmetic_dimension_capability, define_arithmetic_dimension_operation,
    define_elementwise_capability, define_elementwise_operation, define_tracer_operator,
    dispatch_on_array_element_type, impl_differentiable_elementwise_operation, impl_differentiable_operation,
    impl_non_differentiable_operation, impl_non_transposable_operation, impl_nullary_batchable_operation,
    impl_nullary_transposable_operation, impl_reference_free_dischargeable_operation,
};

#[cfg(test)]
mod tests {
    use std::fmt::{Debug, Display, Formatter};
    use std::marker::PhantomData;
    use std::rc::Rc;

    use indoc::indoc;
    use num_complex::Complex;

    use crate::arrays::{
        Array, ArrayBatch, ArrayBatching, ArrayIrOperation, ArrayIrValue, ArrayOperation, ArrayType, DataType, Device,
        DeviceMesh, Dimension, DimensionBounds, DimensionError, DimensionType, DimensionValue, DimensionVariable,
        LogicalMesh, MeshAxis, MeshAxisType, Shape, Sharding, ShardingDimension, ShardingError,
    };
    use crate::batching::{BatchableOperation, BatchingContext, BatchingError, BatchingTracer};
    use crate::contexts::{Context, Domain, EagerContext, StagingContext};
    use crate::differentiation::{
        DifferentiableOperation, DifferentiationContext, DifferentiationDual, DifferentiationError,
        DifferentiationTracer, TransposableOperation, TranspositionContext,
    };
    use crate::interpretation::{InterpretableOperation, InterpretationDriver};
    use crate::operations::{
        Abs, AbsOperation, Add, AddOperation, ArithmeticDimensionOperation, BroadcastOperation, DivOperation,
        ElementwiseOperation, ExpOperation, MulOperation, Neg, NegOperation, Reduce, ReductionKind, SinOperation, Sub,
        SubOperation, TransposeOperation, ZeroOperation,
    };
    use crate::parameters::Parameter;
    use crate::partial::{
        PartialEvaluationContext, PartialEvaluationValue, PartialTracer, PartialValue, PartiallyEvaluatableOperation,
    };
    use crate::programs::{
        EmptyRegionDriver, MaybeZero, Operation, OperationProvider, ProgramError, ReferenceDischargeContext,
        ReferenceDischargePolicy, ReferenceDischargeValue, ReferenceDischargeableOperation, ReferenceType, Type,
        TypeError, TypeIdentityRenaming, Typed, ValueProjection,
    };
    use crate::tracing::{Tracer, TracingContext};

    const TEST_UNARY_OPERATION_NAME: &str = "test_unary";
    const TEST_BINARY_OPERATION_NAME: &str = "test_binary";

    define_elementwise_operation!(
        @unary
        /// Unary operation used to test [`define_elementwise_operation!`].
        TestUnaryOperation, TEST_UNARY_OPERATION_NAME,
        Neg, neg,
        check_data_types = [@numeric @real],
        check_array_types = [@no_unreduced],
    );

    define_elementwise_operation!(
        @binary
        /// Binary operation used to test [`define_elementwise_operation!`].
        TestBinaryOperation, TEST_BINARY_OPERATION_NAME,
        Add, add,
        check_data_types = [@numeric, @real],
        check_array_types = [@same_unreduced_axes, @same_reduced_axes],
    );

    const TEST_MAGNITUDE_OPERATION_NAME: &str = "test_magnitude";
    const TEST_STRICT_ADD_OPERATION_NAME: &str = "test_strict_add";

    define_elementwise_operation!(
        @unary
        /// Unary operation used to test custom data-type inference.
        TestMagnitudeOperation, TEST_MAGNITUDE_OPERATION_NAME,
        Abs, abs,
        infer_data_types = |input_types: &[DataType]| {
            Ok(vec![match input_types[0] {
                DataType::C64 => DataType::F32,
                DataType::C128 => DataType::F64,
                input_type => input_type,
            }])
        },
    );

    define_elementwise_operation!(
        @binary
        /// Binary operation used to test custom array-type inference.
        TestStrictAddOperation, TEST_STRICT_ADD_OPERATION_NAME,
        Add, add,
        infer_array_types = |input_types: &[ArrayType]| {
            if input_types[0] != input_types[1] {
                return Err(TypeError::invalid("test strict-add inputs must have identical array types".to_string()));
            }
            Ok(vec![input_types[0].clone()])
        },
        check_data_types = [@numeric],
    );

    define_elementwise_capability!(
        @unary
        /// Unary capability used to test [`define_elementwise_capability!`].
        TestUnary,
        /// Applies the test unary operation.
        test_unary,
        TestUnaryOperation,
    );

    define_elementwise_capability!(
        @binary
        /// Binary capability used to test [`define_elementwise_capability!`].
        TestBinary,
        /// Applies the test binary operation to this value and `other`.
        test_binary(other),
        TestBinaryOperation,
    );

    const TEST_ARITHMETIC_DIMENSION_OPERATION_NAME: &str = "test_arithmetic_dimension";

    define_arithmetic_dimension_operation!(
        /// Dimension-arithmetic operation used to test [`define_arithmetic_dimension_operation!`].
        TestArithmeticDimensionOperation,
        TEST_ARITHMETIC_DIMENSION_OPERATION_NAME,
        TestArithmeticDimension,
        test_arithmetic_dimension,
        result_name = |left: &DimensionType, right: &DimensionType| {
            format!("{} + {}", left.variable(), right.variable())
        },
        infer_bounds = |left: &DimensionType, right: &DimensionType| {
            let lower = left
                .bounds()
                .lower()
                .checked_add(right.bounds().lower())
                .ok_or_else(|| DimensionError::ArithmeticOverflow {
                    message: "test dimension bounds overflow".to_string(),
                })?;
            let upper = match (left.bounds().upper(), right.bounds().upper()) {
                (Some(left), Some(right)) => left.checked_add(right).and_then(|sum| sum.checked_sub(1)),
                _ => None,
            };
            Ok((DimensionBounds::new(lower, upper)?, false))
        },
    );

    define_arithmetic_dimension_capability!(
        /// Dimension-arithmetic capability used to test [`define_arithmetic_dimension_capability!`].
        TestArithmeticDimension,
        /// Adds this test dimension to `right`.
        test_arithmetic_dimension(right),
        TestArithmeticDimensionOperation,
    );

    const TEST_DIFFERENTIABLE_OPERATION_NAME: &str = "test_differentiable";

    define_elementwise_operation!(
        @binary
        /// Binary operation used to test [`impl_differentiable_operation!`].
        TestDifferentiableOperation, TEST_DIFFERENTIABLE_OPERATION_NAME,
        Add, add,
        check_data_types = [@numeric],
        check_array_types = [@same_unreduced_axes, @same_reduced_axes],
    );

    impl_differentiable_operation! {
        TestDifferentiableOperation<ArrayType>,
        jvp<C>
        where
            C::Type: crate::DifferentiableType,
        {
            |_operation, _context, _driver, inputs| {
                check_count!("input", inputs, 2, ProgramError);
                Ok(vec![DifferentiationDual::new(inputs[0].primal().clone(), inputs[0].tangent().clone())?])
            }
        },
        transpose<V, O> {
            |_operation, _context, _driver, inputs, outputs| {
                check_count!("input", inputs, 2, ProgramError);
                check_count!("output", outputs, 1, ProgramError);
                Ok(vec![outputs[0].clone(), outputs[0].clone()])
            }
        },
    }

    const TEST_REVERSED_SUB_OPERATION_NAME: &str = "test_reversed_sub";
    const TEST_NEGATED_ADD_OPERATION_NAME: &str = "test_negated_add";

    define_elementwise_operation!(
        @binary
        /// Binary operation with a `[@negative, @positive]` differentiation rule used to test
        /// [`impl_differentiable_elementwise_operation!`]. Interpretation reuses [`Sub`] as a stand-in because only
        /// the generated differentiation and transposition rules are under test.
        TestReversedSubOperation, TEST_REVERSED_SUB_OPERATION_NAME,
        Sub, sub,
        check_data_types = [@numeric],
        check_array_types = [@same_unreduced_axes, @same_reduced_axes],
    );

    define_elementwise_operation!(
        @binary
        /// Binary operation with a `[@negative, @negative]` differentiation rule used to test
        /// [`impl_differentiable_elementwise_operation!`]. Interpretation reuses [`Add`] as a stand-in because only
        /// the generated differentiation and transposition rules are under test.
        TestNegatedAddOperation, TEST_NEGATED_ADD_OPERATION_NAME,
        Add, add,
        check_data_types = [@numeric],
        check_array_types = [@same_unreduced_axes, @same_reduced_axes],
    );

    impl_differentiable_elementwise_operation! {
        @linear
        TestReversedSubOperation,
        rule = [@negative, @positive]
    }

    impl_differentiable_elementwise_operation! {
        @linear
        TestNegatedAddOperation,
        rule = [@negative, @negative]
    }

    impl_reference_free_dischargeable_operation!(TestUnaryOperation<ArrayType>);
    impl_non_differentiable_operation!(TestUnaryOperation<ArrayType>);
    impl_non_transposable_operation!(TestUnaryOperation<ArrayType>);
    impl_non_differentiable_operation!(TestBinaryOperation<ArrayType>);

    impl From<ZeroOperation<ArrayType>> for TestUnaryOperation<ArrayType> {
        fn from(_operation: ZeroOperation<ArrayType>) -> Self {
            Self::new()
        }
    }

    impl From<ZeroOperation<ArrayType>> for TestBinaryOperation<ArrayType> {
        fn from(_operation: ZeroOperation<ArrayType>) -> Self {
            Self::new()
        }
    }

    impl From<TransposeOperation> for TestUnaryOperation<ArrayType> {
        fn from(_operation: TransposeOperation) -> Self {
            Self::new()
        }
    }

    impl From<BroadcastOperation> for TestUnaryOperation<ArrayType> {
        fn from(_operation: BroadcastOperation) -> Self {
            Self::new()
        }
    }

    impl From<NegOperation<DataType>> for TestUnaryOperation<DataType> {
        fn from(_operation: NegOperation<DataType>) -> Self {
            Self::new()
        }
    }

    impl From<NegOperation<ArrayType>> for TestUnaryOperation<ArrayType> {
        fn from(_operation: NegOperation<ArrayType>) -> Self {
            Self::new()
        }
    }

    impl From<TransposeOperation> for TestBinaryOperation<ArrayType> {
        fn from(_operation: TransposeOperation) -> Self {
            Self::new()
        }
    }

    impl From<BroadcastOperation> for TestBinaryOperation<ArrayType> {
        fn from(_operation: BroadcastOperation) -> Self {
            Self::new()
        }
    }

    impl From<AddOperation<DataType>> for TestBinaryOperation<DataType> {
        fn from(_operation: AddOperation<DataType>) -> Self {
            Self::new()
        }
    }

    impl From<AddOperation<ArrayType>> for TestBinaryOperation<ArrayType> {
        fn from(_operation: AddOperation<ArrayType>) -> Self {
            Self::new()
        }
    }

    impl From<TestUnaryOperation<ArrayType>> for ArrayOperation<Array> {
        fn from(_operation: TestUnaryOperation<ArrayType>) -> Self {
            Self::Neg(NegOperation::new())
        }
    }

    impl From<TestBinaryOperation<ArrayType>> for ArrayOperation<Array> {
        fn from(_operation: TestBinaryOperation<ArrayType>) -> Self {
            Self::Add(AddOperation::new())
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

    /// Binary operator used to test type-directed operation selection in [`define_tracer_operator!`].
    trait TestProvidedBinaryOperator {
        /// Result of applying this operator.
        type Output;

        /// Applies the operator selected for this tracer's program type.
        fn apply_provided_binary(self, right: Self) -> Self::Output;
    }

    impl OperationProvider<DimensionType> for TestBinaryOperation<DimensionType> {
        type Operation = TestArithmeticDimensionOperation;

        fn provide(input_types: &[&DimensionType]) -> Result<Self::Operation, ProgramError> {
            check_count!("input", input_types, 2, ProgramError);
            Ok(TestArithmeticDimensionOperation::new(input_types[0], input_types[1])?)
        }
    }

    define_elementwise_capability!(
        @binary
        /// Fallible capability used by the ordinary binary operator macro fixture.
        TestBinaryCapability,
        /// Applies the fixture's ordinary binary operation.
        apply_binary_fallible(right),
        TestBinaryOperation,
    );

    define_elementwise_capability!(
        @binary
        /// Fallible capability used by the type-directed binary operator macro fixture.
        TestProvidedBinaryCapability,
        /// Applies the fixture's type-directed binary operation.
        apply_provided_binary_fallible(right),
        TestBinaryOperation,
    );

    define_tracer_operator!(
        @unary TestUnaryOperator,
        apply_unary,
        TestUnaryOperation,
        "test unary operation failed",
    );

    define_tracer_operator!(
        @binary TestBinaryOperator,
        apply_binary,
        capability = TestBinaryCapability,
        method = apply_binary_fallible,
    );

    define_tracer_operator!(
        @binary TestProvidedBinaryOperator,
        apply_provided_binary,
        capability = TestProvidedBinaryCapability,
        method = apply_provided_binary_fallible,
    );

    /// Type-directed nullary operation used to execute generated transposition and batching rules.
    #[derive(Clone, Debug)]
    struct TestNullaryOperation<T: Type>(PhantomData<fn() -> T>);

    impl<T: Type> TestNullaryOperation<T> {
        /// Constructs a fixture operation for the `T` type universe.
        fn new() -> Self {
            Self(PhantomData)
        }
    }

    impl<T: Type> Display for TestNullaryOperation<T> {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("test_nullary")
        }
    }

    impl Operation for TestNullaryOperation<DataType> {
        type Type = DataType;

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

    impl Operation for TestNullaryOperation<ArrayType> {
        type Type = ArrayType;

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

    impl InterpretableOperation<EagerContext<Array, TestNullaryOperation<ArrayType>>> for TestNullaryOperation<ArrayType> {
        fn interpret<D: InterpretationDriver<EagerContext<Array, TestNullaryOperation<ArrayType>>>>(
            &self,
            _context: &EagerContext<Array, TestNullaryOperation<ArrayType>>,
            _driver: &D,
            inputs: &[Array],
        ) -> Result<Vec<Array>, ProgramError> {
            check_count!("input", inputs, 0, ProgramError);
            Ok(vec![Array::scalar(3.0), Array::scalar(4.0)])
        }
    }

    impl_nullary_transposable_operation!(<T> TestNullaryOperation<T> where T: Type);
    impl_nullary_batchable_operation!(@replicated <T> TestNullaryOperation<T> where T: Type);

    /// Generic nullary operation used to instantiate generic macro forms.
    struct TestGenericNullaryOperation<T>(PhantomData<fn() -> T>);

    impl<T> Clone for TestGenericNullaryOperation<T> {
        fn clone(&self) -> Self {
            Self(PhantomData)
        }
    }

    impl<T> Debug for TestGenericNullaryOperation<T> {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("TestGenericNullaryOperation")
        }
    }

    impl<T> Display for TestGenericNullaryOperation<T> {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("test_generic_nullary")
        }
    }

    impl<T: Type> Operation for TestGenericNullaryOperation<T> {
        type Type = T;

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

    impl<C: Domain> InterpretableOperation<C> for TestGenericNullaryOperation<C::Type> {
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

    impl_non_differentiable_operation!(<C> TestGenericNullaryOperation<C>);
    impl_nullary_transposable_operation!(<T> TestGenericNullaryOperation<T>);
    impl_nullary_batchable_operation!(@replicated <D> TestGenericNullaryOperation<D>);

    /// Generic nullary operation used to instantiate generic-plus-`where` macro forms.
    struct TestBoundedNullaryOperation<T>(PhantomData<fn() -> T>);

    impl<T> Clone for TestBoundedNullaryOperation<T> {
        fn clone(&self) -> Self {
            Self(PhantomData)
        }
    }

    impl<T> Debug for TestBoundedNullaryOperation<T> {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("TestBoundedNullaryOperation")
        }
    }

    impl<T> Display for TestBoundedNullaryOperation<T> {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("test_bounded_nullary")
        }
    }

    impl<T: Type> Operation for TestBoundedNullaryOperation<T> {
        type Type = T;

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

    impl<C: Domain> InterpretableOperation<C> for TestBoundedNullaryOperation<C::Type> {
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

    impl_non_differentiable_operation!(<V> TestBoundedNullaryOperation<V> where V: Clone);
    impl_nullary_transposable_operation!(<O> TestBoundedNullaryOperation<O> where O: Clone);
    impl_nullary_batchable_operation!(@replicated <C> TestBoundedNullaryOperation<C> where C: Clone);

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
        assert_eq!(check_operand(&[], 1), Err(TypeError::invalid("expected 1 operand but got 0".to_string())),);
        assert_eq!(check_operand(&[0], 2), Err(TypeError::invalid("expected 2 operands but got 1".to_string())),);
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
            Err(TypeError::invalid("test type signature mismatch: expected [f32, f64] but got [f32, i64]".to_string())),
        );
    }

    #[test]
    fn test_check_types_data_types() {
        let check_numeric = |types: &[DataType]| -> Result<(), TypeError> {
            check_types!(@numeric, "test", types);
            Ok(())
        };
        let check_real = |types: &[DataType]| -> Result<(), TypeError> {
            check_types!(@real, "test", types);
            Ok(())
        };
        let check_float = |types: &[DataType]| -> Result<(), TypeError> {
            check_types!(@float, "test", types);
            Ok(())
        };
        let check_numeric_then_real = |types: &[DataType]| -> Result<(), TypeError> {
            check_types!(@numeric @real, "test", types);
            Ok(())
        };
        let check_real_then_numeric = |types: &[DataType]| -> Result<(), TypeError> {
            check_types!(@real @numeric, "test", types);
            Ok(())
        };
        let check_float_then_real = |types: &[DataType]| -> Result<(), TypeError> {
            check_types!(@float @real, "test", types);
            Ok(())
        };
        let check_real_then_float = |types: &[DataType]| -> Result<(), TypeError> {
            check_types!(@real @float, "test", types);
            Ok(())
        };
        assert_eq!(check_numeric(&[DataType::I32, DataType::F64, DataType::C128]), Ok(()));
        for r#type in [DataType::Boolean, DataType::Token, DataType::Zero] {
            assert_eq!(
                check_numeric(&[DataType::F32, r#type]),
                Err(TypeError::invalid(format!("`test` does not support input data type {type}", type = r#type))),
            );
        }
        assert_eq!(check_real(&[DataType::I64, DataType::F32]), Ok(()));
        assert_eq!(
            check_real(&[DataType::C64]),
            Err(TypeError::invalid("`test` does not support input data type c64".to_string())),
        );
        assert_eq!(check_float(&[DataType::BF16, DataType::F64, DataType::C64]), Ok(()));
        assert_eq!(
            check_float(&[DataType::I64]),
            Err(TypeError::invalid("`test` does not support input data type i64".to_string())),
        );
        assert_eq!(check_numeric_then_real(&[DataType::I32, DataType::F64]), Ok(()));
        assert_eq!(check_real_then_numeric(&[DataType::I32, DataType::F64]), Ok(()));
        assert_eq!(
            check_numeric_then_real(&[DataType::C128]),
            Err(TypeError::invalid("`test` does not support input data type c128".to_string())),
        );
        assert_eq!(
            check_numeric_then_real(&[DataType::Boolean]),
            Err(TypeError::invalid("`test` does not support input data type bool".to_string())),
        );
        assert_eq!(check_float_then_real(&[DataType::BF16, DataType::F64]), Ok(()));
        assert_eq!(check_real_then_float(&[DataType::BF16, DataType::F64]), Ok(()));
        assert_eq!(
            check_float_then_real(&[DataType::C64]),
            Err(TypeError::invalid("`test` does not support input data type c64".to_string())),
        );
        assert_eq!(
            check_float_then_real(&[DataType::I64]),
            Err(TypeError::invalid("`test` does not support input data type i64".to_string())),
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
            Err(TypeError::invalid("`test` does not support unreduced operands".to_string())),
        );
        assert_eq!(check_unreduced_axes(&[unreduced_x.clone(), unreduced_x.clone()]), Ok(()));
        assert_eq!(
            check_unreduced_axes(&[unreduced_x, unreduced_y]),
            Err(TypeError::invalid("`test` operands must be unreduced over the same axes".to_string())),
        );
        assert_eq!(
            check_unreduced_axes(std::slice::from_ref(&plain)),
            Err(TypeError::invalid("expected 2 inputs but got 1".to_string())),
        );
        assert_eq!(check_reduced_axes(&[reduced_x.clone(), reduced_x.clone()]), Ok(()));
        assert_eq!(
            check_reduced_axes(&[reduced_x, reduced_y]),
            Err(TypeError::invalid("`test` operands must be reduced over the same axes".to_string())),
        );
        assert_eq!(
            check_reduced_axes(std::slice::from_ref(&plain)),
            Err(TypeError::invalid("expected 2 inputs but got 1".to_string())),
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

    // TODO(eaplatanios): Generally about this `tests` module, the tests are not defined in the same order as the
    //  corresponding macros. Re-order them accordingly.
    // TODO(eaplatanios): Review this.
    #[test]
    fn test_define_arithmetic_dimension_operation() {
        let left_type = DimensionType::new(DimensionVariable::new("left", DimensionBounds::new(1, Some(4)).unwrap()));
        let right_type = DimensionType::new(DimensionVariable::new("right", DimensionBounds::new(2, Some(6)).unwrap()));
        let operation = TestArithmeticDimensionOperation::new(&left_type, &right_type).unwrap();

        // The generated operation owns stable operand metadata and the result name and bounds needed to infer a fresh
        // program-atom identity.
        assert_eq!(format!("{operation}"), TEST_ARITHMETIC_DIMENSION_OPERATION_NAME);
        assert_eq!(operation.name(), TEST_ARITHMETIC_DIMENSION_OPERATION_NAME,);
        assert_eq!(operation.left_type(), &left_type);
        assert_eq!(operation.right_type(), &right_type);
        assert_eq!(operation.result_bounds(), DimensionBounds::new(3, Some(9)).unwrap());
        let result = Operation::infer_output_types(&operation, &[left_type.clone(), right_type.clone()], &[]).unwrap();
        assert_ne!(result[0].variable(), left_type.variable());
        assert_ne!(result[0].variable(), right_type.variable());
        assert_eq!(result[0].bounds(), operation.result_bounds());
        assert_eq!(
            Operation::infer_output_types(&operation, std::slice::from_ref(&left_type), &[]),
            Err(TypeError::invalid("expected 2 inputs but got 1".to_string())),
        );
        fn assert_arithmetic_dimension_operation<O: ArithmeticDimensionOperation>() {}
        assert_arithmetic_dimension_operation::<TestArithmeticDimensionOperation>();

        // The generated identity-renaming implementation rewrites both declared operand identities while preserving
        // the result metadata from which the output atom's identity will be inferred.
        let renamed_left = DimensionType::new(DimensionVariable::new("renamed_left", left_type.bounds()));
        let renamed_right = DimensionType::new(DimensionVariable::new("renamed_right", right_type.bounds()));
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(left_type.variable().clone(), renamed_left.variable().clone()).unwrap();
        renaming.insert(right_type.variable().clone(), renamed_right.variable().clone()).unwrap();
        let renamed = operation.rename_type_identities(&renaming).unwrap();
        assert_eq!(renamed.left_type(), &renamed_left);
        assert_eq!(renamed.right_type(), &renamed_right);
        assert_eq!(renamed.result_name(), operation.result_name());
        assert_eq!(renamed.result_bounds(), operation.result_bounds());

        // The macro supplies the ordinary partial-evaluation marker implementation for any compatible context.
        fn assert_partially_evaluatable<
            C: Context<Type = DimensionType, Operation = TestArithmeticDimensionOperation>,
        >()
        where
            TestArithmeticDimensionOperation: PartiallyEvaluatableOperation<C>,
        {
        }
        assert_partially_evaluatable::<TracingContext<DimensionValue, TestArithmeticDimensionOperation>>();
    }

    // TODO(eaplatanios): Review this.
    #[test]
    fn test_define_arithmetic_dimension_capability() {
        let left_type = DimensionType::new(DimensionVariable::new("left", DimensionBounds::new(1, Some(4)).unwrap()));
        let right_type = DimensionType::new(DimensionVariable::new("right", DimensionBounds::new(2, Some(6)).unwrap()));
        let context = TracingContext::<DimensionValue, TestArithmeticDimensionOperation>::new();
        let left = context.input(left_type);
        let right = context.input(right_type);
        let output = left.test_arithmetic_dimension(&right).unwrap();

        assert_eq!(output.r#type().bounds(), DimensionBounds::new(3, Some(9)).unwrap());
        let builder = output.builder().borrow();
        assert_eq!(builder.instructions().len(), 1);
        assert_eq!(builder.instructions()[0].operation().name(), TEST_ARITHMETIC_DIMENSION_OPERATION_NAME,);
        assert_eq!(builder.instructions()[0].inputs(), &[left.atom_id().unwrap(), right.atom_id().unwrap()]);
        assert_eq!(builder.instructions()[0].outputs(), &[output.atom_id().unwrap()]);
    }

    #[test]
    fn test_check_operation_type_inference() {
        #[derive(Clone, Debug)]
        struct TestMultiOutputOperation<T: Type>(PhantomData<fn() -> T>);

        impl<T: Type> TestMultiOutputOperation<T> {
            const fn new() -> Self {
                Self(PhantomData)
            }
        }

        impl<T: Type> Display for TestMultiOutputOperation<T> {
            fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("test_multi_output")
            }
        }

        impl<T: Type> Operation for TestMultiOutputOperation<T> {
            type Type = T;

            fn name(&self) -> &'static str {
                "test_multi_output"
            }

            fn infer_output_types(
                &self,
                input_types: &[T],
                _region_interfaces: &[crate::RegionInterface<T>],
            ) -> Result<Vec<T>, TypeError> {
                check_count!("input", input_types, 1, TypeError);
                Ok(vec![input_types[0].clone(), input_types[0].clone()])
            }
        }

        impl ElementwiseOperation for TestMultiOutputOperation<ArrayType> {
            fn input_count(&self) -> usize {
                1
            }

            fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
                Operation::infer_output_types(self, input_types, &[])
            }
        }

        check_operation_type_inference!(
            @elementwise @unary,
            operation = TestMultiOutputOperation,
            cases = [{
                input_data_types = [DataType::F64],
                output_data_types = [DataType::F64, DataType::F64],
            }],
        );

        check_operation_type_inference!(
            @elementwise @unary,
            operation = AbsOperation,
            cases = [
                {
                    input_data_types = [DataType::C64],
                    output_data_types = [DataType::F32],
                },
                {
                    input_data_types = [DataType::Boolean],
                    error = "cannot compute the absolute value of a value of data type bool",
                },
            ],
        );

        check_operation_type_inference!(
            @elementwise @binary,
            operation = AddOperation,
            cases = [
                {
                    input_data_types = [DataType::F32, DataType::F64],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::Boolean, DataType::Boolean],
                    error = "`add` does not support input data type bool",
                },
            ],
        );

        check_operation_type_inference!(
            operation = AddOperation::<DataType>::new(),
            cases = [
                {
                    input_types = [DataType::F32, DataType::F64],
                    output_types = [DataType::F64],
                },
                {
                    type = DataType,
                    input_types = [],
                    error = "expected 2 inputs but got 0",
                },
                {
                    input_types = [DataType::Boolean, DataType::Boolean],
                    error = "`add` does not support input data type bool",
                },
            ],
        );

        check_operation_type_inference!(
            operation = AddOperation::<ArrayType>::new(),
            cases = [{
                input_types = [ArrayType::scalar(DataType::F32), ArrayType::scalar(DataType::F64)],
                output_types = [ArrayType::scalar(DataType::F64)],
            }],
        );

        check_operation_type_inference!(
            @reject @unreduced,
            operation = SinOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );

        check_operation_type_inference!(
            @reject @mismatched_reduced,
            operation = AddOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_check_operation_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = NegOperation::new(),
            inputs = [Array::scalar(2.0)],
            expected = Array::scalar(-2.0),
        );
        check_operation_partial_evaluation!(
            operation = AddOperation::new(),
            cases = [
                {
                    inputs = [
                        (@known, Array::scalar(2.0)),
                        (@known, Array::scalar(3.5)),
                    ],
                    outputs = [
                        (@known, Array::scalar(5.5)),
                    ],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(type = ArrayType::scalar(DataType::F64), replay = Array::scalar(2.0))),
                        (@known, Array::scalar(3.5)),
                    ],
                    outputs = [
                        (@residual, Array::scalar(5.5)),
                    ],
                    residual_instructions = 1,
                },
                {
                    inputs = [
                        (@known, Array::scalar(2.0)),
                        (@unknown(type = ArrayType::scalar(DataType::F64), replay = Array::scalar(3.5))),
                    ],
                    outputs = [
                        (@residual, Array::scalar(5.5)),
                    ],
                    residual_instructions = 1,
                },
                {
                    inputs = [
                        (@unknown(type = ArrayType::scalar(DataType::F64), replay = Array::scalar(2.0))),
                        (@unknown(type = ArrayType::scalar(DataType::F64), replay = Array::scalar(3.5))),
                    ],
                    outputs = [
                        (@residual, Array::scalar(5.5)),
                    ],
                    residual_instructions = 1,
                },
            ],
        );
    }

    #[test]
    fn test_check_operation_batching() {
        #[derive(Clone)]
        struct TestPairOperation;

        impl Operation for TestPairOperation {
            type Type = ArrayType;

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

        check_operation_batching!(
            @exact,
            operation = ZeroOperation::new(ArrayType::scalar(DataType::F64)),
            axis_size = 2,
            cases = [{
                inputs = [],
                outputs = [(@replicated, Array::scalar(0.0))],
            }],
        );

        check_operation_batching!(
            @exact,
            context = EagerContext::<Array>::new(),
            driver = &EmptyRegionDriver,
            operation = TestPairOperation,
            axis_size = 2,
            axis_sharding = ShardingDimension::Replicated,
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

        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = SubOperation::new(),
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

        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = SinOperation::new(),
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

    #[test]
    fn test_check_operation_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            backend = (Array, ArrayOperation<Array>),
            operation = MulOperation::new(),
            cases = [
                {
                    primals = [Array::scalar(2.0), Array::scalar(5.0)],
                    tangents = [Array::scalar(3.0), Array::scalar(-1.0)],
                    primal_outputs = [Array::scalar(10.0)],
                    tangent_outputs = [Array::scalar(13.0)],
                    jvp = indoc! {"
                        lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                        let %4:f64[] = mul %0 %1
                            %5:f64[] = mul %1 %2
                            %6:f64[] = mul %0 %3
                            %7:f64[] = add %5 %6
                        in (%4, %7)
                    "},
                },
                {
                    primals = [Array::scalar(2.0), Array::vector(vec![1.0, 3.0])],
                    tangents = [Array::scalar(0.5), Array::vector(vec![2.0, -1.0])],
                    primal_outputs = [Array::vector(vec![2.0, 6.0])],
                    tangent_outputs = [Array::vector(vec![4.5, -0.5])],
                },
            ],
        );
    }

    #[test]
    fn test_check_operation_transposition() {
        check_operation_transposition!(
            @exact,
            backend = (Array, ArrayOperation<Array>),
            operation = MulOperation::new(),
            cases = [{
                inputs = [
                    (@known, Array::scalar(4.0)),
                    (@linear(type = ArrayType::scalar(DataType::F64))),
                ],
                output_cotangents = [Array::scalar(3.0)],
                input_cotangents = [Array::scalar(12.0)],
                pullback = indoc! {"
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = mul %1 %0
                    in (%2)
                "},
            }],
        );
        check_operation_transposition!(
            @approx(epsilon = 1e-9),
            operation = AddOperation::new(),
            cases = [{
                inputs = [
                    (@linear(type = ArrayType::scalar(DataType::F64))),
                    (@linear(type = ArrayType::scalar(DataType::F64))),
                ],
                output_cotangents = [Array::scalar(3.0)],
                input_cotangents = [Array::scalar(3.0), Array::scalar(3.0)],
            }],
        );
        check_operation_transposition!(
            @rejected,
            operation = SinOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_define_elementwise_operation_unary() {
        let data_operation = TestUnaryOperation::<DataType>::new();
        let array_operation = TestUnaryOperation::<ArrayType>::new();
        assert_eq!(size_of::<TestUnaryOperation<DataType>>(), 0);
        assert_eq!(size_of::<TestUnaryOperation<ArrayType>>(), 0);
        assert_eq!(TestUnaryOperation::<DataType>::default().to_string(), TEST_UNARY_OPERATION_NAME);
        assert_eq!(format!("{data_operation:?}"), "TestUnaryOperation");
        assert_eq!(format!("{data_operation}"), TEST_UNARY_OPERATION_NAME);
        assert_eq!(data_operation.name(), TEST_UNARY_OPERATION_NAME);
        assert_eq!(ElementwiseOperation::input_count(&array_operation), 1);
        assert_eq!(data_operation.infer_output_types(&[DataType::F32], &[]), Ok(vec![DataType::F32]));
        assert_eq!(
            data_operation.infer_output_types(&[], &[]),
            Err(TypeError::invalid("expected 1 input but got 0".to_string())),
        );
        assert_eq!(
            data_operation.infer_output_types(&[DataType::Boolean], &[]),
            Err(TypeError::invalid("`test_unary` does not support input data type bool".to_string())),
        );
        assert_eq!(data_operation.infer_output_types(&[DataType::I64], &[]), Ok(vec![DataType::I64]),);
        assert_eq!(
            data_operation.infer_output_types(&[DataType::C64], &[]),
            Err(TypeError::invalid("`test_unary` does not support input data type c64".to_string())),
        );
        assert_eq!(
            Operation::infer_output_types(&array_operation, &[ArrayType::scalar(DataType::F32)], &[]),
            Ok(vec![ArrayType::scalar(DataType::F32)]),
        );
        let matrix_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        assert_eq!(
            Operation::infer_output_types(&array_operation, std::slice::from_ref(&matrix_type), &[]),
            Ok(vec![matrix_type]),
        );
        assert_eq!(
            Operation::infer_output_types(&array_operation, &[ArrayType::scalar(DataType::C64)], &[]),
            Err(TypeError::invalid("`test_unary` does not support input data type c64".to_string())),
        );
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let unreduced_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::new(mesh, vec![]).unwrap().with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        assert_eq!(
            Operation::infer_output_types(&array_operation, &[unreduced_type], &[]),
            Err(TypeError::invalid("`test_unary` does not support unreduced operands".to_string())),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array, TestUnaryOperation<ArrayType>>>::interpret(
                &array_operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(2.0f32)],
            ),
            Ok(vec![Array::scalar(-2.0f32)]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array, TestUnaryOperation<ArrayType>>>::interpret(
                &array_operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
        let context = PartialEvaluationContext::new(EagerContext::<Array, TestUnaryOperation<ArrayType>>::new());
        let outputs = array_operation
            .partially_evaluate(&context, &EmptyRegionDriver, &[PartialEvaluationValue::known(Array::scalar(2.0f32))])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].as_known(), Some(&Array::scalar(-2.0f32)));
    }

    #[test]
    fn test_define_elementwise_operation_binary() {
        let data_operation = TestBinaryOperation::<DataType>::new();
        let array_operation = TestBinaryOperation::<ArrayType>::new();
        assert_eq!(size_of::<TestBinaryOperation<DataType>>(), 0);
        assert_eq!(size_of::<TestBinaryOperation<ArrayType>>(), 0);
        assert_eq!(TestBinaryOperation::<DataType>::default().to_string(), TEST_BINARY_OPERATION_NAME);
        assert_eq!(format!("{data_operation:?}"), "TestBinaryOperation");
        assert_eq!(format!("{data_operation}"), TEST_BINARY_OPERATION_NAME);
        assert_eq!(data_operation.name(), TEST_BINARY_OPERATION_NAME);
        assert_eq!(ElementwiseOperation::input_count(&array_operation), 2);
        assert_eq!(data_operation.infer_output_types(&[DataType::F32, DataType::F64], &[]), Ok(vec![DataType::F64]),);
        assert_eq!(
            data_operation.infer_output_types(&[DataType::F32], &[]),
            Err(TypeError::invalid("expected 2 inputs but got 1".to_string())),
        );
        assert_eq!(
            data_operation.infer_output_types(&[DataType::Boolean, DataType::Boolean], &[]),
            Err(TypeError::invalid("`test_binary` does not support input data type bool".to_string())),
        );
        assert_eq!(data_operation.infer_output_types(&[DataType::I64, DataType::I64], &[]), Ok(vec![DataType::I64]),);
        assert_eq!(
            data_operation.infer_output_types(&[DataType::C64, DataType::C64], &[]),
            Err(TypeError::invalid("`test_binary` does not support input data type c64".to_string())),
        );
        let scalar_type = ArrayType::scalar(DataType::F32);
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]));
        assert_eq!(
            Operation::infer_output_types(&array_operation, &[scalar_type, vector_type.clone()], &[]),
            Ok(vec![vector_type]),
        );
        assert_eq!(
            Operation::infer_output_types(
                &array_operation,
                &[ArrayType::scalar(DataType::C64), ArrayType::scalar(DataType::C64)],
                &[],
            ),
            Err(TypeError::invalid("`test_binary` does not support input data type c64".to_string())),
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
            Operation::infer_output_types(&array_operation, &[unreduced_x, unreduced_y], &[]),
            Err(TypeError::invalid("`test_binary` operands must be unreduced over the same axes".to_string())),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Array, TestBinaryOperation<ArrayType>>>::interpret(
                &array_operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(2.0f32), Array::scalar(3.0f32)],
            ),
            Ok(vec![Array::scalar(5.0f32)]),
        );
        let context = PartialEvaluationContext::new(EagerContext::<Array, TestBinaryOperation<ArrayType>>::new());
        let outputs = array_operation
            .partially_evaluate(
                &context,
                &EmptyRegionDriver,
                &[
                    PartialEvaluationValue::known(Array::scalar(2.0f32)),
                    PartialEvaluationValue::known(Array::scalar(3.0f32)),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].as_known(), Some(&Array::scalar(5.0f32)));
    }

    #[test]
    fn test_define_elementwise_operation_custom_inference() {
        let data_magnitude = TestMagnitudeOperation::<DataType>::new();
        let array_magnitude = TestMagnitudeOperation::<ArrayType>::new();
        assert_eq!(data_magnitude.infer_output_types(&[DataType::C64], &[]), Ok(vec![DataType::F32]),);
        let complex_vector = ArrayType::new(DataType::C128, Shape::new(vec![Dimension::Static(2)]));
        let real_vector = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]));
        assert_eq!(
            Operation::infer_output_types(&array_magnitude, std::slice::from_ref(&complex_vector), &[]),
            Ok(vec![real_vector.clone()]),
        );
        assert_eq!(
            ElementwiseOperation::infer_output_types(&array_magnitude, std::slice::from_ref(&complex_vector)),
            Ok(vec![real_vector]),
        );

        let data_strict_add = TestStrictAddOperation::<DataType>::new();
        let array_strict_add = TestStrictAddOperation::<ArrayType>::new();
        assert_eq!(data_strict_add.infer_output_types(&[DataType::F32, DataType::F64], &[]), Ok(vec![DataType::F64]),);
        let scalar = ArrayType::scalar(DataType::F32);
        assert_eq!(
            Operation::infer_output_types(&array_strict_add, &[scalar.clone(), scalar.clone()], &[]),
            Ok(vec![scalar.clone()]),
        );
        let vector = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]));
        let expected = Err(TypeError::invalid("test strict-add inputs must have identical array types".to_string()));
        assert_eq!(
            Operation::infer_output_types(&array_strict_add, &[scalar.clone(), vector.clone()], &[]),
            expected.clone(),
        );
        assert_eq!(ElementwiseOperation::infer_output_types(&array_strict_add, &[scalar, vector]), expected);
    }

    #[test]
    fn test_define_elementwise_capability_unary() {
        let context = TracingContext::<Array, TestUnaryOperation<ArrayType>>::new();
        let output = context.input(ArrayType::scalar(DataType::F32)).test_unary().unwrap();
        let builder = output.builder().borrow();
        assert_eq!(builder.instructions().len(), 1);
        assert_eq!(builder.instructions()[0].operation().name(), TEST_UNARY_OPERATION_NAME);
        assert_eq!(builder.instructions()[0].inputs().len(), 1);
        assert_eq!(builder.instructions()[0].outputs(), &[output.atom_id().unwrap()]);
    }

    #[test]
    fn test_define_elementwise_capability_binary() {
        let context = TracingContext::<Array, TestBinaryOperation<ArrayType>>::new();
        let left = context.input(ArrayType::scalar(DataType::F32));
        let right = context.input(ArrayType::scalar(DataType::F32));
        let output = left.test_binary(&right).unwrap();
        let builder = output.builder().borrow();
        assert_eq!(builder.instructions().len(), 1);
        assert_eq!(builder.instructions()[0].operation().name(), TEST_BINARY_OPERATION_NAME);
        assert_eq!(builder.instructions()[0].inputs(), &[left.atom_id().unwrap(), right.atom_id().unwrap()]);
        assert_eq!(builder.instructions()[0].outputs(), &[output.atom_id().unwrap()]);
    }

    #[test]
    fn test_impl_reference_free_dischargeable_operation() {
        // The array universe has no reference-typed spelling, which is valid here because the generated rule only
        // replays ordinary operands and rejects every live reference handle before inspecting its type.
        #[derive(Copy, Clone, Debug, PartialEq)]
        struct WholeArray;

        #[derive(Copy, Clone, Debug)]
        struct TestArrayReferenceDischarge;

        impl<C: Domain<Type = ArrayType>> ReferenceDischargePolicy<C> for TestArrayReferenceDischarge {
            type Referent = ArrayType;
            type Alias = WholeArray;

            fn storage_alias(_referent: &ArrayType) -> WholeArray {
                WholeArray
            }

            fn read(_context: &C, current: &C::Value, _alias: &WholeArray) -> Result<C::Value, ProgramError> {
                Ok(current.clone())
            }

            fn write(
                _context: &C,
                _current: &C::Value,
                replacement: C::Value,
                _alias: &WholeArray,
            ) -> Result<C::Value, ProgramError> {
                Ok(replacement)
            }
        }

        let operation = TestUnaryOperation::<ArrayType>::new();
        let context = ReferenceDischargeContext::<
            EagerContext<Array, TestUnaryOperation<ArrayType>>,
            TestArrayReferenceDischarge,
        >::new(EagerContext::new());
        let inputs = [ReferenceDischargeValue::Value(Array::scalar(2.0f32))];

        // A reference-free, region-free application replays verbatim through the destination, which executes it and
        // returns its outputs as ordinary carriers.
        assert_eq!(
            operation.discharge_references(&context, &EmptyRegionDriver, &inputs),
            Ok(vec![ReferenceDischargeValue::Value(Array::scalar(-2.0f32))]),
        );

        // An operand that is a live reference handle is rejected too, because a reference-touching operation owns
        // its own rewrite. The handle's own rendering is spliced into the expected diagnostic because a top-level
        // environment identity is minted process-globally and is therefore not stable across runs.
        let reference = ReferenceDischargeValue::from(
            context
                .bind_discharged(ReferenceType::new(ArrayType::scalar(DataType::F32)), Array::scalar(1.0f32))
                .unwrap(),
        );
        assert_eq!(
            operation.discharge_references(&context, &EmptyRegionDriver, &[reference.clone()]),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge expected a value operand 0 of `test_unary` but received {reference}",
            ))),
        );
    }

    #[test]
    fn test_impl_differentiable_operation() {
        // The generic macro forwards every differentiation dual to the caller-provided JVP body without imposing
        // elementwise alignment or structural-zero handling.
        let inputs = [
            DifferentiationDual::new(Array::scalar(2.0f32), Array::scalar(4.0f32)).unwrap(),
            DifferentiationDual::new(Array::scalar(3.0f32), Array::scalar(5.0f32)).unwrap(),
        ];
        let outputs = TestDifferentiableOperation::<ArrayType>::new()
            .jvp(&EagerContext::<Array, TestDifferentiableOperation<ArrayType>>::new(), &EmptyRegionDriver, &inputs)
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &Array::scalar(2.0f32));
        assert!(matches!(outputs[0].tangent(), MaybeZero::Value(tangent) if tangent == &Array::scalar(4.0f32)));

        // The generic transposition shell likewise forwards the complete partial-input and cotangent slices to the
        // supplied body and preserves the driver's static dispatch.
        let context = TracingContext::<Array, TestDifferentiableOperation<ArrayType>>::new();
        let output_cotangent = context.input(ArrayType::scalar(DataType::F32));
        let output_cotangent_id = output_cotangent.atom_id();
        let input_cotangents = TestDifferentiableOperation::<ArrayType>::new()
            .transpose(
                &mut TranspositionContext::new(context.clone()),
                &EmptyRegionDriver,
                &[
                    PartialValue::Unknown(ArrayType::scalar(DataType::F32)),
                    PartialValue::Unknown(ArrayType::scalar(DataType::F32)),
                ],
                &[MaybeZero::Value(output_cotangent)],
            )
            .unwrap();
        assert_eq!(input_cotangents.len(), 2);
        assert!(input_cotangents.iter().all(
            |cotangent| matches!(cotangent, MaybeZero::Value(cotangent) if cotangent.atom_id() == output_cotangent_id)
        ));
    }

    #[test]
    fn test_impl_differentiable_elementwise_operation_linear() {
        let inputs = [
            DifferentiationDual::new(Array::scalar(2.0f32), Array::scalar(4.0f32)).unwrap(),
            DifferentiationDual::new(Array::scalar(3.0f32), Array::scalar(5.0f32)).unwrap(),
        ];
        let outputs = AddOperation::new()
            .jvp(&EagerContext::<Array, ArrayOperation<Array>>::new(), &EmptyRegionDriver, &inputs)
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &Array::scalar(5.0f32));
        assert!(matches!(outputs[0].tangent(), MaybeZero::Value(tangent) if tangent == &Array::scalar(9.0f32)));

        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let output_cotangent = context.input(ArrayType::scalar(DataType::F32));
        let output_cotangent_id = output_cotangent.atom_id();
        let input_cotangents =
            <AddOperation<ArrayType> as TransposableOperation<Array, ArrayOperation<Array>>>::transpose(
                &AddOperation::new(),
                &mut TranspositionContext::new(context.clone()),
                &EmptyRegionDriver,
                &[
                    PartialValue::Unknown(ArrayType::scalar(DataType::F32)),
                    PartialValue::Unknown(ArrayType::scalar(DataType::F32)),
                ],
                &[MaybeZero::Value(output_cotangent.clone())],
            )
            .unwrap();
        assert_eq!(input_cotangents.len(), 2);
        assert!(matches!(
            &input_cotangents[0],
            MaybeZero::Value(cotangent) if cotangent.atom_id() == output_cotangent_id,
        ));
        assert!(matches!(
            &input_cotangents[1],
            MaybeZero::Value(cotangent) if cotangent.atom_id() == output_cotangent_id,
        ));
    }

    #[test]
    fn test_impl_differentiable_elementwise_operation_linear_negative_signs() {
        // A `[@negative, @positive]` rule combines both live tangents as `right - left` and negates a left-only
        // tangent. The primal values come from the stand-in `Sub` interpretation and are irrelevant to the rule.
        let context = EagerContext::<Array, TestReversedSubOperation<ArrayType>>::new();
        let inputs = [
            DifferentiationDual::new(Array::scalar(2.0f32), Array::scalar(4.0f32)).unwrap(),
            DifferentiationDual::new(Array::scalar(3.0f32), Array::scalar(5.0f32)).unwrap(),
        ];
        let outputs = TestReversedSubOperation::<ArrayType>::new().jvp(&context, &EmptyRegionDriver, &inputs).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(matches!(outputs[0].tangent(), MaybeZero::Value(tangent) if tangent == &Array::scalar(1.0f32)));
        let inputs = [
            DifferentiationDual::new(Array::scalar(2.0f32), Array::scalar(4.0f32)).unwrap(),
            DifferentiationDual::new_with_zero_tangent(Array::scalar(3.0f32)).unwrap(),
        ];
        let outputs = TestReversedSubOperation::<ArrayType>::new().jvp(&context, &EmptyRegionDriver, &inputs).unwrap();
        assert!(matches!(outputs[0].tangent(), MaybeZero::Value(tangent) if tangent == &Array::scalar(-4.0f32)));
        let inputs = [
            DifferentiationDual::new_with_zero_tangent(Array::scalar(2.0f32)).unwrap(),
            DifferentiationDual::new(Array::scalar(3.0f32), Array::scalar(5.0f32)).unwrap(),
        ];
        let outputs = TestReversedSubOperation::<ArrayType>::new().jvp(&context, &EmptyRegionDriver, &inputs).unwrap();
        assert!(matches!(outputs[0].tangent(), MaybeZero::Value(tangent) if tangent == &Array::scalar(5.0f32)));

        // A `[@negative, @negative]` rule combines both live tangents as `-(left + right)`.
        let context = EagerContext::<Array, TestNegatedAddOperation<ArrayType>>::new();
        let inputs = [
            DifferentiationDual::new(Array::scalar(2.0f32), Array::scalar(4.0f32)).unwrap(),
            DifferentiationDual::new(Array::scalar(3.0f32), Array::scalar(5.0f32)).unwrap(),
        ];
        let outputs = TestNegatedAddOperation::<ArrayType>::new().jvp(&context, &EmptyRegionDriver, &inputs).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(matches!(outputs[0].tangent(), MaybeZero::Value(tangent) if tangent == &Array::scalar(-9.0f32)));
        let inputs = [
            DifferentiationDual::new_with_zero_tangent(Array::scalar(2.0f32)).unwrap(),
            DifferentiationDual::new(Array::scalar(3.0f32), Array::scalar(5.0f32)).unwrap(),
        ];
        let outputs = TestNegatedAddOperation::<ArrayType>::new().jvp(&context, &EmptyRegionDriver, &inputs).unwrap();
        assert!(matches!(outputs[0].tangent(), MaybeZero::Value(tangent) if tangent == &Array::scalar(-5.0f32)));
    }

    #[test]
    fn test_impl_differentiable_elementwise_operation_linear_negative_transposition() {
        // A negative coefficient stages a negation of the output cotangent while a positive coefficient passes it
        // through unchanged.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let output_cotangent = context.input(ArrayType::scalar(DataType::F32));
        let output_cotangent_id = output_cotangent.atom_id();
        let input_cotangents =
            <TestReversedSubOperation<ArrayType> as TransposableOperation<Array, ArrayOperation<Array>>>::transpose(
                &TestReversedSubOperation::<ArrayType>::new(),
                &mut TranspositionContext::new(context.clone()),
                &EmptyRegionDriver,
                &[
                    PartialValue::Unknown(ArrayType::scalar(DataType::F32)),
                    PartialValue::Unknown(ArrayType::scalar(DataType::F32)),
                ],
                &[MaybeZero::Value(output_cotangent.clone())],
            )
            .unwrap();
        assert_eq!(input_cotangents.len(), 2);
        assert!(matches!(
            &input_cotangents[0],
            MaybeZero::Value(cotangent) if cotangent.atom_id() != output_cotangent_id,
        ));
        assert!(matches!(
            &input_cotangents[1],
            MaybeZero::Value(cotangent) if cotangent.atom_id() == output_cotangent_id,
        ));
        {
            let builder = context.builder().borrow();
            assert_eq!(builder.instructions().len(), 1);
            assert_eq!(builder.instructions()[0].operation().name(), "neg");
        }

        // Both coefficients of a `[@negative, @negative]` rule stage negations.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let output_cotangent = context.input(ArrayType::scalar(DataType::F32));
        let output_cotangent_id = output_cotangent.atom_id();
        let input_cotangents =
            <TestNegatedAddOperation<ArrayType> as TransposableOperation<Array, ArrayOperation<Array>>>::transpose(
                &TestNegatedAddOperation::<ArrayType>::new(),
                &mut TranspositionContext::new(context.clone()),
                &EmptyRegionDriver,
                &[
                    PartialValue::Unknown(ArrayType::scalar(DataType::F32)),
                    PartialValue::Unknown(ArrayType::scalar(DataType::F32)),
                ],
                &[MaybeZero::Value(output_cotangent)],
            )
            .unwrap();
        assert_eq!(input_cotangents.len(), 2);
        for input_cotangent in &input_cotangents {
            assert!(matches!(
                input_cotangent,
                MaybeZero::Value(cotangent) if cotangent.atom_id() != output_cotangent_id,
            ));
        }
        let builder = context.builder().borrow();
        assert_eq!(builder.instructions().len(), 2);
        assert_eq!(builder.instructions()[0].operation().name(), "neg");
        assert_eq!(builder.instructions()[1].operation().name(), "neg");
    }

    #[test]
    fn test_impl_differentiable_elementwise_operation_unary_jvp_contributions() {
        let outputs = SinOperation::new()
            .jvp(
                &EagerContext::<Array, ArrayOperation<Array>>::new(),
                &EmptyRegionDriver,
                &[DifferentiationDual::new(Array::scalar(0.0f32), Array::scalar(4.0f32)).unwrap()],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &Array::scalar(0.0f32));
        assert!(matches!(outputs[0].tangent(), MaybeZero::Value(tangent) if tangent == &Array::scalar(4.0f32)));

        let outputs = ExpOperation::new()
            .jvp(
                &EagerContext::<Array, ArrayOperation<Array>>::new(),
                &EmptyRegionDriver,
                &[DifferentiationDual::new(Array::scalar(0.0f32), Array::scalar(3.0f32)).unwrap()],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &Array::scalar(1.0f32));
        assert!(matches!(outputs[0].tangent(), MaybeZero::Value(tangent) if tangent == &Array::scalar(3.0f32)));
    }

    #[test]
    fn test_impl_differentiable_elementwise_operation_binary_jvp_contributions() {
        let outputs = MulOperation::new()
            .jvp(
                &EagerContext::<Array, ArrayOperation<Array>>::new(),
                &EmptyRegionDriver,
                &[
                    DifferentiationDual::new(Array::scalar(2.0f32), Array::scalar(4.0f32)).unwrap(),
                    DifferentiationDual::new(Array::scalar(3.0f32), Array::scalar(5.0f32)).unwrap(),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &Array::scalar(6.0f32));
        assert!(matches!(outputs[0].tangent(), MaybeZero::Value(tangent) if tangent == &Array::scalar(22.0f32)));
    }

    #[test]
    fn test_impl_differentiable_elementwise_operation_symmetric_transposition_diagnostics() {
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let output_cotangent = context.input(ArrayType::scalar(DataType::F32));
        let outputs = [MaybeZero::Value(output_cotangent)];
        assert!(matches!(
            <MulOperation<ArrayType> as TransposableOperation<Array, ArrayOperation<Array>>>::transpose(
                &MulOperation::new(),
                &mut TranspositionContext::new(context.clone()),
                &EmptyRegionDriver,
                &[PartialValue::Unknown(ArrayType::scalar(DataType::F32)), PartialValue::Unknown(ArrayType::scalar(DataType::F32))],
                &outputs,
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message
                    == "operation `mul` does not support transposition for input pattern \
                        [left = linear, right = linear]",
        ));

        let left = context.input(ArrayType::scalar(DataType::F32));
        let right = context.input(ArrayType::scalar(DataType::F32));
        assert!(matches!(
            <MulOperation<ArrayType> as TransposableOperation<Array, ArrayOperation<Array>>>::transpose(
                &MulOperation::new(),
                &mut TranspositionContext::new(context.clone()),
                &EmptyRegionDriver,
                &[PartialValue::Known(left), PartialValue::Known(right)],
                &outputs,
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message
                    == "operation `mul` does not support transposition for input pattern \
                        [left = known, right = known]",
        ));

        let right = context.input(ArrayType::scalar(DataType::I32));
        let output_cotangent = context.input(ArrayType::scalar(DataType::I32));
        assert!(matches!(
            <MulOperation<ArrayType> as TransposableOperation<Array, ArrayOperation<Array>>>::transpose(
                &MulOperation::new(),
                &mut TranspositionContext::new(context.clone()),
                &EmptyRegionDriver,
                &[PartialValue::Unknown(ArrayType::scalar(DataType::I32)), PartialValue::Known(right)],
                &[MaybeZero::Value(output_cotangent)],
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "linear input `left` of operation `mul` has no cotangent space",
        ));
    }

    #[test]
    fn test_impl_differentiable_elementwise_operation_one_sided_transposition_diagnostics() {
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let output_cotangent = context.input(ArrayType::scalar(DataType::F32));
        let outputs = [MaybeZero::Value(output_cotangent)];
        assert!(matches!(
            <DivOperation<ArrayType> as TransposableOperation<Array, ArrayOperation<Array>>>::transpose(
                &DivOperation::new(),
                &mut TranspositionContext::new(context.clone()),
                &EmptyRegionDriver,
                &[PartialValue::Unknown(ArrayType::scalar(DataType::F32)), PartialValue::Unknown(ArrayType::scalar(DataType::F32))],
                &outputs,
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message
                    == "operation `div` does not support transposition for input pattern \
                        [numerator = linear, denominator = linear]",
        ));

        let numerator = context.input(ArrayType::scalar(DataType::F32));
        let denominator = context.input(ArrayType::scalar(DataType::F32));
        assert!(matches!(
            <DivOperation<ArrayType> as TransposableOperation<Array, ArrayOperation<Array>>>::transpose(
                &DivOperation::new(),
                &mut TranspositionContext::new(context.clone()),
                &EmptyRegionDriver,
                &[PartialValue::Known(numerator), PartialValue::Known(denominator)],
                &outputs,
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message
                    == "operation `div` does not support transposition for input pattern \
                        [numerator = known, denominator = known]",
        ));
    }

    #[test]
    fn test_impl_differentiable_elementwise_operation_skips_structural_zero_contributions() {
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = context.input(ArrayType::scalar(DataType::F32));
        let outputs = SinOperation::new()
            .jvp(&context, &EmptyRegionDriver, &[DifferentiationDual::new_with_zero_tangent(input).unwrap()])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(matches!(
            outputs[0].tangent(),
            MaybeZero::Zero(r#type) if r#type == &ArrayType::scalar(DataType::F32),
        ));
        let builder = context.builder().borrow();
        assert_eq!(builder.instructions().len(), 1);
        assert_eq!(builder.instructions()[0].operation().name(), "sin");
    }

    #[test]
    fn test_impl_non_differentiable_operation() {
        // The basic form replays the primal operation and replaces its live tangent with a structural zero.
        let inputs = [DifferentiationDual::new(Array::scalar(2.0f32), Array::scalar(1.0f32)).unwrap()];
        let outputs = TestUnaryOperation::<ArrayType>::new()
            .jvp(&EagerContext::<Array, TestUnaryOperation<ArrayType>>::new(), &EmptyRegionDriver, &inputs)
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &Array::scalar(-2.0f32));
        assert!(matches!(
            outputs[0].tangent(),
            MaybeZero::Zero(r#type) if r#type == &ArrayType::scalar(DataType::F32),
        ));
        let context = TracingContext::<Array, TestUnaryOperation<ArrayType>>::new();
        let primal = context.input(ArrayType::scalar(DataType::F32));
        let tangent = context.input(ArrayType::scalar(DataType::F32));
        let inputs = [DifferentiationDual::new(primal, tangent).unwrap()];
        let outputs = TestUnaryOperation::<ArrayType>::new().jvp(&context, &EmptyRegionDriver, &inputs).unwrap();
        assert_eq!(context.builder().borrow().instructions().len(), 1);
        assert!(matches!(
            outputs[0].tangent(),
            MaybeZero::Zero(r#type) if r#type == &ArrayType::scalar(DataType::F32),
        ));

        // The generic form produces the same implementation shape without constraining its marker parameter.
        let operation = TestGenericNullaryOperation::<ArrayType>(PhantomData);
        let outputs = operation
            .jvp(&EagerContext::<Array, TestGenericNullaryOperation<ArrayType>>::new(), &EmptyRegionDriver, &[])
            .unwrap();
        assert!(outputs.is_empty());

        // The `where` forms remain usable for both non-generic and generic operation types.
        fn assert_differentiable<O>()
        where
            O: Operation<Type = ArrayType>
                + InterpretableOperation<EagerContext<Array, O>>
                + DifferentiableOperation<EagerContext<Array, O>>,
        {
        }

        assert_differentiable::<TestBoundedNullaryOperation<ArrayType>>();
    }

    #[test]
    fn test_impl_non_transposable_operation() {
        let context = TracingContext::<Array, TestUnaryOperation<ArrayType>>::new();
        let inputs: [PartialValue<Tracer<TracingContext<Array, TestUnaryOperation<ArrayType>>>>; 0] = [];
        let outputs: [MaybeZero<Tracer<TracingContext<Array, TestUnaryOperation<ArrayType>>>>; 0] = [];
        assert!(matches!(
            <TestUnaryOperation<ArrayType> as TransposableOperation<Array, TestUnaryOperation<ArrayType>>>::transpose(
                &TestUnaryOperation::<ArrayType>::new(),
                &mut TranspositionContext::new(context.clone()),
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
        let operation = TestNullaryOperation::<ArrayType>::new();
        let context = TracingContext::<Array, TestNullaryOperation<ArrayType>>::new();
        let inputs: [PartialValue<Tracer<TracingContext<Array, TestNullaryOperation<ArrayType>>>>; 0] = [];
        let outputs =
            [MaybeZero::Zero(ArrayType::scalar(DataType::F64)), MaybeZero::Zero(ArrayType::scalar(DataType::F64))];
        let result = <TestNullaryOperation<ArrayType> as TransposableOperation<
            Array,
            TestNullaryOperation<ArrayType>,
        >>::transpose(
            &operation,
            &mut TranspositionContext::new(context.clone()),
            &EmptyRegionDriver,
            &inputs,
            &outputs,
        )
        .unwrap();
        assert!(result.is_empty());
        let input = context.input(ArrayType::scalar(DataType::F64));
        assert!(matches!(
            <TestNullaryOperation<ArrayType> as TransposableOperation<Array, TestNullaryOperation<ArrayType>>>::transpose(
                &operation,
                &mut TranspositionContext::new(context.clone()),
                &EmptyRegionDriver,
                &[PartialValue::Known(input)],
                &outputs,
            ),
            Err(DifferentiationError::Program(ProgramError::InvalidInputCount { expected: 0, actual: 1 })),
        ));
        assert!(matches!(
            <TestNullaryOperation<ArrayType> as TransposableOperation<Array, TestNullaryOperation<ArrayType>>>::transpose(
                &operation,
                &mut TranspositionContext::new(context.clone()),
                &EmptyRegionDriver,
                &inputs,
                &[],
            ),
            Err(DifferentiationError::Program(ProgramError::InvalidOutputCount { expected: 2, actual: 0 })),
        ));

        fn assert_transposable<O: Operation<Type = ArrayType> + TransposableOperation<Array, O>>() {}

        assert_transposable::<TestGenericNullaryOperation<ArrayType>>();
        assert_transposable::<TestBoundedNullaryOperation<ArrayType>>();
    }

    #[test]
    fn test_impl_nullary_batchable_operation() {
        let operation = TestNullaryOperation::<ArrayType>::new();
        let context = BatchingContext::new(EagerContext::<Array, TestNullaryOperation<ArrayType>>::new(), 2);
        let outputs = <TestNullaryOperation<ArrayType> as BatchableOperation<
            EagerContext<Array, TestNullaryOperation<ArrayType>>,
            ArrayBatching,
        >>::batch(&operation, &context, &EmptyRegionDriver, &[])
        .unwrap()
        .into_parts()
        .0;
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].value(), &Array::scalar(3.0));
        assert_eq!(outputs[1].value(), &Array::scalar(4.0));
        assert!(outputs[0].batch_axis().is_replicated());
        assert!(outputs[1].batch_axis().is_replicated());
        assert!(matches!(
            <TestNullaryOperation<ArrayType> as BatchableOperation<
                EagerContext<Array, TestNullaryOperation<ArrayType>>,
                ArrayBatching,
            >>::batch(
                &operation, &context, &EmptyRegionDriver, &[ArrayBatch::replicated(Array::scalar(1.0))],
            ),
            Err(BatchingError::Program(ProgramError::InvalidInputCount { expected: 0, actual: 1 })),
        ));

        fn assert_batchable<O>()
        where
            O: Operation<Type = ArrayType>
                + InterpretableOperation<EagerContext<Array, O>>
                + BatchableOperation<EagerContext<Array, O>, ArrayBatching>,
        {
        }

        assert_batchable::<TestGenericNullaryOperation<ArrayType>>();
        assert_batchable::<TestBoundedNullaryOperation<ArrayType>>();
    }

    #[test]
    fn test_define_tracer_operator_unary() {
        let context = TracingContext::<Array, TestUnaryOperation<ArrayType>>::new();
        let input = context.input(ArrayType::scalar(DataType::F32));
        let input_id = input.atom_id().unwrap();
        let output = input.apply_unary();
        let builder = output.builder().borrow();
        assert_eq!(builder.instructions().len(), 1);
        assert_eq!(builder.instructions()[0].operation().name(), TEST_UNARY_OPERATION_NAME);
        assert_eq!(builder.instructions()[0].inputs(), &[input_id]);
        drop(builder);

        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = context.input(ArrayType::scalar(DataType::F32).into());
        let input = <Tracer<_> as ValueProjection<ArrayType>>::into_projected(input).unwrap();
        let output = input.apply_unary();
        let builder = context.builder().borrow();
        assert!(matches!(builder.instructions()[0].operation(), ArrayIrOperation::Array(ArrayOperation::Neg(_))));
        assert_eq!(output.value().atom_id(), Ok(builder.instructions()[0].outputs()[0]));
        drop(builder);

        let context = PartialEvaluationContext::new(EagerContext::<Array, TestUnaryOperation<ArrayType>>::new());
        let input = PartialTracer::new(context, PartialEvaluationValue::known(Array::scalar(2.0f32)));
        assert_eq!(input.apply_unary().into_value().unwrap().as_known(), Some(&Array::scalar(-2.0f32)));

        let context = BatchingContext::new(EagerContext::<Array, TestUnaryOperation<ArrayType>>::new(), 2);
        let input = BatchingTracer::new(context, ArrayBatch::replicated(Array::scalar(2.0f32)));
        let output = input.apply_unary().into_batch();
        assert_eq!(output.value(), &Array::scalar(-2.0f32));
        assert!(output.batch_axis().is_replicated());

        let context = DifferentiationContext::new(EagerContext::<Array, TestUnaryOperation<ArrayType>>::new());
        let input = DifferentiationTracer::new(
            DifferentiationDual::new(Array::scalar(2.0f32), Array::scalar(1.0f32)).unwrap(),
            context,
        );
        let output = input.apply_unary().into_dual();
        assert_eq!(output.primal(), &Array::scalar(-2.0f32));
        assert!(matches!(
            output.tangent(),
            MaybeZero::Zero(r#type) if r#type == &ArrayType::scalar(DataType::F32),
        ));
    }

    #[test]
    fn test_define_tracer_operator_binary() {
        let context = TracingContext::<Array, TestBinaryOperation<ArrayType>>::new();
        let left = context.input(ArrayType::scalar(DataType::F32));
        let right = context.input(ArrayType::scalar(DataType::F32));
        let input_ids = [left.atom_id().unwrap(), right.atom_id().unwrap()];
        let output = left.apply_binary(right);
        let builder = output.builder().borrow();
        assert_eq!(builder.instructions().len(), 1);
        assert_eq!(builder.instructions()[0].operation().name(), TEST_BINARY_OPERATION_NAME);
        assert_eq!(builder.instructions()[0].inputs(), &input_ids);
        drop(builder);

        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let left = context.input(ArrayType::scalar(DataType::F32).into());
        let right = context.input(ArrayType::scalar(DataType::F32).into());
        let left = <Tracer<_> as ValueProjection<ArrayType>>::into_projected(left).unwrap();
        let right = <Tracer<_> as ValueProjection<ArrayType>>::into_projected(right).unwrap();
        let output = left.apply_binary(right);
        let builder = context.builder().borrow();
        assert!(matches!(builder.instructions()[0].operation(), ArrayIrOperation::Array(ArrayOperation::Add(_))));
        assert_eq!(output.value().atom_id(), Ok(builder.instructions()[0].outputs()[0]));
        drop(builder);

        let context = PartialEvaluationContext::new(EagerContext::<Array, TestBinaryOperation<ArrayType>>::new());
        let left = PartialTracer::new(context.clone(), PartialEvaluationValue::known(Array::scalar(2.0f32)));
        let right = PartialTracer::new(context, PartialEvaluationValue::known(Array::scalar(3.0f32)));
        assert_eq!(left.apply_binary(right).into_value().unwrap().as_known(), Some(&Array::scalar(5.0f32)),);

        let context = BatchingContext::new(EagerContext::<Array, TestBinaryOperation<ArrayType>>::new(), 2);
        let left = BatchingTracer::new(context.clone(), ArrayBatch::replicated(Array::scalar(2.0f32)));
        let right = BatchingTracer::new(context, ArrayBatch::replicated(Array::scalar(3.0f32)));
        let output = left.apply_binary(right).into_batch();
        assert_eq!(output.value(), &Array::scalar(5.0f32));
        assert!(output.batch_axis().is_replicated());

        let context = DifferentiationContext::new(EagerContext::<Array, TestBinaryOperation<ArrayType>>::new());
        let left = DifferentiationTracer::new(
            DifferentiationDual::new(Array::scalar(2.0f32), Array::scalar(1.0f32)).unwrap(),
            context.clone(),
        );
        let right = DifferentiationTracer::new(
            DifferentiationDual::new(Array::scalar(3.0f32), Array::scalar(1.0f32)).unwrap(),
            context,
        );
        let output = left.apply_binary(right).into_dual();
        assert_eq!(output.primal(), &Array::scalar(5.0f32));
        assert!(matches!(
            output.tangent(),
            MaybeZero::Zero(r#type) if r#type == &ArrayType::scalar(DataType::F32),
        ));
    }

    #[test]
    fn test_define_tracer_operator_binary_with_provider() {
        // The array universe selects the ordinary elementwise test operation.
        let context = TracingContext::<Array, TestBinaryOperation<ArrayType>>::new();
        let left = context.input(ArrayType::scalar(DataType::F32));
        let right = context.input(ArrayType::scalar(DataType::F32));
        let input_ids = [left.atom_id().unwrap(), right.atom_id().unwrap()];
        let output = left.apply_provided_binary(right);
        let builder = output.builder().borrow();
        assert_eq!(builder.instructions().len(), 1);
        assert_eq!(builder.instructions()[0].operation().name(), TEST_BINARY_OPERATION_NAME,);
        assert_eq!(builder.instructions()[0].inputs(), &input_ids);
        drop(builder);

        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let left = context.input(ArrayType::scalar(DataType::F32).into());
        let right = context.input(ArrayType::scalar(DataType::F32).into());
        let left = <Tracer<_> as ValueProjection<ArrayType>>::into_projected(left).unwrap();
        let right = <Tracer<_> as ValueProjection<ArrayType>>::into_projected(right).unwrap();
        let output = left.apply_provided_binary(right);
        let builder = context.builder().borrow();
        assert!(matches!(builder.instructions()[0].operation(), ArrayIrOperation::Array(ArrayOperation::Add(_))));
        assert_eq!(output.value().atom_id(), Ok(builder.instructions()[0].outputs()[0]));
        drop(builder);

        // The same operator implementation selects the nominal dimension operation in the dimension universe.
        let left_type = DimensionType::new(DimensionVariable::new("left", DimensionBounds::new(1, Some(4)).unwrap()));
        let right_type = DimensionType::new(DimensionVariable::new("right", DimensionBounds::new(2, Some(6)).unwrap()));
        let context = TracingContext::<DimensionValue, TestArithmeticDimensionOperation>::new();
        let left = context.input(left_type);
        let right = context.input(right_type);
        let output = left.apply_provided_binary(right);
        assert_eq!(output.r#type().bounds(), DimensionBounds::new(3, Some(9)).unwrap());
        let builder = output.builder().borrow();
        assert_eq!(builder.instructions().len(), 1);
        assert_eq!(builder.instructions()[0].operation().name(), TEST_ARITHMETIC_DIMENSION_OPERATION_NAME);
    }

    #[test]
    fn test_check_gradient_rank_zero_array() {
        fn square<V: Clone + std::ops::Mul<Output = V>>(input: V) -> V {
            input.clone() * input
        }

        check_gradient!(square, at = Array::scalar(0.7), step = 1e-6, tolerance = 1e-6);
        check_gradient!(
            |input| input.abs(),
            at = Array::scalar(Complex::new(0.7f64, -0.3)),
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
            |input| square(input).reduce(&[0], ReductionKind::Sum),
            at = Array::vector(vec![0.7f64, -1.3, 2.1]),
            step = 1e-6,
            tolerance = 1e-6,
        );
        check_gradient!(
            |input| input.abs().map(|magnitudes| magnitudes.reduce(&[0], ReductionKind::Sum)),
            at = Array::vector(vec![Complex::new(0.7f64, -0.3), Complex::new(-1.2f64, 0.8)]),
            step = 1e-6,
            tolerance = 1e-6,
        );
    }

    #[test]
    fn test_check_gradient_with_captures() {
        fn scaled_sum_squares<V: Clone + std::ops::Mul<Output = V> + Reduce>(input: V, scale: V) -> V {
            (input.clone() * input).reduce(&[0], ReductionKind::Sum) * scale
        }

        check_gradient!(
            scaled_sum_squares,
            at = Array::vector(vec![0.7f64, -1.3, 2.1]),
            with = Array::scalar(2.5),
            step = 1e-6,
            tolerance = 1e-6,
        );
    }

    #[test]
    #[should_panic(expected = "finite-difference gradient checking requires an f64 or c128 input but got f32")]
    fn test_check_gradient_rank_zero_array_unsupported_input_type() {
        check_gradient!(|input| input, at = Array::scalar(0.7f32), step = 1e-3, tolerance = 1e-3);
    }

    #[test]
    #[should_panic(expected = "finite-difference gradient checking requires an f64 or c128 input but got f32")]
    fn test_check_gradient_array_unsupported_input_type() {
        check_gradient!(
            |input| input.reduce(&[0], ReductionKind::Sum),
            at = Array::vector(vec![0.7f32, -1.3]),
            step = 1e-3,
            tolerance = 1e-3,
        );
    }
}
