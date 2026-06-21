// TODO(eaplatanios): Review this module.

//! Tests for the `#[derive(Operation)]` procedural macro.
//!
//! These tests define local stand-in traits and types that mirror the shapes the derive emits against. That keeps the
//! macro test focused on generated code rather than on the current `ryft-core` implementation details.

#![allow(private_interfaces, dead_code)]

use std::marker::PhantomData;

/// Stand-in for `ryft_core::Type`.
trait Type {}

/// Stand-in for `ryft_core::TypeError`.
#[derive(Debug, PartialEq, Eq)]
struct TypeError;

/// Stand-in for `ryft_core::ProgramError`.
#[derive(Debug, PartialEq, Eq)]
struct ProgramError;

/// Stand-in for `ryft_core::Value`.
trait Value<T: Type> {}

/// Stand-in for `ryft_core::Operation`.
trait Operation<T: Type> {
    fn name(&self) -> &'static str;

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError>;

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        let _ = indentation;
        formatter.write_str(self.name())
    }
}

/// Stand-in for `ryft_core::AbstractTracingContext`.
struct AbstractTracingContext<'context, T: Type, V: Value<T>, O: Operation<T>> {
    marker: PhantomData<(&'context (), T, V, O)>,
}

/// Stand-in for `ryft_core::Cotangent`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Cotangent<'context, T: Type, V: Value<T>, O: Operation<T>> {
    label: &'static str,
    marker: PhantomData<(&'context (), T, V, O)>,
}

/// Stand-in for `ryft_core::TransposableOperation`.
trait TransposableOperation<T: Type, V: Value<T>, O: Operation<T>>: Operation<T> {
    fn transpose<'transpose>(
        &self,
        context: &mut AbstractTracingContext<'transpose, T, V, O>,
        input_types: &[&T],
        output_cotangents: &[Cotangent<'transpose, T, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, O>>, ProgramError>;
}

fn transposed<'context, T: Type, V: Value<T>, O: Operation<T>>(label: &'static str) -> Cotangent<'context, T, V, O> {
    Cotangent { label, marker: PhantomData }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DataType;

impl Type for DataType {}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ArrayType;

impl Type for ArrayType {}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Factor(i64);

impl Value<DataType> for Factor {}
impl Value<ArrayType> for Factor {}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ZeroOperation<T: Type> {
    r#type: T,
}

impl<T: Clone + Type> Operation<T> for ZeroOperation<T> {
    fn name(&self) -> &'static str {
        "zero"
    }

    fn infer_output_types(&self, _input_types: &[T]) -> Result<Vec<T>, TypeError> {
        Ok(vec![self.r#type.clone()])
    }
}

impl<T: Clone + Type, V: Value<T>, O: Operation<T>> TransposableOperation<T, V, O> for ZeroOperation<T> {
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, T, V, O>,
        _input_types: &[&T],
        _output_cotangents: &[Cotangent<'transpose, T, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, O>>, ProgramError> {
        Ok(vec![transposed("zero")])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AddOperation;

impl Operation<DataType> for AddOperation {
    fn name(&self) -> &'static str {
        "add"
    }

    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl<V: Value<DataType>, O: Operation<DataType>> TransposableOperation<DataType, V, O> for AddOperation {
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, DataType, V, O>,
        _input_types: &[&DataType],
        _output_cotangents: &[Cotangent<'transpose, DataType, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, DataType, V, O>>, ProgramError> {
        Ok(vec![transposed("add")])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ScaleOperation<T: Type, V> {
    factor: V,
    marker: PhantomData<T>,
}

impl<T: Clone + Type, V> Operation<T> for ScaleOperation<T, V> {
    fn name(&self) -> &'static str {
        "scale"
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl<T: Clone + Type, V: Value<T>, O: Operation<T>, F> TransposableOperation<T, V, O> for ScaleOperation<T, F> {
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, T, V, O>,
        _input_types: &[&T],
        _output_cotangents: &[Cotangent<'transpose, T, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, O>>, ProgramError> {
        Ok(vec![transposed("scale")])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ConstantOperation<T: Type, V> {
    value: V,
    marker: PhantomData<T>,
}

impl<T: Clone + Type, V> Operation<T> for ConstantOperation<T, V> {
    fn name(&self) -> &'static str {
        "constant"
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl<T: Clone + Type, V: Value<T>, O: Operation<T>, F> TransposableOperation<T, V, O> for ConstantOperation<T, F> {
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, T, V, O>,
        _input_types: &[&T],
        _output_cotangents: &[Cotangent<'transpose, T, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, O>>, ProgramError> {
        Ok(vec![transposed("constant")])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CustomJvpOperation<T: Type, V> {
    tag: &'static str,
    marker: PhantomData<(T, V)>,
}

impl<T: Clone + Type, V> Operation<T> for CustomJvpOperation<T, V> {
    fn name(&self) -> &'static str {
        "custom_jvp"
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        Ok(input_types.to_vec())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation)]
enum ScalarOperation<V: Value<DataType>> {
    Zero(ZeroOperation<DataType>),
    Add(AddOperation),
    Scale(ScaleOperation<DataType, V>),
    CustomJvp(Box<CustomJvpOperation<DataType, V>>),
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation, ryft::TransposableOperation)]
enum LinearScalarOperation<V: Value<DataType>, C: Value<DataType> = V> {
    Zero(ZeroOperation<DataType>),
    Constant(ConstantOperation<DataType, V>),
    Add(AddOperation),
    Scale(ScaleOperation<DataType, C>),
}

#[test]
fn test_scalar_operation() {
    let zero = ScalarOperation::<Factor>::from(ZeroOperation { r#type: DataType });
    let add = ScalarOperation::<Factor>::from(AddOperation);
    let scale = ScalarOperation::<Factor>::from(ScaleOperation { factor: Factor(7), marker: PhantomData });
    let custom_jvp = ScalarOperation::<Factor>::from(CustomJvpOperation { tag: "tag", marker: PhantomData });

    assert_eq!(zero.name(), "zero");
    assert_eq!(add.name(), "add");
    assert_eq!(scale.name(), "scale");
    assert_eq!(custom_jvp.name(), "custom_jvp");

    assert_eq!(add.infer_output_types(&[DataType]), Ok(vec![DataType]));
    assert_eq!(zero.to_string(), "zero");
    assert_eq!(custom_jvp.to_string(), "custom_jvp");

    assert_eq!(<&ZeroOperation<DataType>>::try_from(&zero), Ok(&ZeroOperation { r#type: DataType }));
    assert_eq!(<&AddOperation>::try_from(&add), Ok(&AddOperation));
    assert_eq!(
        <&ScaleOperation<DataType, Factor>>::try_from(&scale),
        Ok(&ScaleOperation { factor: Factor(7), marker: PhantomData }),
    );
    assert_eq!(
        <&CustomJvpOperation<DataType, Factor>>::try_from(&custom_jvp),
        Ok(&CustomJvpOperation { tag: "tag", marker: PhantomData }),
    );
    assert_eq!(<&AddOperation>::try_from(&zero), Err(()));
}

#[test]
fn test_transposable_operation_infers_value_type() {
    type Linear = LinearScalarOperation<Factor>;

    let mut context = AbstractTracingContext::<DataType, Factor, Linear> { marker: PhantomData };
    let add = Linear::from(AddOperation);

    assert_eq!(
        add.transpose(&mut context, &[&DataType], &[]).unwrap(),
        vec![transposed::<DataType, Factor, Linear>("add")],
    );
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DotOperation;

impl Operation<ArrayType> for DotOperation {
    fn name(&self) -> &'static str {
        "dot"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        Ok(input_types.to_vec())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum NoExtension {}

impl Operation<ArrayType> for NoExtension {
    fn name(&self) -> &'static str {
        match *self {}
    }

    fn infer_output_types(&self, _input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        match *self {}
    }
}

impl<V: Value<ArrayType>, O: Operation<ArrayType>> TransposableOperation<ArrayType, V, O> for NoExtension {
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, ArrayType, V, O>,
        _input_types: &[&ArrayType],
        _output_cotangents: &[Cotangent<'transpose, ArrayType, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, O>>, ProgramError> {
        match *self {}
    }
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation)]
enum InferredArrayOperation<V: Value<ArrayType>, C: Value<ArrayType> = V> {
    Zero(ZeroOperation<ArrayType>),
    Constant(ConstantOperation<ArrayType, V>),
    Scale(ScaleOperation<ArrayType, C>),
}

#[test]
fn test_array_operation_type_inference() {
    type Operation = InferredArrayOperation<Factor>;

    let zero = Operation::from(ZeroOperation { r#type: ArrayType });
    let constant = Operation::from(ConstantOperation { value: Factor(5), marker: PhantomData });
    let scale = Operation::from(ScaleOperation { factor: Factor(17), marker: PhantomData });

    assert_eq!(zero.name(), "zero");
    assert_eq!(constant.name(), "constant");
    assert_eq!(scale.name(), "scale");
    assert_eq!(zero.infer_output_types(&[]), Ok(vec![ArrayType]));
    assert_eq!(
        <&ScaleOperation<ArrayType, Factor>>::try_from(&scale),
        Ok(&ScaleOperation { factor: Factor(17), marker: PhantomData }),
    );
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation)]
enum ArrayOperation<V: Value<ArrayType>, Extension = NoExtension> {
    Zero(ZeroOperation<ArrayType>),
    Dot(DotOperation),
    Scale(ScaleOperation<ArrayType, V>),
    Extension(Extension),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct WhileOperation<T: Type, V, O> {
    marker: PhantomData<(T, V, O)>,
}

impl<T: Clone + Type, V, O> Operation<T> for WhileOperation<T, V, O> {
    fn name(&self) -> &'static str {
        "while"
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl<T: Clone + Type, V: Value<T>, O: Operation<T>, W, P> TransposableOperation<T, V, O> for WhileOperation<T, W, P> {
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, T, V, O>,
        _input_types: &[&T],
        _output_cotangents: &[Cotangent<'transpose, T, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, O>>, ProgramError> {
        Ok(vec![transposed("while")])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RecomputeOperation<O> {
    operation: O,
}

impl<O: Operation<ArrayType>> Operation<ArrayType> for RecomputeOperation<O> {
    fn name(&self) -> &'static str {
        self.operation.name()
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        self.operation.infer_output_types(input_types)
    }
}

impl<V: Value<ArrayType>, O: Operation<ArrayType>, P: Operation<ArrayType>> TransposableOperation<ArrayType, V, O>
    for RecomputeOperation<P>
{
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, ArrayType, V, O>,
        _input_types: &[&ArrayType],
        _output_cotangents: &[Cotangent<'transpose, ArrayType, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, O>>, ProgramError> {
        Ok(vec![transposed("recompute")])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CustomVjpCallOperation<T: Type, C, O, F> {
    marker: PhantomData<(T, C, O, F)>,
}

impl<T: Clone + Type, C, O, F> Operation<T> for CustomVjpCallOperation<T, C, O, F> {
    fn name(&self) -> &'static str {
        "custom_vjp_call"
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl<T: Clone + Type, V: Value<T>, O: Operation<T>, C, P, F> TransposableOperation<T, V, O>
    for CustomVjpCallOperation<T, C, P, F>
{
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, T, V, O>,
        _input_types: &[&T],
        _output_cotangents: &[Cotangent<'transpose, T, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, O>>, ProgramError> {
        Ok(vec![transposed("custom_vjp_call")])
    }
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation, ryft::TransposableOperation)]
enum LinearArrayOperation<
    V: Value<ArrayType>,
    C: Value<ArrayType>,
    Extension = NoExtension,
    F: Value<ArrayType> = V,
    P: Operation<ArrayType> = ArrayOperation<C, Extension>,
> {
    Zero(ZeroOperation<ArrayType>),
    Scale(ScaleOperation<ArrayType, F>),
    Recompute(RecomputeOperation<P>),
    While(Box<WhileOperation<ArrayType, V, Self>>),
    CustomVjpCall(Box<CustomVjpCallOperation<ArrayType, C, P, F>>),
    Extension(Extension),
}

#[test]
fn test_array_operation_extension_conversion_skip() {
    let zero = ArrayOperation::<Factor>::from(ZeroOperation { r#type: ArrayType });
    let dot = ArrayOperation::<Factor>::from(DotOperation);
    let scale = ArrayOperation::<Factor>::from(ScaleOperation { factor: Factor(11), marker: PhantomData });

    assert_eq!(zero.name(), "zero");
    assert_eq!(dot.name(), "dot");
    assert_eq!(scale.name(), "scale");
    assert_eq!(dot.infer_output_types(&[ArrayType]), Ok(vec![ArrayType]));
    assert_eq!(dot.to_string(), "dot");

    assert_eq!(<&DotOperation>::try_from(&dot), Ok(&DotOperation));
    assert_eq!(
        <&ScaleOperation<ArrayType, Factor>>::try_from(&scale),
        Ok(&ScaleOperation { factor: Factor(11), marker: PhantomData }),
    );

    // If the derive generated `From<Extension>` for `Extension(Extension)`, it would overlap with `From<DotOperation>`
    // because `Extension` is unconstrained and could be `DotOperation`. This test target compiling proves that the
    // bare generic payload was skipped automatically.
}

#[test]
fn test_linear_array_operation_shape() {
    type Linear = LinearArrayOperation<Factor, Factor>;

    let zero = Linear::from(ZeroOperation { r#type: ArrayType });
    let scale = Linear::from(ScaleOperation { factor: Factor(13), marker: PhantomData });
    let recompute = Linear::from(RecomputeOperation { operation: ArrayOperation::<Factor>::from(DotOperation) });
    let while_operation = Linear::from(WhileOperation::<ArrayType, Factor, Linear> { marker: PhantomData });
    let custom_vjp_call =
        Linear::from(CustomVjpCallOperation::<ArrayType, Factor, ArrayOperation<Factor, NoExtension>, Factor> {
            marker: PhantomData,
        });

    assert_eq!(zero.name(), "zero");
    assert_eq!(scale.name(), "scale");
    assert_eq!(recompute.name(), "dot");
    assert_eq!(while_operation.name(), "while");
    assert_eq!(custom_vjp_call.name(), "custom_vjp_call");
    assert_eq!(while_operation.infer_output_types(&[ArrayType]), Ok(vec![ArrayType]));
    assert_eq!(recompute.infer_output_types(&[ArrayType]), Ok(vec![ArrayType]));

    assert_eq!(recompute, Linear::Recompute(RecomputeOperation { operation: ArrayOperation::from(DotOperation) }));
    assert_eq!(
        <&RecomputeOperation<ArrayOperation<Factor, NoExtension>>>::try_from(&recompute),
        Ok(&RecomputeOperation { operation: ArrayOperation::from(DotOperation) }),
    );
    assert_eq!(
        <&WhileOperation<ArrayType, Factor, Linear>>::try_from(&while_operation),
        Ok(&WhileOperation { marker: PhantomData }),
    );
    assert_eq!(
        <&CustomVjpCallOperation<ArrayType, Factor, ArrayOperation<Factor, NoExtension>, Factor>>::try_from(
            &custom_vjp_call
        ),
        Ok(&CustomVjpCallOperation { marker: PhantomData }),
    );
    assert_eq!(<&ZeroOperation<ArrayType>>::try_from(&while_operation), Err(()));

    // `Extension(Extension)` is a bare generic payload, so its conversion is skipped automatically, while the
    // recompute wrapper and boxed payloads still expose conversions.
}

#[test]
fn test_transposable_operation_forwards_to_variant_payloads() {
    type Linear = LinearArrayOperation<Factor, Factor>;

    let mut context = AbstractTracingContext::<ArrayType, Factor, Linear> { marker: PhantomData };

    let zero = Linear::from(ZeroOperation { r#type: ArrayType });
    let scale = Linear::from(ScaleOperation { factor: Factor(13), marker: PhantomData });
    let recompute = Linear::from(RecomputeOperation { operation: ArrayOperation::<Factor>::from(DotOperation) });
    let while_operation = Linear::from(WhileOperation::<ArrayType, Factor, Linear> { marker: PhantomData });
    let custom_vjp_call =
        Linear::from(CustomVjpCallOperation::<ArrayType, Factor, ArrayOperation<Factor, NoExtension>, Factor> {
            marker: PhantomData,
        });

    assert_eq!(
        zero.transpose(&mut context, &[&ArrayType], &[]).unwrap(),
        vec![transposed::<ArrayType, Factor, Linear>("zero")],
    );
    assert_eq!(
        scale.transpose(&mut context, &[&ArrayType], &[]).unwrap(),
        vec![transposed::<ArrayType, Factor, Linear>("scale")],
    );
    assert_eq!(
        recompute.transpose(&mut context, &[&ArrayType], &[]).unwrap(),
        vec![transposed::<ArrayType, Factor, Linear>("recompute")],
    );
    assert_eq!(
        while_operation.transpose(&mut context, &[&ArrayType], &[]).unwrap(),
        vec![transposed::<ArrayType, Factor, Linear>("while")],
    );
    assert_eq!(
        custom_vjp_call.transpose(&mut context, &[&ArrayType], &[]).unwrap(),
        vec![transposed::<ArrayType, Factor, Linear>("custom_vjp_call")],
    );
}

#[test]
fn test_errors() {
    let test_cases = trybuild::TestCases::new();
    test_cases.compile_fail("tests/operations/error_missing_type.rs");
    test_cases.compile_fail("tests/operations/error_ambiguous_type.rs");
    test_cases.compile_fail("tests/operations/error_bad_variant.rs");
    test_cases.compile_fail("tests/operations/error_bounds_attribute.rs");
    test_cases.compile_fail("tests/operations/error_type_attribute.rs");
}
