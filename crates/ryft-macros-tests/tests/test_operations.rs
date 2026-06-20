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
#[ryft(type = "DataType")]
enum ScalarOperation<V: Value<DataType>> {
    Zero(ZeroOperation<DataType>),
    Add(AddOperation),
    Scale(ScaleOperation<DataType, V>),
    CustomJvp(Box<CustomJvpOperation<DataType, V>>),
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

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation)]
#[ryft(type = "ArrayType")]
enum ArrayOperation<T: Type, V: Value<T>, Extension = NoExtension> {
    Zero(ZeroOperation<T>),
    Dot(DotOperation),
    Scale(ScaleOperation<T, V>),
    Extension(Extension),
}

#[test]
fn test_primary_pin_and_extension_conversion_skip() {
    let zero = ArrayOperation::<ArrayType, Factor>::from(ZeroOperation { r#type: ArrayType });
    let dot = ArrayOperation::<ArrayType, Factor>::from(DotOperation);
    let scale = ArrayOperation::<ArrayType, Factor>::from(ScaleOperation { factor: Factor(11), marker: PhantomData });

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
fn test_errors() {
    let test_cases = trybuild::TestCases::new();
    test_cases.compile_fail("tests/operations/error_missing_type.rs");
    test_cases.compile_fail("tests/operations/error_bad_variant.rs");
}
