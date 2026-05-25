use std::collections::BTreeSet;
use std::fmt::{Debug, Display};

use crate::broadcasting::Broadcastable;
use crate::macros::check_count;
use crate::parameters::Parameterized;
use crate::tracing::{Program, Traceable, TracingError};
use crate::types::{ArrayType, Type, TypeError, Typed};

/// Elementwise arithmetic operations and capability traits.
pub mod arithmetic;

/// Type-driven constant operations and capability traits.
pub mod constants;

/// Scalar operation types built from the core primitive operation traits.
pub mod scalars;

/// Elementwise trigonometric operations and capability traits.
pub mod trigonometric;

pub use arithmetic::*;
pub use constants::*;
pub use scalars::*;
pub use trigonometric::*;

/// Maximum length for the contents of a bracketed section in an [`OperationFormatter`] that should be rendered inline.
/// If the length exceeds this value, then the section contents will be rendered over multiple lines.
const MAX_INLINE_OPERATION_SECTION_CONTENTS_LENGTH: usize = 80;

/// Helper for rendering [`Operation`]s that supports proper bracketing and indentation for operation metadata.
/// [`OperationFormatter`] centralizes the indentation and bracket layout used by higher-order or metadata-carrying
/// operations. The operation name is written immediately by [`OperationFormatter::new`], while
/// [`OperationFormatter::bracketed`] owns the bracketed metadata delimiters. Scalar fields are buffered so that short
/// metadata can render inline when no nested program fields are present, while nested program fields force multiline
/// rendering.
pub struct OperationFormatter<'f, 'a> {
    /// [`Formatter`](std::fmt::Formatter) receiving the rendered text.
    formatter: &'f mut std::fmt::Formatter<'a>,

    /// Indentation of the rendered [`Instruction`](crate::tracing::Instruction) line that owns the
    /// [`Operation`] that is being rendered.
    indentation: usize,

    /// Buffered scalar field name-value pairs that may be rendered inline if no nested [`Program`] fields are present.
    fields: Vec<(String, String)>,

    /// Boolean indicating whether this [`Operation`] being rendered has been forced to use multiple lines.
    is_multiline: bool,
}

impl<'f, 'a> OperationFormatter<'f, 'a> {
    /// Creates a new [`OperationFormatter`] and writes the provided [`Operation`] name.
    #[inline]
    pub fn new(
        formatter: &'f mut std::fmt::Formatter<'a>,
        indentation: usize,
        name: &'static str,
    ) -> Result<Self, std::fmt::Error> {
        write!(formatter, "{name}")?;
        Ok(Self { formatter, indentation, fields: Vec::new(), is_multiline: false })
    }

    /// Renders the provided field name-value pair.
    #[inline]
    pub fn field(&mut self, name: &str, value: impl Display) -> std::fmt::Result {
        if self.is_multiline {
            write!(self.formatter, "\n{:indentation$}{name}={value},", "", indentation = self.indentation + 4)
        } else {
            self.fields.push((name.to_string(), value.to_string()));
            Ok(())
        }
    }

    /// Renders the provided nested field name-[`Program`] pair. This must be used for [`Program`]-valued fields.
    #[inline]
    pub fn program<T: Type, V: Traceable<T>, O: Operation<T>, Input: Parameterized<V>, Output: Parameterized<V>>(
        &mut self,
        name: &str,
        program: &Program<T, V, O, Input, Output>,
    ) -> std::fmt::Result {
        self.is_multiline = true;
        for (name, value) in std::mem::take(&mut self.fields) {
            write!(self.formatter, "\n{:indentation$}{name}={value},", "", indentation = self.indentation + 4)?;
        }
        writeln!(self.formatter)?;
        write!(self.formatter, "{:indentation$}", "", indentation = self.indentation + 4)?;
        writeln!(self.formatter, "{name}={{")?;
        program.render(self.formatter, self.indentation + 8)?;
        writeln!(self.formatter)?;
        write!(self.formatter, "{:indentation$}", "", indentation = self.indentation + 4)?;
        write!(self.formatter, "}},")
    }

    /// Renders a bracketed section (using square brackets) using the provided closure for rendering its contents.
    #[inline]
    pub fn bracketed(mut self, render_contents: impl FnOnce(&mut Self) -> std::fmt::Result) -> std::fmt::Result {
        write!(self.formatter, " [")?;
        render_contents(&mut self)?;
        let inline_contents_length = self
            .fields
            .iter()
            .enumerate()
            .map(|(index, (name, value))| name.len() + 1 + value.len() + if index == 0 { 0 } else { 2 })
            .sum::<usize>();
        if self.is_multiline || inline_contents_length > MAX_INLINE_OPERATION_SECTION_CONTENTS_LENGTH {
            self.is_multiline = true;
            for (name, value) in std::mem::take(&mut self.fields) {
                write!(self.formatter, "\n{:indentation$}{name}={value},", "", indentation = self.indentation + 4)?;
            }
            writeln!(self.formatter)?;
            write!(self.formatter, "{:indentation$}", "", indentation = self.indentation)?;
        } else {
            for (index, (name, value)) in self.fields.iter().enumerate() {
                if index > 0 {
                    write!(self.formatter, ", {name}={value}")?;
                } else {
                    write!(self.formatter, "{name}={value}")?;
                }
            }
        }
        write!(self.formatter, "]")
    }
}

/// [`Operation`] that can appear in [`Program`]s. [`Operation`] invocations are represented as
/// [`Instruction`](crate::tracing::Instruction)s in [`Program`]s. This trait represents the
/// high-level operation interface that only requires operations to be able to provide their name
/// and to infer their output [`Type`]s given their input [`Type`]s.
pub trait Operation<T: Type>: Debug {
    /// Returns the name of this [`Operation`] that is used in diagnostics and when rendering [`Program`]s as strings.
    fn name(&self) -> &'static str;

    /// Infers the output [`Type`]s of this [`Operation`] from the provided input [`Type`]s without executing it.
    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError>;

    /// Renders this [`Operation`] as part of an [`Instruction`](crate::tracing::Instruction). The default
    /// implementation simply renders [`Operation::name`]. Operations carrying semantic metadata
    /// or nested [`Program`]s should override this function and use [`OperationFormatter`] for
    /// consistent bracketed and indented formatting.
    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        let _ = indentation;
        formatter.write_str(self.name())
    }
}

/// [`InterpretableOperation`]s are [`Operation`]s that can be interpreted (i.e., executed) given concrete input values.
pub trait InterpretableOperation<T: Type, V: Typed<T>>: Operation<T> {
    /// Interprets this [`Operation`] given the provided input values and returns the resulting output values.
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError>;
}

/// Represents [`Operation`]s that operate elementwise on arrays and that support _broadcasting_ semantics.
/// [`ElementwiseOperation`] captures the shared type inference behavior of elementwise array operations:
/// implementations declare their fixed input count and operation name, while the default type inference implementation
/// checks the input count, broadcasts all input [`ArrayType`]s while tolerating shardings that differ only by
/// [`Sharding::varying_manual_axes`](crate::Sharding::varying_manual_axes).
pub trait ElementwiseOperation: Debug {
    /// Returns the name of this [`Operation`] that is used in diagnostics and when rendering [`Program`]s as strings.
    fn name(&self) -> &'static str;

    /// Returns the number of input arrays consumed by this elementwise [`Operation`].
    fn input_count(&self) -> usize;

    /// Infers the broadcasted output [`ArrayType`] for this elementwise [`Operation`].
    #[inline]
    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, self.input_count(), TypeError);
        match ArrayType::broadcasted(input_types) {
            Ok(output) => Ok(vec![output]),
            Err(_) => {
                // Ryft keeps generic [`ArrayType`] broadcasting conservative. Here we make binary primitives tolerate
                // differing varying manual axis (VMA) annotations by retrying type inference after erasing only the
                // VMA metadata, and then restoring the union of that metadata on the result, instead of weakening
                // generic `ArrayType` broadcasting everywhere.
                let original_varying_manual_axes = input_types
                    .iter()
                    .filter_map(|input_type| input_type.sharding.as_ref())
                    .flat_map(|sharding| sharding.varying_manual_axes.iter().cloned())
                    .collect::<BTreeSet<_>>();
                let mut input_types = input_types.to_vec();
                for sharding in input_types.iter_mut().filter_map(|input_type| input_type.sharding.as_mut()) {
                    sharding.varying_manual_axes.clear();
                }
                let mut output = ArrayType::broadcasted(input_types.as_slice()).map_err(|_| TypeError {
                    message: format!("{} input types are not broadcast-compatible", self.name()),
                })?;
                if let Some(sharding) = &mut output.sharding {
                    sharding.varying_manual_axes = original_varying_manual_axes;
                }
                Ok(vec![output])
            }
        }
    }
}

impl<O: ElementwiseOperation> Operation<ArrayType> for O {
    #[inline]
    fn name(&self) -> &'static str {
        ElementwiseOperation::name(self)
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        ElementwiseOperation::infer_output_types(self, input_types)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::macros::check_count;
    use crate::parameters::Placeholder;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tracing::{Program, ProgramBuilder};
    use crate::types::{ArrayType, DataType, Shape, Size};

    use super::*;

    #[derive(Clone, Debug)]
    struct IdentityOperation;

    impl Operation<DataType> for IdentityOperation {
        #[inline]
        fn name(&self) -> &'static str {
            "identity"
        }

        fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
            check_count!("input", input_types, 1, TypeError);
            Ok(vec![input_types[0]])
        }
    }

    impl InterpretableOperation<DataType, f64> for IdentityOperation {
        fn interpret(&self, inputs: &[f64]) -> Result<Vec<f64>, TracingError> {
            check_count!("input", inputs, 1, TracingError);
            Ok(vec![inputs[0]])
        }
    }

    #[derive(Clone, Debug)]
    struct InlineMetadataOperation;

    impl Operation<DataType> for InlineMetadataOperation {
        #[inline]
        fn name(&self) -> &'static str {
            "metadata"
        }

        fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
            IdentityOperation.infer_output_types(input_types)
        }

        fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
            OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
                operation.field("mode", "test")?;
                operation.field("count", 2)
            })
        }
    }

    #[derive(Clone, Debug)]
    struct LongMetadataOperation;

    impl LongMetadataOperation {
        const VALUE: &'static str =
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz";
    }

    impl Operation<DataType> for LongMetadataOperation {
        #[inline]
        fn name(&self) -> &'static str {
            "metadata"
        }

        fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
            IdentityOperation.infer_output_types(input_types)
        }

        fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
            OperationFormatter::new(formatter, indentation, self.name())?
                .bracketed(|operation| operation.field("value", Self::VALUE))
        }
    }

    #[derive(Clone, Debug)]
    struct NestedProgramOperation {
        program: Program<DataType, f64, IdentityOperation, f64, f64>,
    }

    impl Operation<DataType> for NestedProgramOperation {
        #[inline]
        fn name(&self) -> &'static str {
            "nested"
        }

        fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
            IdentityOperation.infer_output_types(input_types)
        }

        fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
            OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
                operation.field("tag", "before")?;
                operation.program("body", &self.program)?;
                operation.field("tag", "after")
            })
        }
    }

    struct RenderedOperation<'a, O> {
        operation: &'a O,
        indentation: usize,
    }

    impl<O: Operation<DataType>> std::fmt::Display for RenderedOperation<'_, O> {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.operation.render(formatter, self.indentation)
        }
    }

    fn render_operation(operation: &impl Operation<DataType>) -> String {
        RenderedOperation { operation, indentation: 0 }.to_string()
    }

    fn identity_program() -> Program<DataType, f64, IdentityOperation, f64, f64> {
        let mut builder = ProgramBuilder::<DataType, f64, IdentityOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(IdentityOperation, vec![input]).unwrap()[0];
        builder.build::<f64, f64>(vec![output], Placeholder, Placeholder).unwrap()
    }

    #[test]
    fn default_operation_rendering_uses_the_operation_name() {
        assert_eq!(render_operation(&IdentityOperation), "identity");
    }

    #[test]
    fn operation_inference_and_interpretation_use_concrete_inputs() {
        let operation = IdentityOperation;

        assert_eq!(operation.infer_output_types(&[DataType::F64]), Ok(vec![DataType::F64]));
        assert_eq!(
            operation.infer_output_types(&[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() })
        );
        assert_eq!(operation.interpret(&[3.0f64]), Ok(vec![3.0f64]));
        assert_eq!(operation.interpret(&[]), Err(TracingError::InvalidInputCount { expected: 1, got: 0 }));
    }

    #[test]
    fn operation_formatter_renders_short_fields_inline() {
        assert_eq!(render_operation(&InlineMetadataOperation), "metadata [mode=test, count=2]");
    }

    #[test]
    fn operation_formatter_wraps_long_fields_over_multiple_lines() {
        assert_eq!(
            render_operation(&LongMetadataOperation),
            format!(
                indoc! {"
                    metadata [
                        value={value},
                    ]
                "},
                value = LongMetadataOperation::VALUE,
            )
            .trim_end()
        );
    }

    #[test]
    fn operation_formatter_renders_program_fields_over_multiple_lines() {
        assert_eq!(
            render_operation(&NestedProgramOperation { program: identity_program() }),
            indoc! {"
                nested [
                    tag=before,
                    body={
                        lambda %0:f64 .
                        let %1:f64 = identity %0
                        in (%1)
                    },
                    tag=after,
                ]
            "}
            .trim_end()
        );
    }

    #[test]
    fn elementwise_array_operation() {
        #[derive(Clone, Debug)]
        struct TestElementwiseArrayOperation {
            input_count: usize,
        }

        impl ElementwiseOperation for TestElementwiseArrayOperation {
            #[inline]
            fn name(&self) -> &'static str {
                "elementwise_test"
            }

            #[inline]
            fn input_count(&self) -> usize {
                self.input_count
            }
        }

        let operation = TestElementwiseArrayOperation { input_count: 1 };
        let input_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2), Size::Static(3)]), None, None).unwrap();
        assert_eq!(Operation::<ArrayType>::infer_output_types(&operation, &[input_type.clone()]), Ok(vec![input_type]));
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );

        let operation = TestElementwiseArrayOperation { input_count: 3 };
        let output = Operation::<ArrayType>::infer_output_types(
            &operation,
            &[
                ArrayType::scalar(DataType::F32),
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]), None, None).unwrap(),
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(1), Size::Static(3)]), None, None).unwrap(),
            ],
        )
        .unwrap();
        assert_eq!(
            output,
            vec![
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]), None, None,).unwrap()
            ],
        );

        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let first = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(8)]),
            None,
            Some(
                Sharding::with_manual_axes(
                    mesh.clone(),
                    vec![ShardingDimension::sharded(["x"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["x"],
                )
                .unwrap(),
            ),
        )
        .unwrap();
        let second = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(8)]),
            None,
            Some(
                Sharding::with_manual_axes(
                    mesh.clone(),
                    vec![ShardingDimension::sharded(["x"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["y"],
                )
                .unwrap(),
            ),
        )
        .unwrap();
        let third = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(8)]),
            None,
            Some(
                Sharding::with_manual_axes(
                    mesh,
                    vec![ShardingDimension::sharded(["x"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["z"],
                )
                .unwrap(),
            ),
        )
        .unwrap();
        let output = Operation::<ArrayType>::infer_output_types(&operation, &[first, second, third]).unwrap();
        assert_eq!(
            output[0].sharding().unwrap().varying_manual_axes(),
            &BTreeSet::from(["x".to_string(), "y".to_string(), "z".to_string()]),
        );
    }
}
