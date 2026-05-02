use std::fmt::{Debug, Display};

/// Type-driven constant operations and carrier capability traits.
pub mod constants;

pub use constants::*;

use crate::parameters::Parameterized;
use crate::tracing::{Program, Traceable, TracingError};
use crate::types::{Type, TypeError, Typed};

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
    pub fn program<
        T: Type,
        V: Traceable<T>,
        O: Clone + Operation<T>,
        Input: Parameterized<V>,
        Output: Parameterized<V>,
    >(
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

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::macros::check_input_count;
    use crate::parameters::Placeholder;
    use crate::tracing::{Program, ProgramBuilder};
    use crate::types::DataType;

    use super::*;

    #[derive(Clone, Debug)]
    struct IdentityOperation;

    impl Operation<DataType> for IdentityOperation {
        #[inline]
        fn name(&self) -> &'static str {
            "identity"
        }

        fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
            check_input_count!(input_types, 1, TypeError);
            Ok(vec![input_types[0]])
        }
    }

    impl InterpretableOperation<DataType, f64> for IdentityOperation {
        fn interpret(&self, inputs: &[f64]) -> Result<Vec<f64>, TracingError> {
            check_input_count!(inputs, 1, TracingError);
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
}
