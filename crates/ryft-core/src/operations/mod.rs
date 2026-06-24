use std::collections::BTreeSet;
use std::fmt::Display;

use half::{bf16, f16};

use crate::broadcasting::Broadcastable;
use crate::compilation::CapturedConstant;
use crate::contexts::Context;
use crate::macros::check_count;
use crate::parameters::Parameterized;
use crate::programs::{Program, ProgramError, Value};
use crate::tracing::Tracer;
use crate::types::{ArrayType, DataType, Type, TypeError};

// TODO(eaplatanios): Review this file.

/// Elementwise arithmetic operations and capability traits.
pub mod arithmetic;

/// Elementwise pairwise comparison operations and capability traits.
pub mod compare;

/// Type-driven constant operations and capability traits.
pub mod constants;

/// Higher-order control-flow operations and capability traits.
pub mod control_flow;

/// Differentiation-control operations and capability traits.
pub mod differentiation;

/// Elementwise logical operations and capability traits.
pub mod logical;

/// Array shape and axis manipulation operations and capability traits.
pub mod manipulation;

/// Shared marker types for operations with payload-dependent interpretation.
pub mod payloads;

/// Scalar operation types built from the core primitive operation traits.
pub mod scalars;

/// Sharding-related operations (e.g., resharding and propagation hints) and capability traits.
pub mod sharding;

/// Elementwise trigonometric operations and capability traits.
pub mod trigonometric;

// TODO(eaplatanios): We should be importing specific symbols here.
pub use arithmetic::*;
pub use compare::*;
pub use constants::*;
pub use control_flow::*;
pub use differentiation::*;
pub use logical::*;
pub use manipulation::*;
pub use scalars::*;
pub use sharding::*;
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

    /// Indentation of the rendered [`Instruction`](crate::Instruction) line that owns the [`Operation`]
    /// that is being rendered.
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
    pub fn program<T: Type, V: Value<T>, O: Operation<T>, Input: Parameterized<V>, Output: Parameterized<V>>(
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
/// [`Instruction`](crate::Instruction)s in [`Program`]s. This trait represents the high-level operation interface
/// that only requires operations to be able to provide their name and to infer their output [`Type`]s given their
/// input [`Type`]s.
///
/// # Deriving Operation Enums
///
/// Ryft provides a `#[derive(Operation)]` procedural macro via the `ryft-macros` crate for [`Operation`] sum types.
/// It is meant for enums such as [`ScalarOperation`], where every variant wraps one concrete operation payload and
/// the enum should behave exactly like whichever payload it contains. The derived implementation generates:
///
///   - An [`Operation<T>`](Operation) implementation whose [`name`](Operation::name),
///     [`infer_output_types`](Operation::infer_output_types), and [`render`](Operation::render) methods forward to the
///     active variant payload.
///   - A [`Display`] implementation that renders through [`Operation::render`] with zero indentation, so that the enum
///     display matches the canonical program rendering format.
///   - `From<Payload> for Enum` conversions for concrete payload variants.
///   - Borrowed `TryFrom<&Enum> for &Payload` conversions for concrete payload variants, including boxed payloads.
///
/// The macro has the following requirements:
///
///   - The derivation macro input must be an enum. Structs and unions are not supported.
///   - Every variant must be a tuple variant with exactly one payload field.
///   - A payload may be stored directly as `Payload` or indirectly as `Box<Payload>`. Boxed variants still delegate to
///     `Payload`, and their generated `From<Payload>` implementation boxes the payload for the caller.
///   - Every payload must implement [`Operation<T>`](Operation), either because it is a concrete operation type whose
///     implementation already exists or because the derivation macro adds a bound for a bare generic payload.
///   - Bare generic payload variants such as `Extension(Extension)` do not receive `From` or `TryFrom` conversions.
///     Generating those conversions would overlap with concrete variant conversions when the generic parameter is
///     instantiated as one of the concrete payload types. The operation forwarding implementation still supports the
///     generic payload by adding an `Extension: Operation<T>` bound.
///
/// The operation type `T` is selected as follows:
///
///   - If the enum has exactly one distinct generic bound of the form `Value<T>`, the derivation infers `T` from that
///     bound. For example, `enum BackendOperation<V: Value<ArrayType>>` derives `Operation<ArrayType>`. Multiple
///     generic parameters may use the same `T`. For example, `V: Value<ArrayType>, C: Value<ArrayType>` still infers
///     `ArrayType`.
///   - If no `Value<T>` bound is present, or if multiple distinct operation types are present (e.g., `Value<DataType>`
///     and `Value<ArrayType>`), the derivation macro cannot choose an operation type and reports a compilation error.
///     In those cases, the caller must split the enum by operation type or implement [`Operation`] manually.
///
/// The derivation macro also supports the `#[ryft(crate = "...")]` attribute to override the path used to reference
/// Ryft traits and error types from generated code. The default path is `ryft`, so downstream crates that depend on
/// the `ryft` crate normally do not need this attribute.
///
/// ## Example
///
/// ```rust
/// # use ryft_core as ryft;
/// # use ryft_core::{ConstantOperation, DataType, Operation, Value, ZeroOperation};
/// # use ryft_macros::Operation;
///
/// #[derive(Clone, Debug, Operation)]
/// enum BackendOperation<V: Value<DataType>> {
///     Zero(ZeroOperation<DataType>),
///     Constant(ConstantOperation<DataType, V>),
/// }
///
/// let operation = BackendOperation::<f32>::from(ZeroOperation::new(DataType::F32));
/// assert_eq!(operation.name(), "zero");
/// ```
pub trait Operation<T: Type> {
    /// Returns the name of this [`Operation`] that is used in diagnostics and when rendering [`Program`]s as strings.
    fn name(&self) -> &'static str;

    /// Infers the output [`Type`]s of this [`Operation`] from the provided input [`Type`]s without executing it.
    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError>;

    /// Renders this [`Operation`] as part of an [`Instruction`](crate::Instruction). The default implementation simply
    /// renders [`Operation::name`]. Operations carrying semantic metadata or nested [`Program`]s should override this
    /// function and use [`OperationFormatter`] for consistent bracketed and indented formatting.
    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        let _ = indentation;
        formatter.write_str(self.name())
    }
}

impl<T: Type, O: Operation<T> + ?Sized> Operation<T> for Box<O> {
    #[inline]
    fn name(&self) -> &'static str {
        self.as_ref().name()
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        self.as_ref().infer_output_types(input_types)
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        self.as_ref().render(formatter, indentation)
    }
}

/// [`Operation`]s that can be interpreted (i.e., executed) given concrete input values. Interpretation consumes input
/// values and returns outputs in `V`'s [`InterpretationContext`](Value::InterpretationContext). Eager implementations
/// execute value semantics directly. [`Tracer`] implementations are the staged replay path and may stage operations
/// into the active [`StagingContext`](crate::StagingContext) while preserving operation-owned lowering rules.
pub trait InterpretableOperation<T: Type, V: Value<T>>: Operation<T> {
    /// Interprets this [`Operation`] given the provided input values and returns the resulting output values. The
    /// provided `context` is the [`InterpretationContext`](Value::InterpretationContext) required to produce values
    /// of type `V`. For concrete (i.e., **eager**) values it is set to [`EagerContext`](crate::EagerContext) which
    /// does nothing. For [`Tracer`] values it is set to the surrounding [`StagingContext`](crate::StagingContext),
    /// which enables nullary operations (e.g., [`ZeroOperation`]) to stage themselves into that context rather than
    /// failing for lack of an operand from which to recover the context.
    fn interpret(&self, context: &V::InterpretationContext, inputs: &[V]) -> Result<Vec<V>, ProgramError>;
}

/// Represents closed [`Operation`] families that can recursively interpret nested flat [`Program`]s. This trait names
/// the recursive fixed point needed by higher-order interpretation helpers without requiring the full operation enum's
/// [`InterpretableOperation`] implementation while proving that implementation. Operation families implement it by
/// replaying nested flat [`Program`]s through their operation-owned interpretation rules.
pub trait InterpretableProgramOperation<T: Type, V: Value<T>>: Operation<T> + Sized {
    /// Interprets a nested flat [`Program`].
    ///
    /// # Parameters
    ///
    ///   - `context`: Interpretation context to use.
    ///   - `program`: Nested [`Program`] to interpret.
    ///   - `input`: Input values to use for interpreting the provided [`Program`].
    fn interpret_program(
        context: &V::InterpretationContext,
        program: &Program<T, V, Self, Vec<V>, Vec<V>>,
        input: Vec<V>,
    ) -> Result<Vec<V>, ProgramError>;
}

/// Represents [`Operation`]s that operate elementwise on arrays and that support _broadcasting_ semantics.
/// [`ElementwiseOperation`] captures the shared type inference behavior of elementwise array operations:
/// implementations declare their fixed input count, while the default type inference implementation checks
/// the input count, broadcasts all input [`ArrayType`]s while tolerating shardings that differ only by
/// [`Sharding::varying_manual_axes`](crate::Sharding::varying_manual_axes).
pub trait ElementwiseOperation: Operation<ArrayType> {
    /// Returns the number of input arrays consumed by this elementwise [`Operation`].
    fn input_count(&self) -> usize;

    /// Infers the broadcasted output [`ArrayType`] for this elementwise [`Operation`]. Operations whose output sharding
    /// does not follow plain broadcasting semantics (e.g., [`MulOperation`], which is bilinear in its operands and
    /// combines their reduction state accordingly) must override this function, typically using
    /// [`broadcast_output_type`](Self::broadcast_output_type) for the data type, shapes, and placement, and layering
    /// their own sharding rule on top.
    #[inline]
    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, self.input_count(), TypeError);
        Ok(vec![self.broadcast_output_type(input_types)?])
    }

    /// Broadcasts the elementwise operands into a single output [`ArrayType`], tolerating shardings that differ only by
    /// their [`Sharding::varying_manual_axes`](crate::Sharding::varying_manual_axes). Ryft keeps generic [`ArrayType`]
    /// broadcasting conservative, and so this function retries inference after erasing only the varying-manual-axis
    /// (VMA) metadata and then restores the union of that metadata on the result, instead of weakening generic
    /// [`ArrayType`] broadcasting everywhere.
    ///
    /// This is effectively a shared helper function for the default [`infer_output_types`](Self::infer_output_types)
    /// implementation and for operations that override that default to layer extra sharding rules on top of the
    /// broadcasted placement (e.g., [`MulOperation`]'s bilinear reduction-state rule).
    fn broadcast_output_type(&self, input_types: &[ArrayType]) -> Result<ArrayType, TypeError> {
        match ArrayType::broadcasted(input_types) {
            Ok(output) => Ok(output),
            Err(_) => {
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
                Ok(output)
            }
        }
    }
}

/// Represents [`Type`]s and [`Value`]s that have a Boolean counterpart and that may carry a scalar Rust Boolean.
/// [`BooleanLike`] is the shared contract between predicate-producing and predicate-consuming operations:
///
/// - **Predicate-Producing Operations (e.g., [`CompareOperation`]):** Call [`as_boolean`](Self::as_boolean)
///   on *type metadata* to infer their output types from their broadcasted input types. For type metadata (e.g.,
///   [`DataType`](crate::DataType) and [`ArrayType`]), the Boolean counterpart keeps the same structural metadata
///   (e.g., shape, layout, and sharding) but uses a Boolean element data type.
/// - **Predicate-Consuming Operations (e.g., [`ConditionOperation`] and [`WhileOperation`]):** Call
///   [`boolean`](Self::boolean) on *values* to extract the concrete scalar Rust Boolean that drives branching
///   or selection.
///
/// For values, [`as_boolean`](Self::as_boolean) reinterprets the carried payload as a Boolean value: zero maps to
/// `false` and any non-zero payload maps to `true`. Values that carry no concrete payload (e.g., staged tracers and
/// [`CapturedConstant`]s) cannot reinterpret anything and return themselves unchanged. Similarly,
/// [`boolean`](Self::boolean) errors for type metadata and for staged values because they carry no
/// concrete payload to decode.
pub trait BooleanLike {
    /// Returns the Boolean counterpart of this instance. For type metadata this is the same structural metadata with
    /// a Boolean data type, and for values this is the value with its payload reinterpreted as Boolean (i.e., zero
    /// maps to `false` and any non-zero payload maps to `true`).
    fn as_boolean(&self) -> Self;

    /// Extracts the scalar Rust Boolean value represented by this instance when there is one. For scalar values zero
    /// gets interpreted as `false` while non-zero values get interpreted as `true`, while for array values this
    /// requires a rank-0 Boolean-typed payload. Type metadata and staged values (e.g., tracers) error because they
    /// carry no concrete payload.
    fn boolean(&self) -> Result<bool, ProgramError>;
}

macro_rules! impl_boolean_like_for_scalar {
    ($($type:ty => ($zero:expr, $one:expr)),* $(,)?) => {
        $(
            impl BooleanLike for $type {
                #[inline]
                fn as_boolean(&self) -> Self {
                    if *self != $zero { $one } else { $zero }
                }

                #[inline]
                fn boolean(&self) -> Result<bool, ProgramError> {
                    Ok(*self != $zero)
                }
            }
        )*
    };
}

impl_boolean_like_for_scalar!(
    bf16 => (bf16::ZERO, bf16::ONE),
    f16 => (f16::ZERO, f16::ONE),
    f32 => (0.0, 1.0),
    f64 => (0.0, 1.0),
);

impl BooleanLike for DataType {
    #[inline]
    fn as_boolean(&self) -> Self {
        DataType::Boolean
    }

    #[inline]
    fn boolean(&self) -> Result<bool, ProgramError> {
        // `DataType` is type metadata and carries no concrete payload to decode.
        Err(ProgramError::Concretization {
            message: format!("cannot extract a concrete boolean from a data type instance ({self})"),
        })
    }
}

impl BooleanLike for ArrayType {
    #[inline]
    fn as_boolean(&self) -> Self {
        Self { data_type: DataType::Boolean, ..self.clone() }
    }

    #[inline]
    fn boolean(&self) -> Result<bool, ProgramError> {
        // `ArrayType` is only abstract staged-program metadata. It satisfies generic operation-enum bounds for
        // transform composition, but it never contains the concrete boolean needed to choose a branch.
        Err(ProgramError::Concretization {
            message: format!("cannot extract a concrete boolean from an array type instance ({self})"),
        })
    }
}

impl<C: Context<Type = ArrayType>> BooleanLike for Tracer<C> {
    #[inline]
    fn as_boolean(&self) -> Self {
        // Returns this `Tracer` unchanged. Tracers carry no concrete payload to reinterpret, and a staged Boolean
        // reinterpretation must be expressed explicitly in the traced program (e.g., via a comparison against zero)
        // rather than implicitly through this trait.
        self.clone()
    }

    #[inline]
    fn boolean(&self) -> Result<bool, ProgramError> {
        Err(ProgramError::Concretization { message: "cannot extract a concrete boolean from a tracer".to_string() })
    }
}

impl BooleanLike for CapturedConstant<ArrayType> {
    #[inline]
    fn as_boolean(&self) -> Self {
        // Returns this [`CapturedConstant`] unchanged. A captured constant is a reference into a side table,
        // not the concrete value itself, so there is no payload to reinterpret here.
        self.clone()
    }

    #[inline]
    fn boolean(&self) -> Result<bool, ProgramError> {
        // A captured constant is a reference into a side table, not the concrete predicate value itself. Control-flow
        // staging must keep predicates in the IR or add a transform-specific rule instead of trying to branch here.
        Err(ProgramError::Concretization {
            message: "cannot extract a concrete boolean from a captured constant reference".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::macros::check_count;
    use crate::parameters::Placeholder;
    use crate::programs::{Program, ProgramBuilder, ProgramError, Value};
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
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
        fn interpret(
            &self,
            _context: &<f64 as Value<DataType>>::InterpretationContext,
            inputs: &[f64],
        ) -> Result<Vec<f64>, ProgramError> {
            check_count!("input", inputs, 1, ProgramError);
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
        assert_eq!(operation.interpret(&crate::EagerContext::new(), &[3.0f64]), Ok(vec![3.0f64]));
        assert_eq!(
            operation.interpret(&crate::EagerContext::new(), &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
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

        impl Operation<ArrayType> for TestElementwiseArrayOperation {
            #[inline]
            fn name(&self) -> &'static str {
                "elementwise_test"
            }

            #[inline]
            fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
                ElementwiseOperation::infer_output_types(self, input_types)
            }
        }

        impl ElementwiseOperation for TestElementwiseArrayOperation {
            #[inline]
            fn input_count(&self) -> usize {
                self.input_count
            }
        }

        let operation = TestElementwiseArrayOperation { input_count: 1 };
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2), Size::Static(3)]));
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
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)])),
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(1), Size::Static(3)])),
            ],
        )
        .unwrap();
        assert_eq!(output, vec![ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]))],);

        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let first = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
            .with_sharding(
                Sharding::with_manual_axes(
                    mesh.clone(),
                    vec![ShardingDimension::sharded(["x"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["x"],
                )
                .unwrap(),
            )
            .unwrap();
        let second = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
            .with_sharding(
                Sharding::with_manual_axes(
                    mesh.clone(),
                    vec![ShardingDimension::sharded(["x"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["y"],
                )
                .unwrap(),
            )
            .unwrap();
        let third = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
            .with_sharding(
                Sharding::with_manual_axes(
                    mesh,
                    vec![ShardingDimension::sharded(["x"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["z"],
                )
                .unwrap(),
            )
            .unwrap();
        let output = Operation::<ArrayType>::infer_output_types(&operation, &[first, second, third]).unwrap();
        assert_eq!(
            output[0].sharding().unwrap().varying_manual_axes(),
            &BTreeSet::from(["x".to_string(), "y".to_string(), "z".to_string()]),
        );

        // Dynamic dimensions flow through elementwise congruence when they match exactly, while static-vs-dynamic
        // mismatches are rejected.
        let operation = TestElementwiseArrayOperation { input_count: 2 };
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Dynamic(None), Size::Static(3)]));
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[dynamic_type.clone(), dynamic_type.clone()]),
            Ok(vec![dynamic_type.clone()]),
        );
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(
                &operation,
                &[dynamic_type, ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2), Size::Static(3)]))],
            ),
            Err(TypeError { message: "elementwise_test input types are not broadcast-compatible".to_string() }),
        );
    }
}
