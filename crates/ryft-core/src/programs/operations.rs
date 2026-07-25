use std::fmt::Display;

use crate::parameters::Parameterized;
use crate::programs::Value;
use crate::programs::effects::Effects;
use crate::programs::identities::TypeIdentityRenaming;
use crate::programs::programs::Program;
use crate::programs::regions::{OutputRegionProvenance, RegionInterface};
use crate::programs::types::{Type, TypeError};

/// Maximum length for the contents of a bracketed section in an [`OperationFormatter`] that should be rendered inline.
/// If the length exceeds this value, then the section contents will be rendered over multiple lines.
const MAX_INLINE_OPERATION_SECTION_CONTENTS_LENGTH: usize = 80;

/// Helper for rendering [`Operation`]s that supports proper bracketing and indentation for operation metadata.
/// [`OperationFormatter`] centralizes the indentation and bracket layout used by metadata-carrying operations. The
/// operation name is written immediately by [`OperationFormatter::new`], while [`OperationFormatter::bracketed`] owns
/// the bracketed metadata delimiters. Scalar fields are buffered so that short metadata can render inline when no
/// [`Program`]-valued metadata fields are present, while program-valued fields force multiline rendering. Note that
/// attached nested [`Region`](crate::Region)s do not render through this formatter as the contextual program
/// renderer owns them.
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

    /// Renders the provided nested field name-[`Program`] pair. This must be used for [`Program`]-valued metadata
    /// fields; attached [`Region`](crate::Region)s render through the contextual program renderer instead.
    #[inline]
    pub fn program<V: Value, O: Operation<V::Type>, Input: Parameterized<V>, Output: Parameterized<V>>(
        &mut self,
        name: &str,
        program: &Program<V, O, Input, Output>,
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
/// It is meant for enums such as [`ScalarOperation`](crate::ScalarOperation), where every variant wraps one concrete
/// operation payload and the enum should behave exactly like whichever payload it contains. The derived implementation
/// generates:
///
///   - An [`Operation<T>`](Operation) implementation whose semantic and rendering methods forward to the active
///     variant payload.
///   - An [`InterpretableOperation<C>`](crate::InterpretableOperation) implementation that forwards
///     [`interpret`](crate::InterpretableOperation::interpret) to the active variant payload. Operation-specific eager
///     or staged interpretation semantics still live on the payload implementations; the enum is only a dispatcher.
///   - A [`PartiallyEvaluatableOperation<C>`](crate::PartiallyEvaluatableOperation) implementation that likewise
///     forwards to each concrete payload's partial-evaluation rule.
///   - Optional [`BatchableOperation`](crate::BatchableOperation),
///     [`DifferentiableOperation`](crate::DifferentiableOperation), and
///     [`TransposableOperation`](crate::TransposableOperation) dispatchers selected independently through
///     `#[ryft(dispatch(batching, differentiation, transposition))]`.
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
///   - Each generated semantic dispatcher receives a corresponding bound on every payload, such as
///     `Payload: InterpretableOperation<C>` or `Payload: BatchableOperation<C>`. Payload-specific value and context
///     requirements belong on the payload's own semantic-trait implementation; Rust resolves those requirements through
///     the generated payload bound without requiring the enum to repeat them.
///   - Batching, differentiation, and transposition require selecting the corresponding dispatcher. Interpretation
///     and partial evaluation are always generated and therefore do not appear in the `dispatch(...)` attribute.
///
/// The operation type `T` is selected as follows:
///
///   - If the enum has exactly one distinct generic bound of the form `Value<Type = T>`, the derivation infers `T` from
///     that bound. For example, `enum BackendOperation<V: Value<Type = ArrayType>>` derives `Operation<ArrayType>`.
///     Multiple generic parameters may use the same `T`. For example,
///     `V: Value<Type = ArrayType>, C: Value<Type = ArrayType>` still infers `ArrayType`.
///   - If no `Value<Type = T>` bound is present, or if multiple distinct operation types are present (e.g.,
///     `Value<Type = DataType>` and `Value<Type = ArrayType>`), the derivation macro cannot choose an operation type
///     and reports a compilation error. In those cases, the caller must split the enum by operation type or implement
///     [`Operation`] manually.
///
/// The value types used for interpretation are inferred from the enum's `Value<Type = T>` generic parameters:
///
///   - For enums with one `Value<Type = T>` parameter, the payload parameter is treated as the nested program's
///     captured constant type `C`, and the derived [`InterpretableOperation`](crate::InterpretableOperation)
///     implementation is generic over a runtime value `V` and an interpretation context. The generated implementations
///     require that context to be a [`Constant`](crate::Constant) provider that can lift captured constants from `C`
///     into `V`.
///   - For enums with two or more `Value<Type = T>` parameters, the first value parameter is treated as both the
///     runtime value type and the nested program constant type for direct program interpretation. Later value
///     parameters remain payload-specific metadata; when a dispatcher needs to instantiate a direct-linear operation
///     family, extra value parameters after the first two are substituted with the first value parameter.
///   - The generated dispatcher inherits value capabilities from the payload-owned
///     [`InterpretableOperation`](crate::InterpretableOperation) implementations. For example, a payload that extracts
///     a predicate states its [`Concretizable<bool>`](crate::Concretizable) requirement on its own implementation, and
///     the generated payload bound transports that requirement to the enum's use site.
///
/// The derivation macro supports the `#[ryft(crate = "...")]` attribute to override the path used to reference Ryft
/// traits and error types from generated code. The default path is `ryft`, so downstream crates that depend on the
/// `ryft` crate normally do not need this attribute. The `#[ryft(dispatch(...))]` attribute selects optional transform
/// dispatchers.
///
/// ## Example
///
/// ```rust
/// # use ryft_core as ryft;
/// # use ryft_core::{ConstantOperation, DataType, Operation, Value, ZeroOperation};
/// # use ryft_core::backends::scalars::Scalar;
/// # use ryft_macros::Operation;
///
/// #[derive(Clone, Debug, Operation)]
/// enum BackendOperation<V: Value<Type = DataType>> {
///     Zero(ZeroOperation<DataType>),
///     Constant(ConstantOperation<V>),
/// }
///
/// let operation = BackendOperation::<Scalar>::from(ZeroOperation::new(DataType::F32));
/// assert_eq!(operation.name(), "zero");
/// ```
pub trait Operation<T: Type>: Clone {
    /// Returns the name of this [`Operation`] that is used in diagnostics and when rendering [`Program`]s as strings.
    fn name(&self) -> &'static str;

    /// Returns stable names for this [`Operation`]'s attached-[`Region`](crate::Region) slots, in the operation-defined
    /// region order (e.g., `["true", "false"]` for a condition operation). The declared region count must match the
    /// number of regions attached to every [`Instruction`](crate::Instruction) applying this operation.
    /// [`ProgramBuilder`](crate::ProgramBuilder)s validate this both when the instruction is added and when the final
    /// [`Program`] is built. The names also label the region slots when rendering [`Program`]s. The default declares
    /// no region slots, which is correct for region-free operations.
    #[inline]
    fn region_names(&self) -> &'static [&'static str] {
        &[]
    }

    /// Derives the input [`Type`]s with which each attached [`Region`](crate::Region) will be invoked when this
    /// [`Operation`] receives `input_types`. An attached region is traced independently with a declared input
    /// signature. Before importing that region into the caller's [`Program`], staging must rename the signature's
    /// formal [`TypeIdentity`](crate::TypeIdentity)s to the identities supplied by this particular operation
    /// application. The mapping is operation-specific. For example, a condition passes every instruction input except
    /// its predicate to both branches, while a scan passes loop-carried values together with element types derived from
    /// its stacked inputs.
    ///
    /// Each returned entry corresponds to the same-index entry in `region_interfaces`. [`None`] preserves the region's
    /// declared signature and its existing sharing. [`Some`] provides the concrete region input types from which the
    /// generic staging machinery derives and applies the necessary type-identity renaming. Implementations must return
    /// exactly one entry per attached region.
    #[inline]
    fn infer_region_input_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<Option<Vec<T>>>, TypeError> {
        let _ = input_types;
        Ok(vec![None; region_interfaces.len()])
    }

    /// Infers the output [`Type`]s of this [`Operation`] from the provided input [`Type`]s and
    /// attached-region [`RegionInterface`]s without executing it, validating the complete hypothetical
    /// [`Instruction`](crate::Instruction) that the arguments describe.
    ///
    /// [`ProgramBuilder`](crate::ProgramBuilder)s never ask their callers to provide region interfaces. They receive
    /// input/operand atoms plus attached [`RegionId`](crate::RegionId)s referencing sealed [`Region`](crate::Region)s,
    /// derive the interface slice from their own [`Region`](crate::Region) arena, and invoke this function internally.
    /// Calling this function directly with synthetic [`RegionInterface`] performs a pure hypothetical inference and
    /// cannot mutate or create a [`Program`].
    ///
    /// # Parameters
    ///
    ///   - `input_types`: Input/operand [`Type`]s in instruction input order.
    ///   - `region_interfaces`: Boundary [`RegionInterface`] derived from the instruction's attached regions, in the
    ///     [`Operation`]-defined [`region_names`](Self::region_names) region order. Region-free operations receive an
    ///     empty slice and ignore it.
    fn infer_output_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError>;

    /// Returns information about which attached-[`Region`](crate::Region) `output_index`-th output can come from.
    /// An empty vector means that the [`Instruction`](crate::Instruction) result is produced by the [`Operation`]
    /// itself. A non-empty vector forwards the result to every listed region output, in semantic order. Analyses
    /// recursively resolve those outputs and may therefore recover no instruction producer when a path ends at a
    /// region input or constant. Region-forwarding operations such as the condition and scan operations override this
    /// function. The empty default is correct for every operation whose results are not forwarded region outputs.
    #[inline]
    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        let _ = output_index;
        Vec::new()
    }

    /// Returns `true` if the `output_index`-th output is structurally known to be zero independently
    /// of this [`Operation`]'s inputs. The default implementation returns `false`. Operations such as
    /// [`ZeroOperation`](crate::ZeroOperation) and [`ZeroLikeOperation`](crate::ZeroLikeOperation) override
    /// it so that transforms can preserve symbolic zero-ness when replaying otherwise opaque [`Program`]s.
    /// Implementations must return `false` for out-of-range indices.
    #[inline]
    fn is_zero(&self, output_index: usize) -> bool {
        let _ = output_index;
        false
    }

    /// Returns the observable [`Effect`](crate::Effect) classes of this [`Operation`]. Refer to the documentation of
    /// [`Effects`] and [`Effect`](crate::Effect) for the semantics.
    #[inline]
    fn effects(&self) -> Effects {
        Effects::PURE
    }

    /// Returns this [`Operation`] after simultaneously renaming any [`TypeIdentity`](crate::TypeIdentity)s stored
    /// in its payload, as specified by the provided [`TypeIdentityRenaming`]. Operations whose payload contains no
    /// identity-bearing type metadata return `self.clone()`. An operation that stores shapes, output types, or nested
    /// signature metadata must apply the same renaming as its surrounding program so the payload and atom types remain
    /// consistent.
    #[inline]
    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<T::Identity>) -> Result<Self, TypeError> {
        let _ = renaming;
        Ok(self.clone())
    }

    /// Renders this [`Operation`] as part of an [`Instruction`](crate::Instruction). The default implementation
    /// simply renders [`Operation::name`]. Operations carrying semantic metadata should override this function and
    /// use [`OperationFormatter`] for consistent bracketed and indented formatting. Attached [`Region`](crate::Region)s
    /// are not rendered here. The contextual [`Program`] renderer renders each instruction's attached regions after its
    /// operation.
    ///
    /// # Parameters
    ///
    ///   - `formatter`: [`Formatter`](std::fmt::Formatter) to which the rendered operation must be appended at its
    ///     current position.
    ///   - `indentation`: Indentation of the instruction line containing this operation. Implementations that render
    ///     additional lines must use this value as the base indentation for those continuation lines, preferably by
    ///     delegating their layout to [`OperationFormatter`]. They must not add indentation before the first rendered
    ///     token because the caller has already rendered the instruction line's indentation and prefix.
    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        // The program renderer has already written the indentation and instruction prefix for this line. The
        // `indentation` argument is available to overrides that render continuation lines, but this single-line
        // default only needs to append the operation name.
        let _ = indentation;
        formatter.write_str(self.name())
    }
}

impl<T: Type, O: Operation<T>> Operation<T> for Box<O> {
    #[inline]
    fn name(&self) -> &'static str {
        self.as_ref().name()
    }

    #[inline]
    fn region_names(&self) -> &'static [&'static str] {
        self.as_ref().region_names()
    }

    #[inline]
    fn infer_region_input_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<Option<Vec<T>>>, TypeError> {
        self.as_ref().infer_region_input_types(input_types, region_interfaces)
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        self.as_ref().infer_output_types(input_types, region_interfaces)
    }

    #[inline]
    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        self.as_ref().output_region_provenance(output_index)
    }

    #[inline]
    fn is_zero(&self, output_index: usize) -> bool {
        self.as_ref().is_zero(output_index)
    }

    #[inline]
    fn effects(&self) -> Effects {
        self.as_ref().effects()
    }

    #[inline]
    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<T::Identity>) -> Result<Self, TypeError> {
        Ok(Box::new(self.as_ref().rename_type_identities(renaming)?))
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        self.as_ref().render(formatter, indentation)
    }
}
#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::scalars::Scalar;
    use crate::contexts::Domain;
    use crate::interpretation::{InterpretableOperation, InterpretationDriver};
    use crate::macros::check_count;
    use crate::parameters::Placeholder;
    use crate::programs::{Program, ProgramBuilder, ProgramError};
    use crate::types::DataType;

    use super::*;

    #[derive(Clone, Debug)]
    struct IdentityOperation;

    impl Operation<DataType> for IdentityOperation {
        #[inline]
        fn name(&self) -> &'static str {
            "identity"
        }

        fn infer_output_types(
            &self,
            input_types: &[DataType],
            _region_interfaces: &[RegionInterface<DataType>],
        ) -> Result<Vec<DataType>, TypeError> {
            check_count!("input", input_types, 1, TypeError);
            Ok(vec![input_types[0]])
        }
    }

    impl<C: Domain<Type = DataType, Value = Scalar>> InterpretableOperation<C> for IdentityOperation {
        fn interpret<D: InterpretationDriver<C>>(
            &self,
            _context: &C,
            _driver: &D,
            inputs: &[Scalar],
        ) -> Result<Vec<Scalar>, ProgramError> {
            check_count!("input", inputs, 1, ProgramError);
            Ok(inputs.to_vec())
        }
    }

    #[derive(Clone, Debug)]
    struct InlineMetadataOperation;

    impl Operation<DataType> for InlineMetadataOperation {
        #[inline]
        fn name(&self) -> &'static str {
            "metadata"
        }

        fn infer_output_types(
            &self,
            input_types: &[DataType],
            region_interfaces: &[RegionInterface<DataType>],
        ) -> Result<Vec<DataType>, TypeError> {
            IdentityOperation.infer_output_types(input_types, region_interfaces)
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
        const VALUE: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz";
    }

    impl Operation<DataType> for LongMetadataOperation {
        #[inline]
        fn name(&self) -> &'static str {
            "metadata"
        }

        fn infer_output_types(
            &self,
            input_types: &[DataType],
            region_interfaces: &[RegionInterface<DataType>],
        ) -> Result<Vec<DataType>, TypeError> {
            IdentityOperation.infer_output_types(input_types, region_interfaces)
        }

        fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
            OperationFormatter::new(formatter, indentation, self.name())?
                .bracketed(|operation| operation.field("value", Self::VALUE))
        }
    }

    #[derive(Clone, Debug)]
    struct NestedProgramOperation {
        program: Program<Scalar, IdentityOperation, Scalar, Scalar>,
    }

    impl Operation<DataType> for NestedProgramOperation {
        #[inline]
        fn name(&self) -> &'static str {
            "nested"
        }

        fn infer_output_types(
            &self,
            input_types: &[DataType],
            region_interfaces: &[RegionInterface<DataType>],
        ) -> Result<Vec<DataType>, TypeError> {
            IdentityOperation.infer_output_types(input_types, region_interfaces)
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

    fn identity_program() -> Program<Scalar, IdentityOperation, Scalar, Scalar> {
        let mut builder = ProgramBuilder::<Scalar, IdentityOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(IdentityOperation, Vec::new(), vec![input]).unwrap()[0];
        builder.build::<Scalar, Scalar>(vec![output], Placeholder, Placeholder).unwrap()
    }

    #[test]
    fn default_operation_rendering_uses_the_operation_name() {
        assert_eq!(render_operation(&IdentityOperation), "identity");
    }

    #[test]
    fn operation_inference_and_interpretation_use_concrete_inputs() {
        let operation = IdentityOperation;
        assert_eq!(operation.infer_output_types(&[DataType::F64], &[]), Ok(vec![DataType::F64]));
        assert_eq!(
            operation.infer_output_types(&[], &[]),
            Err(TypeError::invalid("expected 1 input but got 0".to_string())),
        );
        assert_eq!(
            operation
                .interpret(&crate::EagerContext::<Scalar>::new(), &crate::EmptyRegionDriver, &[Scalar::from(3.0)],),
            Ok(vec![Scalar::from(3.0)]),
        );
        assert_eq!(
            operation.interpret(&crate::EagerContext::<Scalar>::new(), &crate::EmptyRegionDriver, &[]),
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
}
