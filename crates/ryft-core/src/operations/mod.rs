use std::collections::BTreeSet;
use std::fmt::Display;

use crate::batching::{ArrayBatch, BatchingTracer};
use crate::broadcasting::Broadcastable;
use crate::compilation::CaptureReference;
use crate::contexts::Context;
use crate::differentiation::{DifferentiationDual, DifferentiationTracer};
use crate::effects::Effects;
use crate::macros::check_count;
use crate::parameters::Parameterized;
use crate::partial::{PartialEvaluationValue, PartialTracer, PartialValue};
use crate::programs::{MaybeZero, Program, ProgramError, Value};
use crate::tracing::Tracer;
use crate::types::{ArrayType, DataType, Type, TypeError, Typed};

// TODO(eaplatanios): Review this file.

/// Elementwise pairwise comparison operations and capability traits.
pub mod compare;

/// Elementwise complex-number construction, conjugation, and part-extraction operations and capability traits.
pub mod complex;

/// Type-driven constant operations and capability traits.
pub mod constants;

/// Higher-order control-flow operations and capability traits.
pub mod control_flow;

/// Debugging operations with observable effects (e.g., printing values from inside programs).
pub mod debugging;

/// Differentiation-control operations and capability traits.
pub mod differentiation;

/// Elementwise exponential, logarithm, and square-root operations and capability traits.
pub mod exponential;

/// Elementwise logical operations and capability traits.
pub mod logical;

/// Array shape and axis manipulation operations and capability traits.
pub mod manipulation;

/// Elementwise arithmetic and trigonometric math operations and capability traits.
pub mod math;

/// Shared marker types for operations with payload-dependent interpretation.
pub mod payloads;

/// Scalar operation types built from the core primitive operation traits.

/// Sharding-related operations (e.g., resharding and propagation hints) and capability traits.
pub mod sharding;

/// Value tagging — attaching a string key to a value in a program (consumed by, e.g., rematerialization policies).
pub mod tag;

// TODO(eaplatanios): We should be importing specific symbols here.
// The fallible `Add`/`Sub`/`Mul`/`Div`/`Neg` capability traits are intentionally not re-exported at this level so
// they do not shadow their `std::ops` counterparts; reach them through `crate::operations::math` instead.
pub use compare::*;
pub use constants::*;
pub use control_flow::*;
pub use debugging::{PRINT_OPERATION_NAME, Print, PrintOperation};
pub use differentiation::*;
pub use logical::*;
pub use manipulation::*;
pub use math::*;
pub use sharding::*;
pub use tag::{MaybeTag, TAG_OPERATION_NAME, Tag, TagOperation};

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
/// It is meant for enums such as [`ScalarOperation`], where every variant wraps one concrete operation payload and
/// the enum should behave exactly like whichever payload it contains. The derived implementation generates:
///
///   - An [`Operation<T>`](Operation) implementation whose [`name`](Operation::name),
///     [`infer_output_types`](Operation::infer_output_types), and [`render`](Operation::render) methods forward to the
///     active variant payload.
///   - An [`InterpretableOperation<V>`](crate::InterpretableOperation) implementation that forwards
///     [`interpret`](crate::InterpretableOperation::interpret) to the active variant payload. Operation-specific eager
///     or staged interpretation semantics still live on the payload implementations; the enum is only a dispatcher.
///   - An [`InterpretableProgramOperation<V, C>`](crate::InterpretableProgramOperation) implementation that
///     interprets nested flat [`Program`]s in the enum's closed operation family. This is the interpretation
///     fixed-point witness used by higher-order payloads such as condition, while, scan, and custom derivative calls.
///     The generated implementation performs a flat program walk and dispatches each instruction to the concrete
///     variant payload directly. That finite dispatch avoids asking Rust to prove the enum's full recursive
///     [`InterpretableOperation`](crate::InterpretableOperation) implementation while defining the program witness.
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
///   - Each payload receives a generated `Payload: InterpretableOperation<V>` bound for the interpretation dispatcher.
///     Payload-specific value or context requirements should live on the payload's own
///     [`InterpretableOperation`](crate::InterpretableOperation) implementation; the enum derivation carries them
///     through this generated payload bound.
///   - `#[ryft(bounds(interpretation(Bound1 + Bound2 + ...)))]` adds extra trait bounds to the generated interpretation
///     implementation value type for both the generated [`InterpretableOperation`](crate::InterpretableOperation)
///     dispatcher and the generated [`InterpretableProgramOperation`](crate::InterpretableProgramOperation) witness.
///     This is useful when recursive higher-order payloads require capabilities on the value being interpreted, while
///     the enum's stored constant or capture type should not be forced to implement those capabilities.
///     For example, an array operation enum that owns condition, while, and scan payloads can write
///     `#[ryft(bounds(interpretation(BooleanLike + Slice + UpdateSlice + Reshape)))]` while keeping its enum parameter
///     declaration at `V: Value<Type = ArrayType>`.
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
///     implementation is generic over a runtime value `V`. The generated
///     [`InterpretableProgramOperation`](crate::InterpretableProgramOperation) implementation requires
///     `V::InterpretationContext` to be a [`Context`] that can lift constants from `C` into `V`. Interpretation-only
///     capabilities should be provided with `#[ryft(bounds(interpretation(...)))]` instead of being placed on the
///     enum's stored constant type.
///   - For enums with two or more `Value<Type = T>` parameters, the first value parameter is treated as both the
///     runtime value type and the nested program constant type for direct program interpretation. Later value
///     parameters remain payload-specific metadata unless the generated program witness needs to instantiate a
///     direct-linear operation family, in which case extra value parameters after the first two are substituted with
///     the first value parameter.
///   - For recursive operation enums, the generated nested-program witness inherits value capabilities from
///     payload-owned [`InterpretableOperation`](crate::InterpretableOperation) implementations and from
///     `#[ryft(bounds(interpretation(...)))]`. If a higher-order payload needs a capability such as [`BooleanLike`] for
///     predicate extraction, prefer the attribute when that capability belongs to interpretation rather than to the
///     enum's stored payload shape.
///   - Bounds provided through `#[ryft(bounds(interpretation(...)))]` are applied to the same generated interpretation
///     value type. For example, one-value-parameter enums apply them to the generated runtime value parameter, while
///     direct linear enums apply them to their first value parameter. When any interpretation bounds are provided, the
///     macro also adds the standard companion requirement `C: Zero<V>` for the generated implementation value type.
///
/// The derivation macro supports the `#[ryft(crate = "...")]` attribute to override the path used to reference Ryft
/// traits and error types from generated code. The default path is `ryft`, so downstream crates that depend on the
/// `ryft` crate normally do not need this attribute. It also supports `#[ryft(bounds(interpretation(...)))]` for the
/// interpretation value bounds described above.
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
pub trait Operation<T: Type> {
    /// Returns the name of this [`Operation`] that is used in diagnostics and when rendering [`Program`]s as strings.
    fn name(&self) -> &'static str;

    /// Infers the output [`Type`]s of this [`Operation`] from the provided input [`Type`]s without executing it.
    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError>;

    /// Returns the observable [`Effect`](crate::Effect) classes of this [`Operation`]. Refer to the documentation of
    /// [`Effects`] and [`Effect`](crate::Effect) for the semantics.
    #[inline]
    fn effects(&self) -> Effects {
        Effects::PURE
    }

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
    fn effects(&self) -> Effects {
        self.as_ref().effects()
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        self.as_ref().render(formatter, indentation)
    }
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
                    message: format!("'{}' input types are not broadcast-compatible", self.name()),
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
///   [`DataType`] and [`ArrayType`]), the Boolean counterpart keeps the same structural metadata (e.g., shape, layout,
///   and sharding) but uses a Boolean element data type.
/// - **Predicate-Consuming Operations (e.g., [`ConditionOperation`] and [`WhileOperation`]):** Call
///   [`boolean`](Self::boolean) on *values* to extract the concrete scalar Rust Boolean that drives branching
///   or selection.
///
/// For values, [`as_boolean`](Self::as_boolean) reinterprets the carried payload as a Boolean value: zero maps to
/// `false` and any non-zero payload maps to `true`. Values that carry no concrete payload (e.g., staged tracers and
/// [`CaptureReference`]s) cannot reinterpret anything and return themselves unchanged. Similarly,
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

impl<C: Context> BooleanLike for Tracer<C> {
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

impl BooleanLike for CaptureReference<ArrayType> {
    #[inline]
    fn as_boolean(&self) -> Self {
        // Returns this [`CaptureReference`] unchanged. A captured constant is a reference into a side table,
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

// A partial-evaluation value's Boolean view uses its known payload's: a known value reinterprets (and decodes) the
// carried known-side value, so branching on a known value in a closure succeeds exactly when the known-side inner
// context is eager, while an unknown value names a residual program variable that carries no concrete payload and so
// returns itself unchanged from `as_boolean` and errors from `boolean`. This is what lets host control flow branch on
// known values while partial evaluation is in progress.
impl<C: Context<Value: BooleanLike, Type: BooleanLike>> BooleanLike for PartialTracer<C> {
    #[inline]
    fn as_boolean(&self) -> Self {
        // Unknown and poisoned values carry no concrete payload to reinterpret and return themselves unchanged.
        match self.value() {
            Ok(value) => match value.value() {
                PartialValue::Known(known) => {
                    PartialTracer::new(self.context().clone(), PartialEvaluationValue::known(known.as_boolean()))
                }
                PartialValue::Unknown(_) => self.clone(),
            },
            Err(_) => self.clone(),
        }
    }

    #[inline]
    fn boolean(&self) -> Result<bool, ProgramError> {
        // A poisoned value surfaces its deferred error here, since branching on it cannot proceed anyway.
        match self.value()?.value() {
            PartialValue::Known(known) => known.boolean(),
            PartialValue::Unknown(_) => Err(ProgramError::Concretization {
                message: "cannot extract a concrete boolean from an unknown partial-evaluation value".to_string(),
            }),
        }
    }
}

// A batch-carrying value's Boolean view uses its packed value's Boolean view. Branching on it via `boolean()` succeeds
// only for a *replicated* value whose packed value is concrete.  A batched value has one Boolean per item and cannot
// drive a single branch, and a staged value carries no concrete payload.
impl<C: Context<Type = ArrayType, Value: BooleanLike>> BooleanLike for BatchingTracer<C> {
    #[inline]
    fn as_boolean(&self) -> Self {
        let r#type = self.batch().r#type().as_boolean();
        let batch = ArrayBatch::new(r#type, self.batch().value().as_boolean(), self.batch().batch_axis()).unwrap();
        BatchingTracer::new(self.context().clone(), batch)
    }

    #[inline]
    fn boolean(&self) -> Result<bool, ProgramError> {
        if !self.batch().batch_axis().is_replicated() {
            return Err(ProgramError::Concretization {
                message: "cannot extract a concrete boolean from a batched value".to_string(),
            });
        }
        self.batch().value().boolean()
    }
}

// TODO(eaplatanios): Review this implementation.
// A dual's Boolean view uses its primal's: [`as_boolean`](BooleanLike::as_boolean) reinterprets the primal with a
// structural zero tangent, and [`boolean`](BooleanLike::boolean) decodes the primal — so branching on a dual in a
// closure succeeds exactly when the primal is a concrete (eager) value and errors when it is a staged tracer.
impl<C: Context<Value: BooleanLike>> BooleanLike for DifferentiationTracer<C> {
    #[inline]
    fn as_boolean(&self) -> Self {
        let primal = self.primal().as_boolean();
        let tangent = MaybeZero::Zero(primal.r#type().into_owned());
        Self::new(DifferentiationDual::new(primal, tangent), self.context().clone())
    }

    #[inline]
    fn boolean(&self) -> Result<bool, ProgramError> {
        self.primal().boolean()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::scalars::Scalar;
    use crate::interpretation::InterpretableOperation;
    use crate::macros::check_count;
    use crate::parameters::Placeholder;
    use crate::programs::{Program, ProgramBuilder, ProgramError};
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

    impl<C> InterpretableOperation<Scalar, C> for IdentityOperation {
        fn interpret(&self, _context: &C, inputs: &[Scalar]) -> Result<Vec<Scalar>, ProgramError> {
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
        program: Program<Scalar, IdentityOperation, Scalar, Scalar>,
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

    fn identity_program() -> Program<Scalar, IdentityOperation, Scalar, Scalar> {
        let mut builder = ProgramBuilder::<Scalar, IdentityOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(IdentityOperation, vec![input]).unwrap()[0];
        builder.build::<Scalar, Scalar>(vec![output], Placeholder, Placeholder).unwrap()
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
        assert_eq!(
            operation.interpret(&crate::EagerContext::<Scalar>::new(), &[Scalar::from(3.0)]),
            Ok(vec![Scalar::from(3.0)])
        );
        assert_eq!(
            operation.interpret(&crate::EagerContext::<Scalar>::new(), &[]),
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
            Err(TypeError { message: "'elementwise_test' input types are not broadcast-compatible".to_string() }),
        );
    }
}
