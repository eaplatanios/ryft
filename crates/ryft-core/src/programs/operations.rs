use crate::parameters::Parameterized;
use crate::programs::effects::Effects;
use crate::programs::identities::TypeIdentityRenaming;
use crate::programs::programs::Program;
use crate::programs::regions::{OutputRegionProvenance, RegionInterface, RegionRole, RegionSlot};
use crate::programs::types::{Type, TypeError};
use crate::programs::{ProgramError, Value};

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
    pub fn field<V: std::fmt::Display>(&mut self, name: &str, value: V) -> std::fmt::Result {
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
    pub fn program<V: Value, O: Operation<Type = V::Type>, Input: Parameterized<V>, Output: Parameterized<V>>(
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
/// It is meant for enums such as [`ArrayOperation`](crate::ArrayOperation), where every variant wraps one concrete
/// operation payload and the enum should behave exactly like whichever payload it contains. Operation-specific
/// semantics always live on the payload implementations, so the derived enum is only an adapter and a dispatcher.
/// The sections below are the canonical reference for the derivation macro, including information on the enum-level
/// attributes, how the operation and constant types are determined, the variant-class grammar, the generated
/// implementations and their contracts, and the macro's requirements.
///
/// ## Enum-Level Attributes
///
/// Every enum-level attribute is optional and may appear at most once. These are the supported enum-level attributes:
///
/// | Attribute                                  | Default  | Role                                               |
/// | ------------------------------------------ | -------- | -------------------------------------------------- |
/// | `#[ryft(crate = "...")]`                   | `ryft`   | Path through which generated code names Ryft items |
/// | `#[ryft(type = T, constant = V)]`          | inferred | Primary operation type and stored constant type    |
/// | `#[ryft(members(U [, structural(S)]...))]` | none     | Member universes the operation family declares     |
/// | `#[ryft(dispatch(...))]`                   | none     | Optional transform dispatchers to generate         |
///
/// ### Crate Path
///
/// `#[ryft(crate = "...")]` overrides the path used to reference Ryft traits, helpers, macros, and error types from
/// generated code. The default path is `ryft`, so downstream crates that depend on the `ryft` crate normally do not
/// need this attribute. Generated code names only items exported at the root of `ryft-core`, so any path that
/// re-exports that root works.
///
/// ## Operation And Constant Types
///
/// The operation type `T` is selected as follows:
///
///   - An enum whose stored values belong to a member type rather than its primary type may declare both contracts
///     explicitly with `#[ryft(type = T, constant = C)]`. Here, `T` is the enum's primary operation type and `C` is
///     the concrete value type stored as constants in programs using the enum. Both attributes must be supplied
///     together. For example, a composite operation enum parameterized by an array-member value `A` can declare
///     `#[ryft(type = CompositeType, constant = CompositeValue<A>)]` without adding a phantom composite-value generic.
///     Composite families declare them explicitly because their stored constants live in a member universe, so no
///     single `Value<Type = T>` bound mentions the composite type the family's instructions actually flow.
///   - If the enum has exactly one distinct generic bound of the form `Value<Type = T>`, the derivation infers `T`
///     from that bound. For example, `enum BackendOperation<V: Value<Type = ArrayType>>` derives `Operation<Type =
///     ArrayType>`. Multiple generic parameters may use the same `T`. For example, `V: Value<Type = ArrayType>,
///     C: Value<Type = ArrayType>` still infers `ArrayType`.
///   - If no `Value<Type = T>` bound is present, or if multiple distinct operation types are present (e.g., `Value<Type
///     = DataType>` and `Value<Type = ArrayType>`), the derivation macro cannot choose an operation type and reports a
///     compilation error. In those cases, the caller must split the enum by operation type or implement [`Operation`]
///     manually.
///
/// The value types used for interpretation are inferred from the enum's `Value<Type = T>` generic parameters:
///
///   - For an enum with an explicit `#[ryft(type = T, constant = C)]` declaration, `C` is the nested program's stored
///     constant type. The derived interpretation implementation remains generic over the context's runtime value type
///     and requires the context to lift `C` into that runtime value type.
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
/// ### Member Universes
///
/// An operation family may declare its member universes once, at the enum level, with `#[ryft(members(U [,
/// structural(S)]...))]`. Each entry is either a bare [`Type`] naming a computational member or `structural(Type)`
/// naming a structural member, member types may not repeat, and an empty list is rejected. For example, the dynamic
/// array family declares `#[ryft(members(ArrayType, structural(DimensionType)))]`. Declaring the members has three
/// effects: (i) a `mixed` marker may omit its member type (see [Variant Classes](#variant-classes)), (ii) every member
/// marker's type is checked against the declaration, and (iii) the generated structural mixed forward-mode arm knows
/// every computational universe an output may belong to.
///
/// ### Transform Dispatchers
///
/// `#[ryft(dispatch(...))]` selects the optional transform dispatchers, in any order:
///
/// | Token             | Generated Dispatcher                                        |
/// | ----------------- | ----------------------------------------------------------- |
/// | `batching`        | [`BatchableOperation`](crate::BatchableOperation)           |
/// | `differentiation` | [`DifferentiableOperation`](crate::DifferentiableOperation) |
/// | `transposition`   | [`TransposableOperation`](crate::TransposableOperation)     |
///
/// An empty list, an unknown token, and a repeated token are all compilation errors. Interpretation and partial
/// evaluation implementations are always generated and therefore never appear in `dispatch(...)`. The selected
/// dispatchers generate the following per-variant arms:
///
///   - `batching` delegates native variants to the payload's own [`BatchableOperation`](crate::BatchableOperation)
///     implementation, batches projected variants in either role through
///     [`batch_projected_operation`](crate::batch_projected_operation) under the policy projection named by
///     [`BatchingPolicyProjection`](crate::BatchingPolicyProjection), and dispatches mixed variants in either role to
///     the payload's parent-universe [`MemberBatchableOperation`](crate::MemberBatchableOperation) implementation. A
///     family whose constant type is inferred stays generic over its array batching mode, while an explicitly declared
///     family uses the canonical policy named by [`BatchableType`](crate::BatchableType).
///   - `differentiation` delegates native variants to the payload's own forward-mode rule and computational member
///     variants to [`MemberDifferentiableOperation`](crate::MemberDifferentiableOperation), and generates the
///     structural arms described under [Variant Classes](#variant-classes) directly.
///   - `transposition` delegates native variants to the payload's own rule, mixed variants in either role to
///     [`transpose_mixed_operation`](crate::transpose_mixed_operation), and computational projected variants to
///     [`transpose_projected_operation`](crate::transpose_projected_operation), while a structural projected variant
///     returns zero cotangents directly.
///
/// ## Variant Classes
///
/// An unmarked variant is native to the enum's primary type `T` and delegates directly to a payload implementing
/// `Operation<Type = T>`. A variant whose payload instead lives in a member universe `U` declares a _member class_
/// using the single grammar `class(U [, structural])`, where the class name selects the **boundary shape** relating
/// the payload's type boundary to `T` and the optional `structural` token selects the transform role, stating whether
/// transforms delegate to the payload’s computational rules or treat the payload as structural bookkeeping. The role
/// defaults to computational, so the two boundaries and two roles give four member classes:
///
/// | Boundary \ Role | Computational           | Structural                          |
/// | --------------- | ----------------------- | ----------------------------------- |
/// | Projected       | `#[ryft(projected(U))]` | `#[ryft(projected(U, structural))]` |
/// | Mixed           | `#[ryft(mixed(U))]`     | `#[ryft(mixed(U, structural))]`     |
///
///   - A **projected** boundary means that every operand and result of the instruction belongs to `U`, so inference and
///     eager execution project the composite boundary down to `U` and lift the results back into `T`. The member type
///     may occur only once per boundary because [`OperationProjection<U>`] names one canonical projected family.
///   - A **mixed** boundary means the instruction crosses member kinds: the payload keeps its native `Operation<Type =
///     U>` contract while its parent instruction also consumes operands belonging to other members of `T` (e.g., the
///     first-class dimensions that supply a dynamic result geometry) and may produce results in those members too. The
///     mixed machinery classifies each operand and each result by the member universe it belongs to rather than by its
///     position, so the two operand kinds may be arranged in any order. The payload supplies that boundary through
///     [`MemberOperation`] and, for interpretation, through
///     [`MemberInterpretableOperation`](crate::MemberInterpretableOperation).
///     Several mixed variants may share one member type.
///   - A **computational** role means that transforms recurse into the payload's own rules. A projected payload uses
///     the member family's ordinary rules, while a computational mixed payload states its parent-universe derivative
///     through [`MemberDifferentiableOperation`](crate::MemberDifferentiableOperation) and its parent-universe batching
///     rule through [`MemberBatchableOperation`](crate::MemberBatchableOperation). Transposition never needs a mixed
///     rule because [`transpose_mixed_operation`](crate::transpose_mixed_operation) delegates the instruction's
///     `U`-typed data operands, in operand order, to the payload's ordinary homogeneous
///     [`TransposableOperation`](crate::TransposableOperation) rule and gives every other
///     operand a structural zero cotangent.
///   - A **structural** role declares that the payload is bookkeeping. There is nothing to differentiate (i.e., the
///     type has a zero differential space) and nothing to batch per item, so batched inputs must be replicated. A
///     structural projected member reaches that behavior through its member family's rules and its projected batching
///     policy, and its generated forward-mode and transposition arms stage a zero tangent and return zero cotangents
///     directly. A structural mixed member instead has its forward-mode rule generated as the payload's primal plus
///     one zero tangent per declared parent output, staged over the same operands so that the runtime geometry those
///     operands carry stays available to both. Each output's tangent is constructed in the computational member
///     universe that output belongs to, discovered by projecting the declared output type across the family's
///     computational members (see [Member Universes](#member-universes)). An output outside all of them, such as a
///     structural member output, has a zero differential space and receives a symbolic zero tangent instead of a
///     staged instruction. Batching is the one transform the structural role does not cover for a mixed boundary,
///     because a mixed signature cannot be projected into one member kind. A structural mixed payload therefore still
///     implements [`MemberBatchableOperation`](crate::MemberBatchableOperation), normally as a replicated-operands-only
///     rule.
///
/// ### Defaulting The Member Type
///
/// Only the `mixed` class can default its member type. A bare `#[ryft(mixed)]` or `#[ryft(mixed(structural))]` takes
/// its data universe from the family's single computational member, so in the abbreviated role form the sole argument
/// is a role and not a member type. The derivation reports an error when the family declares no
/// [member universes](#member-universes) at all and when it declares several computational members, because neither
/// case has a unique default; both ask for the explicit `mixed(U)` form. A named member type that the family does not
/// declare is also an error. The `projected` class always names its member type, so `#[ryft(projected)]` does not
/// parse.
///
/// ### Suppressing The Owned Conversion
///
/// A variant may additionally use `#[ryft(skip_from)]` to suppress its generated owned `From<Payload>` conversion.
/// Its borrowed `TryFrom<&Enum>` projection is still generated. Use this only when the enum provides a handwritten
/// `From<Payload>` implementation whose result is not always that variant (e.g., because conversion promotes a member
/// operation into a composite carrier or selects a static versus dynamic representation from payload metadata). Without
/// the marker, that handwritten implementation would conflict with the generated implementation, and the generated
/// direct wrapper would bypass the required normalization.
///
/// ## Generated Implementations
///
/// The derived implementation generates:
///
///   - An [`Operation<Type = T>`](Operation) implementation whose semantic and rendering methods forward to the active
///     variant payload.
///   - An [`InterpretableOperation<C>`](crate::InterpretableOperation) implementation that forwards native variants
///     directly and projects member variants into their native eager value family. Operation-specific eager semantics
///     still live on the payload implementations; the enum is only an adapter and dispatcher.
///   - A [`PartiallyEvaluatableOperation<C>`](crate::PartiallyEvaluatableOperation) implementation that forwards native
///     variants to their payload rules. Member variants use the enclosing operation's canonical fold-or-residualize
///     path, avoiding a second projected partial-value protocol.
///   - A canonical [`OperationProjection<U>`] implementation for every projected member variant, in either role, naming
///     that variant's payload family as the enum's projection into `U`. Native and mixed variants do not define a
///     homogeneous operation-family projection.
///   - [`BatchableOperation`](crate::BatchableOperation), [`DifferentiableOperation`](crate::DifferentiableOperation),
///     and [`TransposableOperation`](crate::TransposableOperation) dispatchers selected through
///     [`#[ryft(dispatch(...))]`](#transform-dispatchers).
///   - A [`Display`](std::fmt::Display) implementation that renders through [`Operation::render`] with zero
///     indentation, so that the enum display matches the canonical program rendering format.
///   - `From<Payload> for Enum` conversions for concrete payload variants.
///   - Borrowed `TryFrom<&Enum> for &Payload` conversions for concrete payload variants, including boxed payloads.
///     These conversions fail with `Error = TypeError`, reporting `"cannot project operation '<name>' into a '<payload
///     type>' payload"`, where `<name>` is the stored operation's [`Operation::name`] and `<payload type>` is the
///     statically expected payload type.
///
/// ## Requirements And Limitations
///
/// The macro has the following requirements:
///
///   - The derivation macro input must be an enum. Structs and unions are not supported.
///   - Every variant must be a tuple variant with exactly one payload field.
///   - Member-class markers belong on the variant. Field-level `#[ryft(...)]` attributes are not supported.
///   - A payload may be stored directly as `Payload` or indirectly as `Box<Payload>`. Boxed variants still delegate to
///     `Payload`, and their generated `From<Payload>` implementation boxes the payload for the caller.
///   - Every native payload must implement [`Operation<Type = T>`](Operation), where `T` is the enum's primary type.
///     A member payload must instead implement `Operation<Type = U>` for the member type declared by its variant
///     marker, or defaulted from `#[ryft(members(...))]`. The enclosing type and value families must provide the
///     corresponding projection vocabulary.
///   - A projected-boundary member type may occur only once because [`OperationProjection<U>`] has one canonical
///     projected operation family. Several operations in the same member universe should first be collected into that
///     family rather than declared as separate outer variants. Mixed-boundary variants do not claim that projection,
///     so they may repeat a member type freely.
///   - Bare generic payload variants such as `Extension(Extension)` do not receive `From` or `TryFrom` conversions.
///     Generating those conversions would overlap with concrete variant conversions when the generic parameter is
///     instantiated as one of the concrete payload types. The operation forwarding implementation still supports the
///     generic payload by adding an `Extension: Operation<Type = T>` bound.
///   - Generated semantic dispatchers place their required bounds on the participating payloads. Payload-specific
///     value and context requirements belong on the payload's own semantic-trait implementation; the enum does not
///     duplicate them.
///   - Batching, differentiation, and transposition require selecting the corresponding dispatcher. Interpretation
///     and partial evaluation are always generated and therefore do not appear in the `dispatch(...)` attribute.
///
/// ## Examples
///
/// A homogeneous operation family infers its operation and constant types from the enum's `Value<Type = T>` generic
/// parameter and needs no enum-level attributes:
///
/// ```rust
/// # use ryft_core as ryft;
/// # use ryft_core::arrays::{ArrayType, DataType};
/// # use ryft_core::{Array, ConstantOperation, Operation, Value, ZeroOperation};
/// # use ryft_macros::Operation;
///
/// #[derive(Clone, Debug, Operation)]
/// enum BackendOperation<V: Value<Type = ArrayType>> {
///     Zero(ZeroOperation<ArrayType>),
///     Constant(ConstantOperation<V>),
/// }
///
/// let operation = BackendOperation::<Array>::from(ZeroOperation::new(ArrayType::scalar(DataType::F32)));
/// assert_eq!(operation.name(), "zero");
/// ```
///
/// A composite operation family declares its operation and constant types explicitly, declares its member universes
/// once with `members(...)`, and marks each non-native variant with its class. Unmarked variants are native to the
/// composite type, `projected(U)` variants embed a whole member family behind a projection, and `mixed` variants
/// default their data universe to the family's unique computational member:
///
/// ```rust
/// # use ryft_core as ryft;
/// # use ryft_core::arrays::{ArrayIrType, ArrayType, DataType, Dimension, DimensionType, Shape};
/// # use ryft_core::{Operation, Value, ZeroOperation};
/// # use ryft_core::arrays::{ArrayIrValue, DimensionOperation, DimensionValue};
/// # use ryft_core::backends::{Array, ArrayOperation};
/// # use ryft_core::operations::dimensions::DimensionSizeOperation;
/// # use ryft_macros::Operation;
///
/// #[derive(Clone, Debug, Operation)]
/// #[ryft(type = ArrayIrType, constant = ArrayIrValue<A>)]
/// #[ryft(members(ArrayType, structural(DimensionType)))]
/// enum CompositeOperation<A: Value<Type = ArrayType>> {
///     /// Mixed structural constructor whose operands are the stored type's dynamic extents, its data universe
///     /// defaults to `ArrayType` (i.e., the unique computational member), and its transforms are fully generated.
///     #[ryft(mixed(structural))]
///     Zero(ZeroOperation<ArrayType>),
///
///     /// Projected computational member family that includes every homogeneous array operation, behind a projection.
///     #[ryft(projected(ArrayType))]
///     Array(ArrayOperation<A>),
///
///     /// Projected structural member family representing dimension bookkeeping with zero differential space.
///     #[ryft(projected(DimensionType, structural))]
///     Dimension(DimensionOperation<DimensionValue>),
///
///     /// Native mixed-signature payload that consumes an array and produces a first-class dimension.
///     DimensionSize(DimensionSizeOperation),
/// }
///
/// let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));
/// let operation = CompositeOperation::<Array>::from(DimensionSizeOperation::new(&input_type, 0).unwrap());
/// assert_eq!(operation.name(), "dimension_size");
/// ```
pub trait Operation: Clone {
    /// Canonical [`Type`] universe of this [`Operation`]. Every payload has exactly one operation contract, so
    /// this associated type is the single source of truth for the operation's type universe. [`Program`]s require
    /// `O: Operation<Type = V::Type>`, which makes the value/operation type agreement compiler-enforced instead of
    /// depending on a separately supplied type argument.
    type Type: Type;

    /// Returns the name of this [`Operation`] that is used in diagnostics and when rendering [`Program`]s as strings.
    fn name(&self) -> &'static str;

    /// Returns declarations for this [`Operation`]'s attached-[`Region`](crate::Region) slots, in operation-defined
    /// order. Each [`RegionSlot`] supplies the stable name used by diagnostics and rendering together with the
    /// [`RegionRole`] that determines whether the region may execute during ordinary interpretation. The declared
    /// region count must match the number of regions attached to every [`Instruction`](crate::Instruction).
    /// [`ProgramBuilder`](crate::ProgramBuilder)s validate this both when the instruction is added and when the final
    /// [`Program`] is built. The default declares no region slots, which is correct for region-free operations.
    #[inline]
    fn region_slots(&self) -> &'static [RegionSlot] {
        &[]
    }

    /// Returns the declared [`RegionRole`] for the attached region at `index` or [`None`] when `index` is out of range.
    #[inline]
    fn region_role(&self, index: usize) -> Option<RegionRole> {
        self.region_slots().get(index).map(|slot| slot.role)
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
        input_types: &[Self::Type],
        region_interfaces: &[RegionInterface<Self::Type>],
    ) -> Result<Vec<Option<Vec<Self::Type>>>, TypeError> {
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
    ///     [`Operation`]-defined [`region_slots`](Self::region_slots) order. Region-free operations receive an
    ///     empty slice and ignore it.
    fn infer_output_types(
        &self,
        input_types: &[Self::Type],
        region_interfaces: &[RegionInterface<Self::Type>],
    ) -> Result<Vec<Self::Type>, TypeError>;

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
    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<Self::Type as Type>::Identity>,
    ) -> Result<Self, TypeError> {
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

impl<O: Operation> Operation for Box<O> {
    type Type = O::Type;

    #[inline]
    fn name(&self) -> &'static str {
        self.as_ref().name()
    }

    #[inline]
    fn region_slots(&self) -> &'static [RegionSlot] {
        self.as_ref().region_slots()
    }

    #[inline]
    fn infer_region_input_types(
        &self,
        input_types: &[Self::Type],
        region_interfaces: &[RegionInterface<Self::Type>],
    ) -> Result<Vec<Option<Vec<Self::Type>>>, TypeError> {
        self.as_ref().infer_region_input_types(input_types, region_interfaces)
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[Self::Type],
        region_interfaces: &[RegionInterface<Self::Type>],
    ) -> Result<Vec<Self::Type>, TypeError> {
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
    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<Self::Type as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        Ok(Box::new(self.as_ref().rename_type_identities(renaming)?))
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        self.as_ref().render(formatter, indentation)
    }
}

/// Names the `T`-typed member [`Operation`] family embedded in a composite [`Operation`] family. [`Program`]s that
/// store several kinds of values in one composite [`Type`] still keep most of their operations narrow. For example,
/// in a program mixing arrays with first-class runtime dimensions, array-only operations implement `Operation<Type =
/// ArrayType>` and dimension-only operations implement `Operation<Type = DimensionType>`, while the composite operation
/// family merely stores and dispatches them, providing one [`From`] implementation per member family to lift member
/// operations into it. Those [`From`] implementations cannot, however, answer the reverse question that generic code
/// needs: *given* the composite family and a member type `T`, which family is *the* `T`-typed member? [`From`] is
/// many-to-one (a composite family is [`From`] many payload types), so the answer must be a type-level function from
/// `(composite family, T)` to the member family. The [`Projected`](Self::Projected) associated type is that function.
/// This trait adds no methods as lifting a member operation into the composite family remains the [`From`] trait's job.
///
/// A composite operation family implements this trait once per member kind, pointing at the member family that its
/// corresponding [`From`] implementation lifts. For example:
///
/// ```rust,ignore
/// impl<A: Value<Type = ArrayType>> OperationProjection<ArrayType> for ArrayIrOperation<A> {
///     type Projected = ArrayOperation<A>;
/// }
///
/// impl<A: Value<Type = ArrayType>> OperationProjection<DimensionType> for ArrayIrOperation<A> {
///     type Projected = DimensionOperation<DimensionValue>;
/// }
/// ```
///
/// The consumers are the projection adapters that must *name* the member family rather than merely
/// convert into the composite one: [`ProjectedContext`](crate::ProjectedContext) defines its member-typed
/// [`Domain::Operation`](crate::Domain::Operation) as `<C::Operation as OperationProjection<T>>::Projected`
/// so that member-typed operations bind directly through the composite parent context, and
/// [`ProjectedValue`](crate::ProjectedValue)'s blanket [`Value`] implementation derives its dispatch domains
/// the same way. Code that only needs to convert a known operation into the composite family should keep writing plain
/// `Operation: From<X>` bounds instead of this trait. Genuinely mixed operations, whose signatures cross member kinds,
/// belong to no member family and are deliberately not projectable; dispatchers that inspect a composite operation must
/// classify their own variants explicitly.
pub trait OperationProjection<T: Type>: From<Self::Projected> {
    /// The `T`-typed member [`Operation`] family embedded in this composite family. This is the operation-side
    /// counterpart of [`ValueProjection::Projected`](crate::ValueProjection::Projected). Lifting a member operation
    /// into the composite family goes through the [`From`] super-trait, and trait coherence guarantees that each
    /// composite family names at most one member family per member type `T`.
    type Projected: Operation<Type = T>;
}

/// Selects and constructs the concrete [`Operation`] staged by one value-level capability for program type `T`.
///
/// Capabilities such as [`Mul`](crate::Mul) are declared once through the elementwise capability macro and are
/// blanket-implemented for every [`Value`]. The same call, for example `left.mul(&right)`, must therefore stage
/// a different concrete operation depending on the operands' type family (e.g., multiplying arrays stages the
/// stateless [`MulOperation`](crate::MulOperation) itself, while multiplying first-class dimensions must stage a
/// [`DimensionMulOperation`](crate::DimensionMulOperation) whose payload is computed from the operand types).
/// This trait is that selection point. The generated capability implementation calls
/// `<Marker as OperationProvider<V::Type>>::provide(&[..input types..])` and binds the returned operation
/// with the declared operation marker acting as its own family's provider.
///
/// Two implementation levels cover every case:
///
///   - **Self-Provision (Blanket):** A stateless operation that implements [`Operation<Type = T>`](Operation) and
///     [`Default`] provides itself for `T`, so ordinary homogeneous operation families need no code at all.
///   - **Per-Type-Family Override:** A type family whose concrete operation carries type-derived metadata implements
///     this trait for the marker directly (e.g., `impl OperationProvider<DimensionType> for MulOperation`),
///     constructing the family-specific operation from the input types. Such an override is coherent with the blanket
///     implementation exactly because the marker does not implement `Operation<Type = T>` for that type family, which
///     is also precisely the situation that requires providing a different concrete operation.
///
/// The contract is deliberately narrow. `input_types` contains exactly the operation's input type descriptors in
/// operand order, callers (i.e., the generated capability implementations) always pass borrowed stack arrays, and
/// provider implementations validate the arity they support. Selection is based strictly on types, never on runtime
/// values. This is *not* a universal operation factory: mixed operations whose signatures cross member kinds,
/// region-carrying operations, and operations requiring explicit user parameters keep their ordinary constructors.
pub trait OperationProvider<T: Type> {
    /// Concrete [`Operation`] type provided for program [`Type`] `T`.
    type Operation: Operation<Type = T>;

    /// Selects and constructs the [`Operation`] from its ordered input [`Type`]s.
    fn provide(input_types: &[&T]) -> Result<Self::Operation, ProgramError>;
}

impl<O: Default + Operation> OperationProvider<O::Type> for O {
    type Operation = O;

    #[inline]
    fn provide(_input_types: &[&O::Type]) -> Result<Self::Operation, ProgramError> {
        Ok(Self::default())
    }
}

/// Parent-universe [`Operation`] contract for a payload whose native operation type does not describe
/// the complete instruction boundary in that parent universe. This is the base-operation counterpart
/// of [`MemberBatchableOperation`](crate::MemberBatchableOperation) and
/// [`MemberDifferentiableOperation`](crate::MemberDifferentiableOperation).
///
/// Most [`Operation`] enum variants do not need this capability. Composite-native payloads already
/// implement `Operation<Type = U>`, while homogeneous member payloads use projected boundary helpers like
/// [`infer_projected_operation_region_input_types`] and [`infer_projected_operation_output_types`]. A mixed member
/// needs this trait only when it deliberately retains a native payload type `T` but its enclosing instruction has a
/// different or mixed `U`-typed signature. Dynamic array constructors and shape-changing collectives are examples.
/// Their payloads remain canonical array operations, while their [`Instruction`](crate::Instruction)s additionally
/// consume first-class dimension inputs/operands.
///
/// Implementations own only the boundary-dependent parts of [`Operation`]. Name, [`Region`](crate::Region) slots,
/// provenance, structural zero classification, effects, and rendering remain properties of the native payload and are
/// delegated directly by the enclosing operation family.
pub trait MemberOperation<U: Type>: Operation {
    /// Infers attached-[`Region`](crate::Region) input [`Type`]s for this payload's instruction in parent universe `U`.
    fn infer_parent_region_input_types(
        &self,
        input_types: &[U],
        region_interfaces: &[RegionInterface<U>],
    ) -> Result<Vec<Option<Vec<U>>>, TypeError>;

    /// Infers output [`Type`]s for this payload's instruction in parent universe `U`.
    fn infer_parent_output_types(
        &self,
        input_types: &[U],
        region_interfaces: &[RegionInterface<U>],
    ) -> Result<Vec<U>, TypeError>;

    /// Renames parent-universe [`TypeIdentity`](crate::TypeIdentity)s referenced by this payload.
    fn rename_parent_type_identities(&self, renaming: &TypeIdentityRenaming<U::Identity>) -> Result<Self, TypeError>;
}

/// Infers the region input types of a member [`Operation`] through a composite type boundary. Every composite input
/// and attached-region [`RegionInterface`] is first projected to `T`; the inferred member types are then lifted back
/// into `U`. Projection fails with the composite type's canonical wrong-member [`TypeError`].
///
/// This function supports code generated by `#[derive(Operation)]` for `#[ryft(projected(T))]` variants in either
/// transform role. Operation implementations normally call their payload's [`Operation::infer_region_input_types`]
/// method directly.
///
/// # Parameters
///
///   - `operation`: Member operation whose native type is `T`.
///   - `input_types`: Composite input types supplied to the enclosing operation.
///   - `region_interfaces`: Composite interfaces of regions attached to the enclosing operation.
pub fn infer_projected_operation_region_input_types<T: Type, I: Type + From<T>, O: Operation<Type = T>>(
    operation: &O,
    input_types: &[I],
    region_interfaces: &[RegionInterface<I>],
) -> Result<Vec<Option<Vec<I>>>, TypeError>
where
    for<'t> &'t T: TryFrom<&'t I, Error = TypeError>,
{
    let (input_types, region_interfaces) = project_operation_boundary(input_types, region_interfaces)?;
    Ok(operation
        .infer_region_input_types(&input_types, &region_interfaces)?
        .into_iter()
        .map(|types| types.map(|types| types.into_iter().map(I::from).collect()))
        .collect())
}

/// Infers the output types of a member [`Operation`] through a composite type boundary. Every composite input and
/// attached-region [`RegionInterface`] is first projected to `T`; the inferred member outputs are then lifted back
/// into `U`. Projection fails with the composite type's canonical wrong-member [`TypeError`].
///
/// This function supports code generated by `#[derive(Operation)]` for `#[ryft(projected(T))]` variants in either
/// transform role. Operation implementations normally call their payload's [`Operation::infer_output_types`] method
/// directly.
///
/// # Parameters
///
///   - `operation`: Member operation whose native type is `T`.
///   - `input_types`: Composite input types supplied to the enclosing operation.
///   - `region_interfaces`: Composite interfaces of regions attached to the enclosing operation.
pub fn infer_projected_operation_output_types<T: Type, I: Type + From<T>, O: Operation<Type = T>>(
    operation: &O,
    input_types: &[I],
    region_interfaces: &[RegionInterface<I>],
) -> Result<Vec<I>, TypeError>
where
    for<'t> &'t T: TryFrom<&'t I, Error = TypeError>,
{
    let (input_types, region_interfaces) = project_operation_boundary(input_types, region_interfaces)?;
    Ok(operation.infer_output_types(&input_types, &region_interfaces)?.into_iter().map(I::from).collect())
}

/// Projects one member [`Operation`]'s complete inference boundary from the enclosing composite type `U` to its native
/// member type `T`. This includes every ordinary input type and both the input and output types of every attached
/// [`RegionInterface`]. Region effects are structural metadata rather than member-typed values, so they are preserved
/// unchanged.
///
/// Projection is atomic: if any boundary type belongs to another member kind, this function returns that conversion's
/// [`TypeError`] before the member operation's inference rule runs. The returned collections are owned because the
/// native inference APIs require `&[T]` and `RegionInterface<T>`, rather than borrowed member views whose containing
/// values remain typed as `U`.
///
/// # Parameters
///
///   - `input_types`: Composite types of the member operation's ordinary operands.
///   - `region_interfaces`: Composite input/output type contracts and effects of its attached regions.
fn project_operation_boundary<T: Type, U: Type>(
    input_types: &[U],
    region_interfaces: &[RegionInterface<U>],
) -> Result<(Vec<T>, Vec<RegionInterface<T>>), TypeError>
where
    for<'t> &'t T: TryFrom<&'t U, Error = TypeError>,
{
    Ok((
        input_types.iter().map(|r#type| <&T>::try_from(r#type).cloned()).collect::<Result<_, _>>()?,
        region_interfaces
            .iter()
            .map(|interface| {
                Ok(RegionInterface::new(
                    interface
                        .input_types()
                        .iter()
                        .map(|r#type| <&T>::try_from(r#type).cloned())
                        .collect::<Result<_, _>>()?,
                    interface
                        .output_types()
                        .iter()
                        .map(|r#type| <&T>::try_from(r#type).cloned())
                        .collect::<Result<_, _>>()?,
                    interface.effects(),
                ))
            })
            .collect::<Result<_, TypeError>>()?,
    ))
}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{ArrayIrType, ArrayType, DataType};
    use crate::backends::arrays::Array;
    use crate::operations::differentiation::StopGradientOperation;
    use crate::parameters::Placeholder;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::effects::Effect;

    use super::*;

    /// Test operation that makes every [`Operation`] method's forwarding observable for an arbitrary type family.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct ForwardingOperation<T: Type> {
        /// Whether [`Operation::rename_type_identities`] has been called.
        renamed: bool,

        /// Type family whose operation contract this fixture implements.
        marker: PhantomData<fn() -> T>,
    }

    impl<T: Type> Operation for ForwardingOperation<T> {
        type Type = T;

        fn name(&self) -> &'static str {
            "forwarding"
        }

        fn region_slots(&self) -> &'static [RegionSlot] {
            const { &[RegionSlot::computation("body")] }
        }

        fn infer_region_input_types(
            &self,
            input_types: &[T],
            _region_interfaces: &[RegionInterface<T>],
        ) -> Result<Vec<Option<Vec<T>>>, TypeError> {
            Ok(vec![Some(input_types.to_vec())])
        }

        fn infer_output_types(
            &self,
            input_types: &[T],
            region_interfaces: &[RegionInterface<T>],
        ) -> Result<Vec<T>, TypeError> {
            Ok(input_types.iter().chain(region_interfaces[0].output_types()).cloned().collect())
        }

        fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
            vec![OutputRegionProvenance { region_index: 0, output_index }]
        }

        fn is_zero(&self, output_index: usize) -> bool {
            output_index == 2
        }

        fn effects(&self) -> Effects {
            Effects::single(Effect::OrderedIo)
        }

        fn rename_type_identities(&self, _renaming: &TypeIdentityRenaming<T::Identity>) -> Result<Self, TypeError> {
            Ok(Self { renamed: true, marker: PhantomData })
        }

        fn render(&self, formatter: &mut std::fmt::Formatter<'_>, _indentation: usize) -> std::fmt::Result {
            formatter.write_str("forwarded")
        }
    }

    #[test]
    fn test_operation_formatter() {
        // Check that short fields are rendered inline.
        assert_eq!(
            std::fmt::from_fn(|formatter| {
                OperationFormatter::new(formatter, 0, "metadata")?.bracketed(|operation| {
                    operation.field("mode", "test")?;
                    operation.field("count", 2)
                })
            })
            .to_string(),
            "metadata [mode=test, count=2]",
        );

        // Check that long fields are wrapped over multiple lines.
        const LONG_VALUE: &str =
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz";
        assert_eq!(
            std::fmt::from_fn(|formatter| {
                OperationFormatter::new(formatter, 0, "metadata")?
                    .bracketed(|operation| operation.field("value", LONG_VALUE))
            })
            .to_string(),
            format!(
                indoc! {"
                    metadata [
                        value={value},
                    ]
                "},
                value = LONG_VALUE,
            )
            .trim_end()
        );

        // Check that program fields are rendered over multiple lines.
        let mut builder = ProgramBuilder::<Array, StopGradientOperation<ArrayType>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(StopGradientOperation::new(), Vec::new(), vec![input]).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            std::fmt::from_fn(|formatter| {
                OperationFormatter::new(formatter, 0, "nested")?.bracketed(|operation| {
                    operation.field("tag", "before")?;
                    operation.program("body", &program)?;
                    operation.field("tag", "after")
                })
            })
            .to_string(),
            indoc! {"
                nested [
                    tag=before,
                    body={
                        lambda %0:f64[] .
                        let %1:f64[] = stop_gradient %0
                        in (%1)
                    },
                    tag=after,
                ]
            "}
            .trim_end()
        );
    }

    #[test]
    fn test_operation() {
        let operation = StopGradientOperation::<DataType>::new();

        // Check required inference and the default operation contract.
        assert_eq!(operation.infer_output_types(&[DataType::F64], &[]), Ok(vec![DataType::F64]));
        assert_eq!(
            operation.infer_output_types(&[], &[]),
            Err(TypeError::invalid("expected 1 input but got 0".to_string())),
        );
        let region_interfaces = [
            RegionInterface::new(vec![DataType::F32], vec![DataType::F64], Effects::PURE),
            RegionInterface::new(vec![DataType::I32], vec![DataType::I64], Effects::PURE),
        ];

        assert_eq!(operation.region_slots(), &[]);
        assert_eq!(operation.region_role(0), None);
        assert_eq!(operation.infer_region_input_types(&[DataType::F64], &region_interfaces), Ok(vec![None, None]),);
        assert_eq!(operation.output_region_provenance(0), Vec::new());
        assert!(!operation.is_zero(0));
        assert_eq!(operation.effects(), Effects::PURE);
        assert!(operation.rename_type_identities(&TypeIdentityRenaming::new()).is_ok());
        assert_eq!(std::fmt::from_fn(|formatter| operation.render(formatter, 0)).to_string(), "stop_gradient");

        // Check that `Box<O>` forwards every method rather than silently falling back to a trait default.
        let operation = Box::new(ForwardingOperation::<DataType> { renamed: false, marker: PhantomData });
        let region_interfaces = [RegionInterface::new(vec![DataType::F32], vec![DataType::F64], Effects::PURE)];

        assert_eq!(operation.name(), "forwarding");
        assert_eq!(operation.region_slots(), &[RegionSlot::computation("body")]);
        assert_eq!(operation.region_role(0), Some(RegionRole::Computation));
        assert_eq!(operation.region_role(1), None);
        assert_eq!(
            operation.infer_region_input_types(&[DataType::F32], &region_interfaces),
            Ok(vec![Some(vec![DataType::F32])]),
        );
        assert_eq!(
            operation.infer_output_types(&[DataType::F32], &region_interfaces),
            Ok(vec![DataType::F32, DataType::F64]),
        );
        assert_eq!(
            operation.output_region_provenance(3),
            vec![OutputRegionProvenance { region_index: 0, output_index: 3 }],
        );
        assert!(!operation.is_zero(1));
        assert!(operation.is_zero(2));
        assert_eq!(operation.effects(), Effects::single(Effect::OrderedIo));
        assert_eq!(
            operation.rename_type_identities(&TypeIdentityRenaming::new()),
            Ok(Box::new(ForwardingOperation::<DataType> { renamed: true, marker: PhantomData })),
        );
        assert_eq!(std::fmt::from_fn(|formatter| operation.render(formatter, 0)).to_string(), "forwarded",);
    }

    #[test]
    fn test_infer_projected_operation_types() {
        let operation = ForwardingOperation::<ArrayType> { renamed: false, marker: PhantomData };
        let input_type = ArrayType::scalar(DataType::F32);
        let region_output_type = ArrayType::scalar(DataType::F64);
        let input_types = [ArrayIrType::from(input_type.clone())];
        let region_interfaces = [RegionInterface::new(
            vec![ArrayIrType::from(input_type.clone())],
            vec![ArrayIrType::from(region_output_type.clone())],
            Effects::single(Effect::OrderedIo),
        )];
        assert_eq!(
            infer_projected_operation_region_input_types(&operation, &input_types, &region_interfaces),
            Ok(vec![Some(vec![ArrayIrType::from(input_type.clone())])]),
        );
        assert_eq!(
            infer_projected_operation_output_types(&operation, &input_types, &region_interfaces),
            Ok(vec![ArrayIrType::from(input_type), ArrayIrType::from(region_output_type)]),
        );
    }
}
