//! Represents runtime values captured by staged [`Program`]s without embedding those values in reusable intermediate
//! representation.
//!
//! A captured program has two coordinated parts. Constant atoms in the staged program carry lifetime-free, typed
//! [`CaptureReference`] indices, while the concrete values they name live in a side table owned by [`ClosedProgram`].
//! The program therefore depends on capture types and positions rather than concrete data. Large arrays stay out of the
//! Intermediate Representation (IR), device buffers remain device-resident, and compiled executables can be reused with
//! different captured values of compatible types. Refer to [`ClosedProgram`] for a rendered diagram of this
//! representation and its compilation boundary. The compilation lifecycle built on these types lives in
//! the [`compilation`](crate::compilation) module.
//!
//! # Capturing During Tracing
//!
//! [`CapturingContext::capture`] registers one closed-over runtime value in the capture-owning trace and returns the
//! staged constant payload that refers to its new table slot. [`TracingContext`] owns the table. Nested tracing,
//! batching, differentiation, partial evaluation, and projected contexts delegate registration through their parent,
//! so a nested transform does not create a competing capture scope or embed a backend value as a literal.
//!
//! Multiple program atoms may refer to the same slot, and references may appear in any region of the program. The table
//! order is registration order and becomes the capture-argument order used by compilation.
//!
//! # Constant Families
//!
//! [`CaptureReference`] is the canonical constant representation for a purely capture-backed program.
//! [`CaptureConstant`] also supports sum-like backend constant families that can hold either a capture reference or an
//! immediate host-sized payload. Immediate constants return no capture index, carry their own data, and bypass capture
//! validation, pruning, and lifting. Every such family can still be constructed from a plain capture reference, which
//! is the payload returned when a runtime value is registered.
//!
//! # Closed-Program Invariant
//!
//! [`ClosedProgram::new`] is the sole construction boundary. It walks every region in the program arena and verifies
//! that each capture-referencing constant names an existing table entry with exactly the declared type. The program and
//! table are immutable afterward, so interpretation and transformation can rely on this invariant without rechecking
//! individual references.
//!
//! # Pruning and Transform Identity
//!
//! [`ClosedProgram::without_unused_captures`] removes table entries that no constant atom references and renumbers
//! surviving references into a contiguous table while preserving their relative order. A renumbered constant changes
//! the contents of its region, so that region and every transitive attaching ancestor discard retained program
//! transform artifacts. Unchanged siblings and descendants retain theirs. Removing only unused trailing captures
//! rewrites no reference and therefore preserves all region transform caches.
//!
//! # Lifting Captures for Compilation
//!
//! [`ClosedProgram::to_program_with_lifted_captures`] derives an open program whose leading inputs correspond to the
//! capture table and whose remaining inputs are the original public inputs. Every reference to one capture is rewritten
//! to the same leading input, while immediate constants remain constants. Compilation supplies arguments in
//! `[captures..., public inputs...]` order. The original closed program remains unchanged.
//!
//! # Extending Capture Support
//!
//! A new staged constant family implements [`CaptureConstant`] by exposing and rewriting only its reference-bearing
//! variants. A new capture-owning trace implements [`CapturingContext`] by storing concrete values and returning typed
//! staged references. Context wrappers should delegate capture registration to their parent unless they intentionally
//! establish a new closed-program boundary.

use std::borrow::Cow;
use std::fmt::{Debug, Display};

use ryft_macros::Parameter;

use crate::batching::{BatchableOperation, BatchingContext, RecursiveBatchingPolicy};
use crate::contexts::{Context, EagerContext, ProjectedContext};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationContext, ResidualZeroProvider,
};
use crate::macros::check_count;
use crate::parameters::{Parameter, Parameterized, Placeholder};
use crate::partial::{PartialEvaluationContext, PartiallyEvaluatableOperation};
use crate::programs::{
    Atom, AtomId, Operation, OperationProjection, Program, ProgramBuilder, ProgramError, ProjectedValue, RegionArena,
    Type, TypeError, TypeIdentityRenaming, Typed, Value, ValueProjection,
};
use crate::tracing::{NestedTracingContext, TracingContext};

/// Reference to a value captured outside a staged [`Program`]. A program stores only this lifetime-free reference
/// in its [`Atom`] table. The corresponding runtime value lives at [`index`](Self::index) in the surrounding
/// [`ClosedProgram`]'s capture table. This keeps concrete runtime values in the closed program's side environment
/// instead of embedding them in reusable staged IR.
///
/// # Why capture by reference instead of baking in a literal?
///
/// Closed-over runtime values (e.g., the arrays a just-in-time-compiled function closes over) are recorded as captures
/// and handed to the compiled program as runtime arguments, rather than embedded as literal constants in its IR. The
/// compiled program therefore depends only on the captured values' abstract types and never on their concrete data,
/// which buys three things:
///
///   - **Executable Reuse:** Compiled executables are cached by operand type and shape, not by value, so a single
///     compilation serves any captured value of a given type. Baking a value in as a literal would make the executable
///     value-specific and force a recompilation whenever the captured data changes.
///   - **On-Device Buffers:** Captured values are typically device (e.g., GPU) buffers. Passing them as arguments keeps
///     them resident on-device, whereas embedding them as literals would require copying them back to the host first.
///   - **Compact IR:** Large captured arrays never bloat the program IR or the serialized executables.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct CaptureReference<T: Type> {
    /// Index of this [`CaptureReference`] into the surrounding capture table.
    index: usize,

    /// [`Type`] of the underlying captured value.
    r#type: T,
}

impl<T: Type> CaptureReference<T> {
    /// Creates a new [`CaptureReference`].
    #[inline]
    pub fn new(index: usize, r#type: T) -> Self {
        Self { index, r#type }
    }

    /// Returns the index of this [`CaptureReference`] into the surrounding capture table.
    #[inline]
    pub fn index(&self) -> usize {
        self.index
    }
}

impl<T: Type> Display for CaptureReference<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "capture#{}:{}", self.index, self.r#type)
    }
}

impl<T: Type> Typed for CaptureReference<T> {
    type Type = T;

    #[inline]
    fn r#type(&self) -> Cow<'_, T> {
        Cow::Borrowed(&self.r#type)
    }
}

impl<T: Type> Value for CaptureReference<T> {
    type DispatchDomain = EagerContext<Self>;
    type ExecutionDomain = EagerContext<Self>;

    #[inline]
    fn dispatch_domain(&self) -> EagerContext<Self> {
        EagerContext::new()
    }

    #[inline]
    fn execution_domain(&self) -> EagerContext<Self> {
        EagerContext::new()
    }

    #[inline]
    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<T::Identity>) -> Result<Self, TypeError> {
        Ok(Self::new(self.index, self.r#type.rename_identities(renaming)?))
    }
}

impl<T: Type + From<P>, P: Type> ValueProjection<P> for CaptureReference<T>
where
    for<'a> &'a P: TryFrom<&'a T, Error = TypeError>,
{
    type Projected = CaptureReference<P>;
    type ProjectedRef<'v>
        = ProjectedValue<P, &'v Self>
    where
        Self: 'v,
        P: 'v;

    #[inline]
    fn from_projected(value: Self::Projected) -> Self {
        Self::new(value.index, value.r#type.into())
    }

    #[inline]
    fn projected<'v>(&'v self) -> Result<Self::ProjectedRef<'v>, TypeError>
    where
        P: 'v,
    {
        Ok(ProjectedValue::new(self, <&P>::try_from(&self.r#type)?.clone()))
    }

    #[inline]
    fn into_projected(self) -> Result<Self::Projected, TypeError> {
        let r#type = <&P>::try_from(&self.r#type)?.clone();
        Ok(CaptureReference::new(self.index, r#type))
    }
}

/// Constant payload that a [`ClosedProgram`] can store in its [`Atom`] table. A payload is either a _capture reference_
/// that names a runtime value in the surrounding capture table (the canonical [`CaptureReference`] case, which keeps
/// concrete data out of reusable staged intermediate representation) or an _immediate_ that carries its own host-sized
/// data and therefore never participates in capture bookkeeping. Backends that stage host-sized payloads directly
/// (e.g., a first-class dimension extent staged inside a manual `shard_map` region, where no capture table is
/// reachable) model their constant family as a sum of the two and report [`None`] from
/// [`capture_index`](Self::capture_index) for the immediate variants.
///
/// Capture validation, dead-capture elimination, and capture lifting are all expressed through this trait, and so
/// extending a constant family with immediates preserves the capture-table invariant that [`ClosedProgram`] upholds.
/// Every such family must be able to represent a plain capture reference, which is what
/// [`CapturingContext::capture`] hands back when a trace registers a runtime value.
pub trait CaptureConstant: Value + From<CaptureReference<Self::Type>> {
    /// Returns the capture-table index that this constant references, or [`None`] when it is an immediate payload
    /// that carries its own data.
    fn capture_index(&self) -> Option<usize>;

    /// Returns this constant with its capture-table index replaced by the result of applying `map` to it. Immediate
    /// payloads carry no index and are returned unchanged.
    fn map_capture_index<F: FnOnce(usize) -> usize>(&self, map: F) -> Self;
}

impl<T: Type> CaptureConstant for CaptureReference<T> {
    #[inline]
    fn capture_index(&self) -> Option<usize> {
        Some(self.index)
    }

    #[inline]
    fn map_capture_index<F: FnOnce(usize) -> usize>(&self, map: F) -> Self {
        Self::new(map(self.index), self.r#type.clone())
    }
}

/// [`Context`] that can register runtime values as captures (e.g., for a [`Program`] that is being built). The returned
/// value is the context's staged constant payload. For captured-program backends this is usually a lifetime-free
/// [`CaptureReference`] into a side table owned by the surrounding compiled function. Stackable transform contexts
/// delegate registration to their parent so captures follow the same nesting path as ordinary staged operations.
/// Capturing a closed-over value instead of staging it as a literal keeps the program independent of concrete data,
/// enabling executable reuse across captured values, retaining device buffers on-device, and keeping the IR compact.
pub trait CapturingContext: Context {
    /// Concrete runtime value type that this [`CapturingContext`] registers in its capture table. This is deliberately
    /// distinct from both [`Value`](crate::Domain::Value) (i.e., the value type flowing through the context, which is
    /// typically a [`Tracer`](crate::Tracer) while capturing) and [`Constant`](crate::Domain::Constant) (i.e., the
    /// staged payload that [`capture`](Self::capture) returns).
    type Capture: Value<Type = Self::Type>;

    /// Appends `value` to the current captures table of this [`CapturingContext`] and returns the constant payload
    /// that refers to it.
    fn capture(&self, value: Self::Capture) -> Result<Self::Constant, ProgramError>;
}

impl<C: CapturingContext, T: Type> CapturingContext for ProjectedContext<C, T>
where
    C::Value: ValueProjection<T, Projected: Value<Type = T>>,
    C::Constant: ValueProjection<T, Projected: Value<Type = T>>,
    C::Operation: OperationProjection<T>,
    C::Capture: ValueProjection<T, Projected: Value<Type = T>>,
{
    type Capture = <C::Capture as ValueProjection<T>>::Projected;

    #[inline]
    fn capture(&self, value: Self::Capture) -> Result<Self::Constant, ProgramError> {
        self.parent()
            .capture(<C::Capture as ValueProjection<T>>::from_projected(value))?
            .into_projected()
            .map_err(Into::into)
    }
}

// Any traced constant family that can embed a `CaptureReference` registers captures the same way: the runtime value
// is appended to the trace's capture table and the staged payload is the reference to its slot. Constant families that
// also carry immediate payloads therefore inherit capture registration unchanged.
impl<T: Type, V: Value<Type = T> + From<CaptureReference<T>>, O: Operation<Type = T>, C: Value<Type = T>>
    CapturingContext for TracingContext<V, O, C>
{
    type Capture = C;

    #[inline]
    fn capture(&self, value: C) -> Result<Self::Constant, ProgramError> {
        let mut captures = self.captures().borrow_mut();
        let constant = CaptureReference::new(captures.len(), value.r#type().into_owned());
        captures.push(value);
        Ok(constant.into())
    }
}

impl<C: CapturingContext> CapturingContext for NestedTracingContext<C> {
    type Capture = C::Capture;

    #[inline]
    fn capture(&self, value: Self::Capture) -> Result<Self::Constant, ProgramError> {
        self.parent().capture(value)
    }
}

impl<C: CapturingContext> CapturingContext for PartialEvaluationContext<C>
where
    C::Operation:
        PartiallyEvaluatableOperation<C> + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>,
{
    type Capture = C::Capture;

    #[inline]
    fn capture(&self, value: Self::Capture) -> Result<Self::Constant, ProgramError> {
        self.parent().capture(value)
    }
}

impl<C: CapturingContext<Operation: BatchableOperation<C, P>>, P: RecursiveBatchingPolicy<C>> CapturingContext
    for BatchingContext<C, P>
{
    type Capture = C::Capture;

    #[inline]
    fn capture(&self, value: Self::Capture) -> Result<Self::Constant, ProgramError> {
        self.parent().capture(value)
    }
}

impl<C: CapturingContext> CapturingContext for DifferentiationContext<C>
where
    C::Type: DifferentiableType,
    C::Operation: PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
        + DifferentiableOperation<C>
        + DifferentiableOperation<TracingContext<C::Constant, C::Operation>>
        + DifferentiableOperation<PartialEvaluationContext<TracingContext<C::Constant, C::Operation>>>
        + ResidualZeroProvider<C::Type>,
{
    type Capture = C::Capture;

    #[inline]
    fn capture(&self, value: Self::Capture) -> Result<Self::Constant, ProgramError> {
        self.parent().capture(value)
    }
}

/// A [`Program`] paired with the concrete runtime values referenced by its captured constants. Concrete values live
/// only in [`captures`](Self::captures). Program constant atoms carry lifetime-free [`CaptureConstant`] payloads,
/// including [`CaptureReference`] indices and any immediate variants admitted by the backend's constant family.
/// [`new`](Self::new) validates every reference across the complete region arena before constructing the pair.
///
/// # Capture Representation
///
/// ```mermaid
/// %%{init: {"themeCSS": ".nodeLabel code { white-space: nowrap !important; }"}}%%
/// flowchart TD
///   runtime["Closed-Over Runtime Value"] -->|"&lt;code&gt;capture&lt;/code&gt;"| context["Capturing Context"]
///   context --> table["Append Concrete Value to Runtime Capture Table"]
///   context --> reference["Return Typed Capture Reference"]
///   reference --> atom["Program Constant Atom"]
///   immediate["Immediate Host-Sized Constant"] --> atom
///   atom --> program["Staged Program with Complete Region Arena"]
///   table --> validate["&lt;code&gt;ClosedProgram::new&lt;/code&gt; Validates Every Reference"]
///   program --> validate
///   validate --> closed["Validated Closed Program"]
///   closed --> reusable["Reusable Staged IR plus Concrete Capture Environment"]
///   lifted["Leading Capture Inputs then Public Inputs"]
///   closed -->|"&lt;code&gt;to_program_with_lifted_captures&lt;/code&gt;"| lifted
///   lifted --> compilation["Compilation Boundary"]
/// ```
///
/// One capture table scopes the entry region and every attached region. Immediate constants participate in the program
/// but not in the table, while repeated capture references continue to name one table slot and one lifted input.
#[cfg_attr(doc, aquamarine::aquamarine)]
pub struct ClosedProgram<
    C: Value,
    V: CaptureConstant<Type = C::Type>,
    O,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
> {
    /// [`Program`] whose constants are [`CaptureConstant`] payloads.
    program: Program<V, O, Input, Output>,

    /// Captured values referenced by [`CaptureConstant::capture_index`] indices in [`Self::program`].
    captures: Vec<C>,
}

impl<
    C: Value,
    V: CaptureConstant<Type = C::Type>,
    O: Operation<Type = C::Type>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
> ClosedProgram<C, V, O, Input, Output>
{
    /// Creates a [`ClosedProgram`] from a capture-referenced `program` and its concrete `captures`, validating that
    /// every capture-referencing constant in the program references an existing capture whose type matches the type
    /// stored in the reference. Immediate constants carry their own data and are skipped. This is the sole construction
    /// path and both the program and its capture table are immutable after construction, and so every [`ClosedProgram`]
    /// upholds this capture-table invariant for its entire lifetime and no separate re-validation is ever needed.
    pub fn new(program: Program<V, O, Input, Output>, captures: Vec<C>) -> Result<Self, ProgramError> {
        // Every region in the arena participates in the single capture scope, so validation walks all of them.
        for region in program.regions().iter() {
            for (atom_index, atom) in region.atoms().iter().enumerate() {
                let Atom::Constant(value) = atom else {
                    continue;
                };
                let Some(index) = value.capture_index() else {
                    continue;
                };
                let capture = captures.get(index).ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "captured constant atom %{atom_index} references missing capture #{index}",
                    ))
                })?;
                let expected_type = value.r#type();
                let actual_type = capture.r#type();
                if expected_type.as_ref() != actual_type.as_ref() {
                    return Err(ProgramError::MalformedProgram(format!(
                        "captured constant atom %{} references capture #{} with type {}, but the atom has type {}",
                        atom_index, index, actual_type, expected_type,
                    )));
                }
            }
        }
        Ok(Self { program, captures })
    }

    /// Returns the underlying [`Program`].
    #[inline]
    pub fn program(&self) -> &Program<V, O, Input, Output> {
        &self.program
    }

    /// Returns the underlying captures table.
    #[inline]
    pub fn captures(&self) -> &[C] {
        self.captures.as_slice()
    }

    /// Removes captures that no [`Atom`] of the underlying [`Program`] references and re-indexes the survivors into
    /// a contiguous capture table. This is _dead-capture elimination_: any capture whose index never appears in an
    /// [`Atom::Constant`] is dropped, and the remaining captures are renumbered to occupy `0..captures().len()` in
    /// their original relative order. The program structure (i.e., inputs, instructions, and outputs) is preserved
    /// and only capture-referencing constant atoms are rewritten to carry their new indices; immediate constants are
    /// index-free and pass through untouched. This function consumes `self` so that the surviving captures, the
    /// instruction operations, and the input/output structures can all be moved into the rebuilt program instead of
    /// being cloned. Retained per-region transforms survive wherever the rewrite provably changed nothing (a region
    /// detaches from them only when one of its own constant atoms was renumbered or when a region attached to it
    /// detached).
    pub fn without_unused_captures(self) -> Result<Self, ProgramError> {
        let Self { program, captures } = self;

        // Mark the captures that at least one constant atom references. Every region participates in the program's
        // single capture scope, so the marking pass walks all of them. Indexing into `is_referenced` cannot fail
        // because `new` validated every capture reference at construction time.
        let mut is_referenced = vec![false; captures.len()];
        for region in program.regions().iter() {
            for atom in region.atoms() {
                if let Atom::Constant(constant) = atom
                    && let Some(index) = constant.capture_index()
                {
                    is_referenced[index] = true;
                }
            }
        }

        // Move the referenced captures into a contiguous table that preserves their original relative order,
        // recording each survivor's new index. Captures that are never referenced are dropped here.
        let mut capture_index_map = vec![None; captures.len()];
        let mut filtered_captures = Vec::new();
        for (source_index, capture) in captures.into_iter().enumerate() {
            if is_referenced[source_index] {
                capture_index_map[source_index] = Some(filtered_captures.len());
                filtered_captures.push(capture);
            }
        }

        // Rewrite every capture-referencing constant atom in place, across every region, to carry its capture's new
        // index. The program structure (i.e., atoms, identifiers, instructions, regions, and boundaries) is preserved
        // exactly. The `capture_index_map` lookups cannot fail because the marking pass above assigns a slot to every
        // capture referenced by any constant atom. Note that renumbering a capture reference changes what the region's
        // constants denote, so a region whose atoms actually change detaches from the transforms derived from its
        // previous contents, and so does every region that attaches such a region, because a transform of a region
        // consumes its attached descendants too. Regions precede the regions that reference them, which makes this one
        // ascending pass enough to propagate that. Every other subtree keeps its transforms: dropping captures that
        // shift no survivor (i.e., trailing unused ones) rewrites no reference at all, and renumbering one region's
        // references leaves regions that reference no renumbered capture reusable.
        let Program { input_structure, output_structure, regions, entry, .. } = program;
        let mut regions = regions.into_regions();
        let mut is_invalidated = vec![false; regions.len()];
        for (region_index, region) in regions.iter_mut().enumerate() {
            let mut is_rewritten = false;
            for atom in &mut region.atoms {
                if let Atom::Constant(constant) = atom
                    && let Some(source_index) = constant.capture_index()
                {
                    let destination_index = capture_index_map[source_index].unwrap();
                    if destination_index != source_index {
                        *constant = constant.map_capture_index(|_| destination_index);
                        is_rewritten = true;
                    }
                }
            }
            let invalidates = is_rewritten
                || region
                    .instructions()
                    .iter()
                    .flat_map(|instruction| instruction.regions())
                    .any(|attached| is_invalidated.get(attached.index()).copied().unwrap_or(true));
            if invalidates {
                region.invalidate_transform_cache();
            }
            is_invalidated[region_index] = invalidates;
        }

        // Re-sealing preserves each region's retained transforms because this rebuild moves the complete source arena
        // in its original order, so every attached identifier still names the same body. The regions whose reachable
        // contents actually changed detached from their transforms above.
        let program = Program::new_preserving_transform_caches(input_structure, output_structure, regions, entry)?;

        // Constructing through `new` re-validates the rewritten references against the pruned capture table.
        Self::new(program, filtered_captures)
    }

    /// Returns a [`Program`] where the captures have been lifted into explicit leading inputs that are followed by
    /// the original program inputs. The [`ClosedProgram`] itself is unchanged. The returned program is a derived
    /// view used by compilation, which supplies arguments in `[captures..., public inputs...]` order.
    pub fn to_program_with_lifted_captures(&self) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError> {
        /// Materializes the rebuilt atom identifiers for `atom_ids`, resolving each atom either through `mapped_atoms`
        /// (which memoizes the rebuilt identifiers of program inputs, instruction outputs, and re-added immediate
        /// constants) or, for a capture-referencing constant, to the leading capture input that `capture_inputs`
        /// records for the capture the constant references. The `capture_inputs` lookup cannot fail because
        /// [`ClosedProgram::new`] validated that every capture reference names an existing capture, and one leading
        /// input exists per capture.
        fn map_atoms<Constant: CaptureConstant>(
            atoms: &[Atom<Constant>],
            mapped_atoms: &[Option<AtomId>],
            capture_inputs: &[AtomId],
            atom_ids: &[AtomId],
        ) -> Result<Vec<AtomId>, ProgramError> {
            atom_ids
                .iter()
                .copied()
                .map(|atom_id| {
                    if let Some(mapped) = mapped_atoms.get(atom_id.index()).copied().flatten() {
                        return Ok(mapped);
                    }
                    match atoms.get(atom_id.index()) {
                        Some(Atom::Constant(constant)) => constant.capture_index().map(|index| capture_inputs[index]),
                        Some(Atom::Variable(_)) => None,
                        None => return Err(ProgramError::UnboundAtomId { id: atom_id }),
                    }
                    .ok_or_else(|| {
                        ProgramError::MalformedProgram(format!(
                            "atom {atom_id} has no mapped input, instruction output, or capture argument",
                        ))
                    })
                })
                .collect()
        }

        // Add one leading input per capture, in capture-table order and unconditionally (a capture that no atom
        // references still occupies its input slot), because execution supplies arguments positionally in
        // `[captures..., public inputs...]` order. Entry-region constant atoms referencing capture `k` resolve to
        // `capture_inputs[k]` during the replay below. Constants inside attached regions are preserved verbatim
        // (backend-specific lowering like XLA lowering will resolve capture references against the same hidden capture
        // argument prefix while lowering those regions, and materialize immediate constants in place). Nested regions
        // are imported verbatim ahead of the replay. Their identifiers are arena indices assigned in order, so copying
        // them in order preserves every entry-instruction region reference, which is also why re-sealing them keeps the
        // transforms already derived from their contents.
        let mut builder = ProgramBuilder::new();
        builder.regions = RegionArena::from_regions_preserving_transform_caches(
            self.program.regions().iter().take(self.program.entry().index()).cloned().collect(),
        )?;
        let capture_inputs = self
            .captures
            .iter()
            .map(|capture| builder.add_input(capture.r#type().into_owned()))
            .collect::<Vec<_>>();

        // The original program inputs follow the capture inputs and must be re-added before any instruction is replayed
        // so that instruction operands referencing them resolve through `mapped_atoms` instead of being treated as
        // unmapped atoms. `mapped_atoms` tracks the rebuilt `AtomId` of every source atom so that an atom shared across
        // instructions maps to a single rebuilt atom instead of being duplicated.
        let mut mapped_atoms = vec![None; self.program.atoms().len()];
        for input_id in self.program.input_ids().iter().copied() {
            let input =
                self.program.atoms().get(input_id.index()).ok_or(ProgramError::UnboundAtomId { id: input_id })?;
            let Atom::Variable(input_type) = input else {
                return Err(ProgramError::MalformedProgram("program input atom was not a variable".to_string()));
            };
            mapped_atoms[input_id.index()] = Some(builder.add_input(input_type.clone()));
        }

        // Immediate constants carry their own data instead of naming a capture slot, and so they are re-added to the
        // rebuilt entry region as constant atoms rather than resolving to a leading capture argument.
        for (atom_index, atom) in self.program.atoms().iter().enumerate() {
            if let Atom::Constant(constant) = atom
                && constant.capture_index().is_none()
            {
                mapped_atoms[atom_index] = Some(builder.add_constant(constant.clone()));
            }
        }

        // Replay the instructions in order, mapping their operands and recording their rebuilt outputs so that later
        // instructions (and the program outputs) can resolve them. This borrows `self`, and so the operations are
        // cloned into the rebuilt program (cheaply, since nested regions are copied by arena splicing with sharing
        // preserved).
        for instruction in self.program.instructions() {
            let inputs = map_atoms(
                self.program.atoms(),
                mapped_atoms.as_slice(),
                capture_inputs.as_slice(),
                instruction.inputs(),
            )?;
            let outputs = builder
                .add_instruction(instruction.operation().clone(), instruction.regions().to_vec(), inputs)?
                .to_vec();
            check_count!("output", outputs, instruction.outputs().len(), ProgramError);
            for (source_output, rebuilt_output) in instruction.outputs().iter().copied().zip(outputs) {
                let mapped = mapped_atoms
                    .get_mut(source_output.index())
                    .ok_or(ProgramError::UnboundAtomId { id: source_output })?;
                *mapped = Some(rebuilt_output);
            }
        }

        // Program outputs may be original inputs, instruction outputs, or captured constants,
        // and so they are resolved through the same atom mapping as instruction operands.
        let output_ids = map_atoms(
            self.program.atoms(),
            mapped_atoms.as_slice(),
            capture_inputs.as_slice(),
            self.program.output_ids(),
        )?;

        // The lifted program is flat. Its input and output boundaries are placeholder vectors sized to the
        // `[captures..., public inputs...]` argument convention and the original outputs, respectively.
        builder.build::<Vec<V>, Vec<V>>(
            output_ids,
            vec![Placeholder; self.captures.len() + self.program.input_ids().len()],
            vec![Placeholder; self.program.output_ids().len()],
        )
    }
}

impl<C: Value, V: CaptureConstant<Type = C::Type>, O: Clone, Input: Parameterized<V>, Output: Parameterized<V>> Clone
    for ClosedProgram<C, V, O, Input, Output>
{
    #[inline]
    fn clone(&self) -> Self {
        Self { program: self.program.clone(), captures: self.captures.clone() }
    }
}

impl<C: Value, V: CaptureConstant<Type = C::Type>, O, Input: Parameterized<V>, Output: Parameterized<V>> Debug
    for ClosedProgram<C, V, O, Input, Output>
{
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ClosedProgram")
            .field("captures", &self.captures.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, DataType, Dimension,
        DimensionBounds, DimensionType, DimensionVariable, Shape,
    };
    use crate::contexts::{EagerContext, StagingContext};
    use crate::interpretation::InterpretableOperation;
    use crate::operations::{AddOperation, CompareOperation, ComparisonDirection, WhileOperation};
    use crate::parameters::Placeholder;
    use crate::programs::{
        EmptyRegionDriver, ProgramBuilder, ProgramError, ReferenceType, RegionId, RegionSlot, TypeIdentityRenaming,
        ValueProjection,
    };
    use crate::tests::TestRegionOperation;
    use crate::tracing::{NestedTracingContext, Tracer, TracingContext};

    use super::*;

    #[test]
    fn test_capture_reference_identity_renaming_preserves_capture_index() {
        let bounds = DimensionBounds::non_negative(Some(16)).unwrap();
        let source = DimensionVariable::new("source", bounds);
        let target = DimensionVariable::new("target", bounds);
        let capture = CaptureReference::new(
            3,
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(source.clone())])),
        );
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(source, target.clone()).unwrap();

        let renamed = capture.rename_type_identities(&renaming).unwrap();
        assert_eq!(renamed.index(), 3);
        assert_eq!(
            renamed.r#type().as_ref(),
            &ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(target)])),
        );
    }

    #[test]
    fn test_capture_reference_projection_preserves_capture_index() {
        let array_type = ArrayType::scalar(DataType::F32);
        let capture = CaptureReference::new(3, ArrayIrType::Array(array_type.clone()));
        let projected = <CaptureReference<ArrayIrType> as ValueProjection<ArrayType>>::projected(&capture).unwrap();
        assert_eq!(projected.value().index(), 3);
        assert_eq!(projected.r#type().as_ref(), &array_type);

        let projected = <CaptureReference<ArrayIrType> as ValueProjection<ArrayType>>::into_projected(capture).unwrap();
        assert_eq!(projected.index(), 3);
        assert_eq!(projected.r#type().as_ref(), &array_type);
        let lifted = <CaptureReference<ArrayIrType> as ValueProjection<ArrayType>>::from_projected(projected);
        assert_eq!(lifted.index(), 3);
        assert_eq!(lifted.r#type().as_ref(), &ArrayIrType::Array(array_type));

        assert_eq!(
            <CaptureReference<ArrayIrType> as ValueProjection<DimensionType>>::projected(&lifted),
            Err(TypeError::invalid("expected dimension type but got array type")),
        );

        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let capture = CaptureReference::new(5, ArrayIrType::Reference(reference_type.clone()));
        let projected =
            <CaptureReference<ArrayIrType> as ValueProjection<ReferenceType<ArrayType>>>::projected(&capture).unwrap();
        assert_eq!(projected.value().index(), 5);
        assert_eq!(projected.r#type().as_ref(), &reference_type);
        let projected =
            <CaptureReference<ArrayIrType> as ValueProjection<ReferenceType<ArrayType>>>::into_projected(capture)
                .unwrap();
        assert_eq!(projected.index(), 5);
        assert_eq!(projected.r#type().as_ref(), &reference_type);
        let lifted =
            <CaptureReference<ArrayIrType> as ValueProjection<ReferenceType<ArrayType>>>::from_projected(projected);
        assert_eq!(lifted.index(), 5);
        assert_eq!(lifted.r#type().as_ref(), &ArrayIrType::Reference(reference_type));
    }

    #[test]
    fn test_projected_context_capture_delegates_to_parent_capture_table() {
        let parent =
            TracingContext::<CaptureReference<ArrayIrType>, ArrayIrOperation<Array>, ArrayIrValue<Array>>::new();
        let array_context = ProjectedContext::<_, ArrayType>::new(parent.clone());
        let array = Array::scalar(3.0_f32);
        let array_reference = array_context.capture(array.clone()).unwrap();
        assert_eq!(array_reference.index(), 0);
        assert_eq!(array_reference.r#type(), array.r#type());
        let dimension_context = ProjectedContext::<_, DimensionType>::new(parent.clone());
        let dimension = crate::DimensionValue::constant(7).unwrap();
        let dimension_reference = dimension_context.capture(dimension.clone()).unwrap();
        assert_eq!(dimension_reference.index(), 1);
        assert_eq!(dimension_reference.r#type().as_ref(), dimension.r#type().as_ref());
        assert_eq!(
            parent.captures().borrow().as_slice(),
            &[ArrayIrValue::Array(array), ArrayIrValue::Dimension(dimension),],
        );
    }

    #[test]
    fn test_closed_program_without_unused_captures() {
        // Construction rejects references to missing captures.
        let mut builder = ProgramBuilder::<CaptureReference<ArrayType>, ArrayOperation<Array>>::new();
        let capture = builder.add_constant(CaptureReference::new(1, ArrayType::scalar(DataType::F64)));
        let program = builder
            .build::<Vec<CaptureReference<ArrayType>>, Vec<CaptureReference<ArrayType>>>(
                vec![capture],
                Vec::<Placeholder>::new(),
                vec![Placeholder],
            )
            .unwrap();
        assert!(matches!(
            ClosedProgram::new(program, vec![Array::scalar(3.0)]),
            Err(ProgramError::MalformedProgram(message))
                if message == "captured constant atom %0 references missing capture #1",
        ));

        // Construction rejects references whose declared type differs from their capture's runtime type.
        let mut builder = ProgramBuilder::<CaptureReference<ArrayType>, ArrayOperation<Array>>::new();
        let capture = builder.add_constant(CaptureReference::new(0, ArrayType::scalar(DataType::I64)));
        let program = builder
            .build::<Vec<CaptureReference<ArrayType>>, Vec<CaptureReference<ArrayType>>>(
                vec![capture],
                Vec::<Placeholder>::new(),
                vec![Placeholder],
            )
            .unwrap();
        assert!(matches!(
            ClosedProgram::new(program, vec![Array::scalar(3.0)]),
            Err(ProgramError::MalformedProgram(message))
                if message
                    == "captured constant atom %0 references capture #0 with type f64[], but the atom has type i64[]",
        ));

        // Pruning drops the dead capture #0 and re-indexes the surviving capture #1 into a contiguous table
        // while preserving the program structure.
        let mut builder = ProgramBuilder::<CaptureReference<ArrayType>, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let capture = builder.add_constant(CaptureReference::new(1, ArrayType::scalar(DataType::F64)));
        let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, capture]).unwrap()[0];
        let program = builder
            .build::<Vec<CaptureReference<ArrayType>>, Vec<CaptureReference<ArrayType>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let program = ClosedProgram::new(program, vec![Array::scalar(3.0), Array::scalar(99.0)]).unwrap();
        let pruned = program.without_unused_captures().unwrap();
        assert_eq!(pruned.captures(), &[Array::scalar(99.0)]);
        let capture_indices = pruned
            .program()
            .atoms()
            .iter()
            .filter_map(|atom| atom.as_constant().map(CaptureReference::index))
            .collect::<Vec<_>>();
        assert_eq!(capture_indices, vec![0]);
        assert_eq!(
            pruned.program().to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = const capture#0:f64[]
                    %2:f64[] = add %0 %1
                in (%2)
            "}
            .trim_end(),
        );

        // Pruning a program whose captures are all referenced is an identity transformation.
        let repruned = pruned.clone().without_unused_captures().unwrap();
        assert_eq!(repruned.captures(), pruned.captures());
        assert_eq!(repruned.program().to_string(), pruned.program().to_string());

        // The pruned program interprets equivalently through the re-indexed capture table.
        let output = pruned
            .program()
            .interpret_with(
                vec![Array::scalar(2.0)],
                |_, reference| Ok::<_, ProgramError>(pruned.captures()[reference.index()].clone()),
                |instruction, inputs| {
                    instruction.operation().interpret(
                        &EagerContext::<Array, ArrayOperation<Array>>::new(),
                        &EmptyRegionDriver,
                        inputs,
                    )
                },
            )
            .unwrap();
        assert_eq!(output, vec![Array::scalar(101.0)]);
    }

    #[test]
    fn test_closed_program_without_unused_captures_invalidates_only_on_renumbering() {
        /// Array-captured program family this test prunes.
        type TestClosedProgram = ClosedProgram<
            Array,
            CaptureReference<ArrayType>,
            ArrayOperation<Array>,
            Vec<CaptureReference<ArrayType>>,
            Vec<CaptureReference<ArrayType>>,
        >;

        /// Builds `input + capture#<used_capture>` over a two-entry capture table.
        fn closed_program(used_capture: usize) -> TestClosedProgram {
            let mut builder = ProgramBuilder::<CaptureReference<ArrayType>, ArrayOperation<Array>>::new();
            let input = builder.add_input(ArrayType::scalar(DataType::F64));
            let capture = builder.add_constant(CaptureReference::new(used_capture, ArrayType::scalar(DataType::F64)));
            let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, capture]).unwrap()[0];
            let program = builder
                .build::<Vec<CaptureReference<ArrayType>>, Vec<CaptureReference<ArrayType>>>(
                    vec![output],
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap();
            ClosedProgram::new(program, vec![Array::scalar(3.0), Array::scalar(99.0)]).unwrap()
        }

        // Dropping an unused trailing capture shifts no surviving capture, so no constant atom is rewritten and every
        // region keeps the transforms already derived from its contents.
        let closed = closed_program(0);
        let retained = closed.program().entry_region_ref().retained_identity_transform();
        let pruned = closed.without_unused_captures().unwrap();
        assert_eq!(pruned.captures(), &[Array::scalar(3.0)]);
        let artifact = pruned.program().entry_region_ref().retained_identity_transform();
        assert!(Arc::ptr_eq(&artifact, &retained));

        // Dropping an unused leading capture renumbers the survivor, which changes what the rewritten constant atoms
        // denote and must therefore discard every transform retained for the region.
        let closed = closed_program(1);
        let retained = closed.program().entry_region_ref().retained_identity_transform();
        let pruned = closed.without_unused_captures().unwrap();
        assert_eq!(pruned.captures(), &[Array::scalar(99.0)]);
        let artifact = pruned.program().entry_region_ref().retained_identity_transform();
        assert!(!Arc::ptr_eq(&artifact, &retained));
    }

    #[test]
    fn test_closed_program_without_unused_captures_invalidates_only_affected_regions() {
        // Capture #0 survives at index 0 and is referenced only by the first attached region, capture #1 is unused and
        // dropped, and capture #2 is referenced only by the second attached region and renumbers to #1. Only the
        // second sibling's contents therefore change, and the entry region inherits that through the attachment.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut unchanged_builder = ProgramBuilder::<CaptureReference<ArrayType>, TestRegionOperation>::new();
        let unchanged_constant = unchanged_builder.add_constant(CaptureReference::new(0, scalar_type.clone()));
        let unchanged_program = unchanged_builder
            .build::<Vec<CaptureReference<ArrayType>>, Vec<CaptureReference<ArrayType>>>(
                vec![unchanged_constant],
                Vec::<Placeholder>::new(),
                vec![Placeholder],
            )
            .unwrap();
        let mut renumbered_builder = ProgramBuilder::<CaptureReference<ArrayType>, TestRegionOperation>::new();
        let renumbered_constant = renumbered_builder.add_constant(CaptureReference::new(2, scalar_type.clone()));
        let renumbered_program = renumbered_builder
            .build::<Vec<CaptureReference<ArrayType>>, Vec<CaptureReference<ArrayType>>>(
                vec![renumbered_constant],
                Vec::<Placeholder>::new(),
                vec![Placeholder],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<CaptureReference<ArrayType>, TestRegionOperation>::new();
        let unchanged_region = builder.import_region(unchanged_program.entry_region_ref());
        let renumbered_region = builder.import_region(renumbered_program.entry_region_ref());
        let input = builder.add_input(scalar_type);
        let output = builder
            .add_instruction(
                TestRegionOperation::WithRegions(
                    const { &[RegionSlot::computation("first"), RegionSlot::computation("second")] },
                ),
                vec![unchanged_region, renumbered_region],
                vec![input],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<CaptureReference<ArrayType>>, Vec<CaptureReference<ArrayType>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let closed =
            ClosedProgram::new(program, vec![Array::scalar(1.0), Array::scalar(2.0), Array::scalar(3.0)]).unwrap();
        let unchanged_retained = closed.program().region_ref(RegionId::new(0)).unwrap().retained_identity_transform();
        let renumbered_retained = closed.program().region_ref(RegionId::new(1)).unwrap().retained_identity_transform();
        let entry_retained = closed.program().entry_region_ref().retained_identity_transform();

        // The untouched sibling keeps both its reference and its retained transform, while the rewritten sibling and
        // the entry region that attaches it discard theirs.
        let pruned = closed.without_unused_captures().unwrap();
        assert_eq!(pruned.captures(), &[Array::scalar(1.0), Array::scalar(3.0)]);
        assert_eq!(pruned.program().regions()[0].atoms()[0].as_constant().map(CaptureReference::index), Some(0));
        assert_eq!(pruned.program().regions()[1].atoms()[0].as_constant().map(CaptureReference::index), Some(1));
        let unchanged = pruned.program().region_ref(RegionId::new(0)).unwrap().retained_identity_transform();
        let renumbered = pruned.program().region_ref(RegionId::new(1)).unwrap().retained_identity_transform();
        let entry = pruned.program().entry_region_ref().retained_identity_transform();
        assert!(Arc::ptr_eq(&unchanged, &unchanged_retained));
        assert!(!Arc::ptr_eq(&renumbered, &renumbered_retained));
        assert!(!Arc::ptr_eq(&entry, &entry_retained));
    }

    #[test]
    fn test_closed_program_without_unused_captures_with_attached_regions() {
        // Captures referenced only inside an attached region survive pruning, and reference indices rewrite in every
        // region. Capture #0 is unused (dropped), #1 is referenced only by the nested region (kept, becomes #0), and
        // #2 is referenced only by the entry region (kept, becomes #1).
        let mut builder = ProgramBuilder::<CaptureReference<ArrayType>, TestRegionOperation>::new();
        let mut region_builder = ProgramBuilder::<CaptureReference<ArrayType>, TestRegionOperation>::new();
        let nested_capture = region_builder.add_constant(CaptureReference::new(1, ArrayType::scalar(DataType::F64)));
        let region_program = region_builder
            .build::<Vec<CaptureReference<ArrayType>>, Vec<CaptureReference<ArrayType>>>(
                vec![nested_capture],
                Vec::<Placeholder>::new(),
                vec![Placeholder],
            )
            .unwrap();
        let sealed = builder.import_region(region_program.entry_region_ref());
        let entry_capture = builder.add_constant(CaptureReference::new(2, ArrayType::scalar(DataType::F64)));
        let output = builder
            .add_instruction(
                TestRegionOperation::WithRegions(const { &[RegionSlot::computation("body")] }),
                vec![sealed],
                vec![entry_capture],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<CaptureReference<ArrayType>>, Vec<CaptureReference<ArrayType>>>(
                vec![output],
                Vec::<Placeholder>::new(),
                vec![Placeholder],
            )
            .unwrap();
        let program =
            ClosedProgram::new(program, vec![Array::scalar(1.0), Array::scalar(2.0), Array::scalar(3.0)]).unwrap();
        let pruned = program.without_unused_captures().unwrap();
        assert_eq!(pruned.captures(), &[Array::scalar(2.0), Array::scalar(3.0)]);
        let nested_indices = pruned.program().regions()[0]
            .atoms()
            .iter()
            .filter_map(|atom| atom.as_constant().map(CaptureReference::index))
            .collect::<Vec<_>>();
        assert_eq!(nested_indices, vec![0]);
        let entry_indices = pruned
            .program()
            .atoms()
            .iter()
            .filter_map(|atom| atom.as_constant().map(CaptureReference::index))
            .collect::<Vec<_>>();
        assert_eq!(entry_indices, vec![1]);
    }

    #[test]
    fn test_closed_program_to_program_with_lifted_captures() {
        // The program computes `(input + capture#0) + capture#0` through one shared constant atom, and capture #1 is
        // never referenced, so the lift covers shared references and dead captures at once.
        let mut builder = ProgramBuilder::<CaptureReference<ArrayType>, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let capture = builder.add_constant(CaptureReference::new(0, ArrayType::scalar(DataType::F64)));
        let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, capture]).unwrap()[0];
        let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![sum, capture]).unwrap()[0];
        let program = builder
            .build::<Vec<CaptureReference<ArrayType>>, Vec<CaptureReference<ArrayType>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let program = ClosedProgram::new(program, vec![Array::scalar(3.0), Array::scalar(7.0)]).unwrap();

        // Captures become leading inputs in capture-table order (one per capture, dead ones included), followed by
        // the original program input, and no captured-constant atoms remain. The shared constant atom maps to a
        // single capture input.
        let lifted = program.to_program_with_lifted_captures().unwrap();
        assert_eq!(lifted.input_ids().len(), 3);
        assert!(lifted.atoms().iter().all(|atom| !atom.is_constant()));
        assert_eq!(
            lifted.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[] .
                let %3:f64[] = add %2 %0
                    %4:f64[] = add %3 %0
                in (%4)
            "}
            .trim_end(),
        );

        // The closed program is unchanged: the lift is a derived view.
        assert_eq!(program.captures().len(), 2);
        assert_eq!(program.program().atoms().iter().filter(|atom| atom.is_constant()).count(), 1);

        // The lifted program interprets with arguments supplied in `[captures..., public inputs...]` order.
        let output = lifted
            .interpret_with::<Array, ProgramError, _, _>(
                vec![Array::scalar(3.0), Array::scalar(7.0), Array::scalar(2.0)],
                |_, _| unreachable!("the lifted program contains no captured-constant atoms"),
                |instruction, inputs| {
                    instruction.operation().interpret(
                        &EagerContext::<Array, ArrayOperation<Array>>::new(),
                        &EmptyRegionDriver,
                        inputs,
                    )
                },
            )
            .unwrap();
        assert_eq!(output, vec![Array::scalar(8.0)]);
    }

    #[test]
    fn test_closed_program_immediate_constants_bypass_the_capture_table() {
        /// Constant family that mixes capture references with immediate literals, mirroring how a backend extends
        /// its constant universe with payloads that carry their own host-sized data.
        #[derive(Clone, Debug, PartialEq)]
        enum TestConstant {
            /// Reference into the surrounding capture table.
            Captured(CaptureReference<ArrayType>),

            /// Immediate literal that carries its own data.
            Immediate(Array),
        }

        impl Parameter for TestConstant {}

        impl Display for TestConstant {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    Self::Captured(value) => Display::fmt(value, formatter),
                    Self::Immediate(value) => Display::fmt(value, formatter),
                }
            }
        }

        impl Typed for TestConstant {
            type Type = ArrayType;

            fn r#type(&self) -> Cow<'_, ArrayType> {
                match self {
                    Self::Captured(value) => value.r#type(),
                    Self::Immediate(value) => value.r#type(),
                }
            }
        }

        impl Value for TestConstant {
            type DispatchDomain = EagerContext<Self>;
            type ExecutionDomain = EagerContext<Self>;

            fn dispatch_domain(&self) -> EagerContext<Self> {
                EagerContext::new()
            }

            fn execution_domain(&self) -> EagerContext<Self> {
                EagerContext::new()
            }

            fn rename_type_identities(
                &self,
                renaming: &TypeIdentityRenaming<<ArrayType as Type>::Identity>,
            ) -> Result<Self, TypeError> {
                match self {
                    Self::Captured(value) => Ok(Self::Captured(value.rename_type_identities(renaming)?)),
                    Self::Immediate(value) => Ok(Self::Immediate(value.rename_type_identities(renaming)?)),
                }
            }
        }

        impl From<CaptureReference<ArrayType>> for TestConstant {
            fn from(value: CaptureReference<ArrayType>) -> Self {
                Self::Captured(value)
            }
        }

        impl CaptureConstant for TestConstant {
            fn capture_index(&self) -> Option<usize> {
                match self {
                    Self::Captured(value) => Some(value.index()),
                    Self::Immediate(_) => None,
                }
            }

            fn map_capture_index<F: FnOnce(usize) -> usize>(&self, map: F) -> Self {
                match self {
                    Self::Captured(value) => Self::Captured(value.map_capture_index(map)),
                    Self::Immediate(value) => Self::Immediate(value.clone()),
                }
            }
        }

        // The program computes `(input + capture#1) + immediate`, capture #0 is dead, and the immediate constant
        // carries its own literal instead of naming a capture slot.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut builder = ProgramBuilder::<TestConstant, ArrayOperation<Array>>::new();
        let input = builder.add_input(scalar_type.clone());
        let capture = builder.add_constant(TestConstant::Captured(CaptureReference::new(1, scalar_type.clone())));
        let immediate = builder.add_constant(TestConstant::Immediate(Array::scalar(5.0)));
        let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, capture]).unwrap()[0];
        let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![sum, immediate]).unwrap()[0];
        let program = builder
            .build::<Vec<TestConstant>, Vec<TestConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // Construction validates only the capture-referencing constant, so an immediate never has to name a slot.
        let closed = ClosedProgram::new(program, vec![Array::scalar(3.0), Array::scalar(7.0)]).unwrap();

        // Dead-capture elimination renumbers the surviving reference and leaves the immediate untouched.
        let closed = closed.without_unused_captures().unwrap();
        assert_eq!(closed.captures(), &[Array::scalar(7.0)]);
        let constants =
            closed.program().atoms().iter().filter_map(|atom| atom.as_constant()).cloned().collect::<Vec<_>>();
        assert_eq!(
            constants,
            vec![
                TestConstant::Captured(CaptureReference::new(0, scalar_type)),
                TestConstant::Immediate(Array::scalar(5.0)),
            ],
        );

        // Lifting turns the capture reference into a leading argument while the immediate stays a constant atom, and
        // so the lifted program takes `[capture, input]` and still folds its own literal in.
        let lifted = closed.to_program_with_lifted_captures().unwrap();
        assert_eq!(lifted.input_ids().len(), 2);
        assert_eq!(
            lifted.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = const 5.0
                    %3:f64[] = add %1 %0
                    %4:f64[] = add %3 %2
                in (%4)
            "}
            .trim_end(),
        );
        let output = lifted
            .interpret_with::<Array, ProgramError, _, _>(
                vec![Array::scalar(7.0), Array::scalar(2.0)],
                |_, constant| match constant {
                    TestConstant::Captured(_) => unreachable!("capture references are lifted into inputs"),
                    TestConstant::Immediate(value) => Ok(value.clone()),
                },
                |instruction, inputs| {
                    instruction.operation().interpret(
                        &EagerContext::<Array, ArrayOperation<Array>>::new(),
                        &EmptyRegionDriver,
                        inputs,
                    )
                },
            )
            .unwrap();
        assert_eq!(output, vec![Array::scalar(14.0)]);
    }

    #[test]
    fn test_closed_program_to_program_with_lifted_captures_with_attached_regions() {
        // Lifting supports capture-free attached regions (imported verbatim ahead of the entry replay) and rejects
        // nested-region capture references until region boundaries can thread them.
        let mut builder = ProgramBuilder::<CaptureReference<ArrayType>, TestRegionOperation>::new();
        let mut region_builder = ProgramBuilder::<CaptureReference<ArrayType>, TestRegionOperation>::new();
        let region_input = region_builder.add_input(ArrayType::scalar(DataType::F64));
        let region_program = region_builder
            .build::<Vec<CaptureReference<ArrayType>>, Vec<CaptureReference<ArrayType>>>(
                vec![region_input],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let sealed = builder.import_region(region_program.entry_region_ref());
        let entry_capture = builder.add_constant(CaptureReference::new(0, ArrayType::scalar(DataType::F64)));
        let output = builder
            .add_instruction(
                TestRegionOperation::WithRegions(const { &[RegionSlot::computation("body")] }),
                vec![sealed],
                vec![entry_capture],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<CaptureReference<ArrayType>>, Vec<CaptureReference<ArrayType>>>(
                vec![output],
                Vec::<Placeholder>::new(),
                vec![Placeholder],
            )
            .unwrap();
        let program = ClosedProgram::new(program, vec![Array::scalar(1.0)]).unwrap();
        let lifted = program.to_program_with_lifted_captures().unwrap();
        assert_eq!(lifted.regions().len(), 2);
        assert_eq!(lifted.input_ids().len(), 1);
        assert_eq!(lifted.instructions()[0].regions(), &[RegionId::new(0)]);

        // A nested region that references a capture keeps that reference for backends that resolve nested constants
        // against the lifted capture prefix while lowering attached regions.
        let mut builder = ProgramBuilder::<CaptureReference<ArrayType>, TestRegionOperation>::new();
        let mut region_builder = ProgramBuilder::<CaptureReference<ArrayType>, TestRegionOperation>::new();
        let nested_capture = region_builder.add_constant(CaptureReference::new(0, ArrayType::scalar(DataType::F64)));
        let region_program = region_builder
            .build::<Vec<CaptureReference<ArrayType>>, Vec<CaptureReference<ArrayType>>>(
                vec![nested_capture],
                Vec::<Placeholder>::new(),
                vec![Placeholder],
            )
            .unwrap();
        let sealed = builder.import_region(region_program.entry_region_ref());
        let entry_input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(
                TestRegionOperation::WithRegions(const { &[RegionSlot::computation("body")] }),
                vec![sealed],
                vec![entry_input],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<CaptureReference<ArrayType>>, Vec<CaptureReference<ArrayType>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let program = ClosedProgram::new(program, vec![Array::scalar(1.0)]).unwrap();
        let lifted = program.to_program_with_lifted_captures().unwrap();
        assert_eq!(lifted.input_ids().len(), 2);
        assert_eq!(lifted.instructions()[0].regions(), &[RegionId::new(0)]);
        assert_eq!(lifted.regions()[0].atoms()[0].as_constant().map(CaptureReference::index), Some(0));
    }

    #[test]
    fn test_capturing_context_nested_trace_preserves_nested_only_captures() {
        // A capture registered through a nested trace is referenced only by a constant inside an attached region.
        // Region-aware capture discovery must still keep it when pruning unused captures.
        let root =
            TracingContext::<CaptureReference<ArrayType>, ArrayOperation<CaptureReference<ArrayType>>, Array>::new();
        let state = root.input(ArrayType::scalar(DataType::F64));

        // The condition is set to `state < state` (always false) and it is traced as a nested payload program.
        let (_, condition) = NestedTracingContext::trace(
            root.clone(),
            |inputs: Vec<Tracer<_>>| {
                inputs[0].context().bind(
                    CompareOperation::new(ComparisonDirection::LessThan),
                    Vec::new(),
                    &[inputs[0].clone(), inputs[0].clone()],
                )
            },
            vec![ArrayType::scalar(DataType::F64)],
        )
        .unwrap();

        // The body is set to `state + capture#0`, with the capture registered through the *nested* trace,
        // and so its reference constant is staged only into the nested body program.
        let (_, body) = NestedTracingContext::trace(
            root.clone(),
            |inputs: Vec<Tracer<_>>| {
                let context = inputs[0].context().clone();
                let reference = context.capture(Array::scalar(3.0))?;
                let captured = StagingContext::constant(&context, reference);
                context.bind(AddOperation::new(), Vec::new(), &[inputs[0].clone(), captured])
            },
            vec![ArrayType::scalar(DataType::F64)],
        )
        .unwrap();
        let operation = WhileOperation::new();
        let outputs = root
            .bind(ArrayOperation::While(operation), vec![condition, body], std::slice::from_ref(&state))
            .unwrap();

        // The top-level program holds no capture-reference atoms. The only reference lives in the while body.
        let output_ids = outputs.iter().map(Tracer::atom_id).collect::<Result<Vec<_>, _>>().unwrap();
        let builder = root.builder().borrow().clone();
        let captures = root.captures().borrow().clone();
        let program = builder
            .build::<Vec<CaptureReference<ArrayType>>, Vec<CaptureReference<ArrayType>>>(
                output_ids,
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let closed = ClosedProgram::new(program, captures).unwrap();
        assert!(closed.program().atoms().iter().all(|atom| !atom.is_constant()));
        assert_eq!(closed.captures(), &[Array::scalar(3.0)]);

        // The while body is an attached region, so capture-use discovery walks into it and pruning keeps the
        // nested-only capture.
        let pruned = closed.clone().without_unused_captures().unwrap();
        assert_eq!(pruned.captures(), &[Array::scalar(3.0)]);
    }
}
