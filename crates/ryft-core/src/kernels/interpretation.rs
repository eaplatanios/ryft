//! Host interpretation of qualified kernel calls through canonical references and program replay.
//!
//! Qualification precedes storage creation. Input arrays remain immutable; every invocation owns private reference
//! roots, and only completed writable roots become results. Masked edge windows use private tiles, explicit load
//! fallbacks, and publication restricted to valid coordinates. Qualification proves initialization, coverage, and
//! disjointness before execution; the interpreter never chooses an unresolved dynamic launch extent.
//!
//! Scalar-prefetched values are bound through canonical program specialization before qualification or allocation.
//! Async copies capture initialized source elements when issued and publish their destination only at the matching
//! wait. Ordinary reference effects retain their existing semantics; the host traversal chooses one admitted order
//! for parallel atomic updates, and does not promise bitwise agreement across other valid floating-point orderings.
//!
//! To investigate a failure, retain the trace before propagating the execution result. Optional NaN checks inspect
//! ordinary operation inputs/results, including complex components and low-precision floating-point formats. They
//! never read uninitialized scratch merely to inspect its contents. Precision and rounding remain those of the
//! canonical operations; diagnostics do not substitute a wider-precision execution model.
//!
//! ```
//! use ryft_core::arrays::Array;
//! use ryft_core::kernels::{KernelDebugOptions, KernelDefinition};
//! use ryft_core::programs::ProgramError;
//!
//! # fn inspect(definition: &KernelDefinition, inputs: Vec<Array>) -> Result<(), ProgramError> {
//! let options = KernelDebugOptions { check_nans: true, maximum_steps: 10_000, ..Default::default() };
//! let mut trace = Vec::new();
//! let result = definition.interpret_with_trace(inputs, &options, &mut trace);
//! for entry in trace {
//!     eprintln!("{entry}");
//! }
//! let outputs = result?;
//! # Ok(())
//! # }
//! ```

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::rc::Rc;

use num_complex::Complex;
use thiserror::Error;

use crate::arrays::{
    Array, ArrayAddressing, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayReference, ArrayReferenceView,
    ArraySliceAxis, ArrayType, DataType, DimensionValue,
};
use crate::contexts::{Context, Domain, EagerContext, ValueResolution};
use crate::interpretation::{
    EagerInterpretationDriver, InterpretableOperation, InterpretationDriver, RegionInterpreter,
};
use crate::kernels::calls::{KernelCallOperation, KernelDefinition};
use crate::kernels::initialization::{KernelInitializationError, validate_kernel_initialization};
use crate::kernels::mappings::BoundaryPolicy;
use crate::kernels::memory::{KernelMemoryError, ScratchOperation};
use crate::kernels::operations::{KernelExtension, KernelOperation};
use crate::kernels::validation::KernelParameterAccess;
use crate::operations::Zero;
use crate::programs::{
    BindingRegionDriver, Operation, ProgramError, Provenance, ProvenanceScope, ProvenanceState, ReferenceAccessMode,
    ReferenceId, RegionDriver, RegionRef, TypeError, Typed,
};

/// Host debugging limits exceeded while replaying a kernel; these limits do not alter its device semantics.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
pub enum KernelInterpretationError {
    /// Optional numerical checking found a NaN in an ordinary value. Reference storage is never read for this check.
    #[error(
        "nan in `{operation}` {position} {value} element {element} of type `{data_type}` at grid point {coordinate:?}"
    )]
    Nan {
        /// Canonical operation that produced or consumed the value.
        operation: &'static str,
        /// Whether the value is an operation `input` or `output`.
        position: &'static str,
        /// Position in that operation's input or output list.
        value: usize,
        /// Logical row-major element index; either complex component may be NaN.
        element: usize,
        /// Original element type, before diagnostic inspection.
        data_type: DataType,
        /// Logical grid coordinate at the failing operation.
        coordinate: Vec<usize>,
        /// Original source provenance at the failing operation.
        provenance: Provenance,
    },

    /// A wait did not identify a pending copy owned by this invocation.
    #[error("async copy completion token is unknown or already consumed")]
    UnknownCopyToken {
        /// Canonical allocation identity of the supplied completion token.
        token: ReferenceId,
    },

    /// Defensive publication check for copies that have not completed successfully.
    #[error("kernel grid point {coordinate:?} has {count} copies without a wait")]
    PendingCopies {
        /// Number of incomplete copies in the program instance.
        count: usize,
        /// Logical grid coordinate whose output cannot be published.
        coordinate: Vec<usize>,
    },

    /// The invocation exhausted its operation and region-entry budget before successful completion.
    #[error("kernel interpretation exceeded {maximum_steps} steps at grid point {coordinate:?} in `{operation}`")]
    StepLimit {
        /// Maximum replay steps requested for the complete invocation.
        maximum_steps: usize,
        /// Logical grid coordinate being interpreted when the limit was reached.
        coordinate: Vec<usize>,
        /// Canonical operation executing or entering an attached region.
        operation: &'static str,
        /// Existing program provenance at the failing step.
        provenance: Provenance,
    },
}

/// Host-only debugging choices. Mandatory bounds, race, and initialization validation cannot be disabled. Numerical
/// checking observes ordinary values without changing their dtype, rounding, or accumulator behavior; those remain
/// properties of canonical operations. These options are excluded from kernel and compiler identity.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelDebugOptions {
    /// Maximum logical grid size admitted before storage is allocated.
    pub maximum_programs: usize,
    /// Maximum operation bindings and region entries across the invocation.
    pub maximum_steps: usize,
    /// Rejects NaNs in ordinary floating-point or complex operation inputs/results. Infinity is allowed.
    pub check_nans: bool,
}

impl Default for KernelDebugOptions {
    fn default() -> Self {
        Self {
            maximum_programs: DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS,
            maximum_steps: DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_STEPS,
            check_nans: false,
        }
    }
}

/// One attempted operation binding during host replay. Entries precede effects, so the final entry is retained when
/// its operation fails. Reference identities are numbered by first observation within this invocation, independent
/// of process-global allocation IDs. Region-entry budget checks do not create operation entries.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelTraceEntry {
    /// Logical program coordinate in the declared grid order.
    pub coordinate: Vec<usize>,
    /// Canonical operation name.
    pub operation: &'static str,
    /// Canonical provenance active at this binding.
    pub provenance: Provenance,
    /// Input reference accesses in the operation's declared effect order.
    pub accesses: Vec<KernelTraceAccess>,
    /// Invocation-local identities of copies pending before this operation starts.
    pub pending_copies: Vec<usize>,
}

/// A reference access in a [`KernelTraceEntry`], with its canonical view and invocation-local root identity.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelTraceAccess {
    /// Input position of the accessed reference.
    pub input: usize,
    /// Exact canonical reference access mode.
    pub mode: ReferenceAccessMode,
    /// Allocation number assigned on first observation during this invocation.
    pub root: usize,
    /// Canonical view transformations from this allocation to the selected elements.
    pub view: String,
}

impl Display for KernelTraceEntry {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{:?} {}", self.coordinate, self.operation)?;
        for access in &self.accesses {
            write!(formatter, " input[{}]={}(root[{}]{})", access.input, access.mode, access.root, access.view)?;
        }
        write!(formatter, " pending={:?}", self.pending_copies)?;
        if !self.provenance.is_unknown() {
            write!(formatter, " at {}", self.provenance)?;
        }
        Ok(())
    }
}

/// Mutable collection shared only by nested host replay contexts. It never enters semantic identity or compiled IR.
#[derive(Default)]
struct KernelTraceState {
    /// Attempted bindings collected in execution order.
    entries: Vec<KernelTraceEntry>,
    /// Stable local names for observed canonical allocation identities.
    roots: BTreeMap<ReferenceId, usize>,
    /// Optional numerical checks applied to bound ordinary values.
    check_nans: bool,
}

/// Default host replay budget, counting operation bindings and region entries across the entire grid. Counting region
/// entries also bounds loops with constant conditions and empty bodies. This limit is excluded from semantic identity.
pub const DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_STEPS: usize = 1_000_000;

/// Maximum number of logical programs qualified by an ordinary
/// [`Program::interpret`](crate::programs::Program::interpret) kernel call. [`KernelDefinition::interpret`] accepts
/// an explicit budget for larger debug launches. This is a host resource limit, not part of a kernel's semantic
/// identity or a device launch limit.
pub const DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS: usize = 10_000;

impl<Extension> KernelDefinition<Extension>
where
    Extension: KernelExtension + InterpretableOperation<EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>>,
{
    /// Interprets this definition on host arrays after checking initialization, output coverage, and races. The
    /// inputs and results follow the functional signature: write-only parameters consume no input, and read-only
    /// parameters produce no result. A failure never publishes a partially updated array or mutates an input.
    ///
    /// # Parameters
    ///
    ///   - `inputs`: arrays in the definition's ordinary input order.
    ///   - `maximum_programs`: maximum logical grid size allowed by this host invocation; zero permits only an empty
    ///     launch. Qualification rejects a larger grid before allocating reference storage or replaying a body.
    pub fn interpret(&self, inputs: Vec<Array>, maximum_programs: usize) -> Result<Vec<Array>, ProgramError> {
        self.interpret_with_limits(inputs, maximum_programs, DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_STEPS)
    }

    /// Interprets this definition with explicit host grid and replay limits. A replay step is an operation binding or
    /// region entry, so empty loop bodies still consume the budget. All grid points share one counter; nested region
    /// replay preserves the original instruction provenance. Limit failures discard private output storage.
    ///
    /// # Parameters
    ///
    ///   - `inputs`: arrays in the definition's ordinary input order.
    ///   - `maximum_programs`: maximum number of grid points allowed before execution begins.
    ///   - `maximum_steps`: maximum operation bindings and region entries across the invocation. Zero permits an
    ///     empty grid but rejects the first body entry of a nonempty launch.
    pub fn interpret_with_limits(
        &self,
        inputs: Vec<Array>,
        maximum_programs: usize,
        maximum_steps: usize,
    ) -> Result<Vec<Array>, ProgramError> {
        self.interpret_recorded(inputs, maximum_programs, maximum_steps, None, None)
    }

    /// Interprets with an operation trace retained on success or failure. Existing entries are cleared before
    /// qualification; a qualification failure therefore leaves an empty trace. The trace records attempted bindings,
    /// exact reference accesses, pending copy tokens, and original source provenance. No array contents are recorded.
    /// Host limits have the same meaning as in [`Self::interpret_with_limits`].
    pub fn interpret_with_trace(
        &self,
        inputs: Vec<Array>,
        options: &KernelDebugOptions,
        trace: &mut Vec<KernelTraceEntry>,
    ) -> Result<Vec<Array>, ProgramError> {
        self.interpret_in_order(inputs, options, None, trace)
    }

    /// Shares ordinary replay with bounded scheduler-selected orders. Only the scheduler supplies an explicit order;
    /// it has already checked flat-region eligibility and generated every instruction occurrence exactly once.
    pub(super) fn interpret_in_order(
        &self,
        inputs: Vec<Array>,
        options: &KernelDebugOptions,
        order: Option<&[usize]>,
        trace: &mut Vec<KernelTraceEntry>,
    ) -> Result<Vec<Array>, ProgramError> {
        trace.clear();
        let state = Rc::new(RefCell::new(KernelTraceState { check_nans: options.check_nans, ..Default::default() }));
        let result = self.interpret_recorded(
            inputs,
            options.maximum_programs,
            options.maximum_steps,
            Some(state.clone()),
            order,
        );
        *trace = std::mem::take(&mut state.borrow_mut().entries);
        result
    }

    /// Runs the ordinary interpreter with an optional invocation-local trace collector.
    fn interpret_recorded(
        &self,
        inputs: Vec<Array>,
        maximum_programs: usize,
        maximum_steps: usize,
        trace: Option<Rc<RefCell<KernelTraceState>>>,
        order: Option<&[usize]>,
    ) -> Result<Vec<Array>, ProgramError> {
        let context = EagerContext::<ArrayIrValue<Array>, KernelOperation<Extension>>::new();
        let regions = vec![self.body().clone()];
        let driver = EagerInterpretationDriver::new(&regions);
        self.operation()
            .interpret_with_budget(
                &context,
                &driver,
                &inputs.into_iter().map(ArrayIrValue::Array).collect::<Vec<_>>(),
                maximum_programs,
                maximum_steps,
                trace,
                order,
            )?
            .into_iter()
            .map(|value| match value {
                ArrayIrValue::Array(array) => Ok(array),
                _ => Err(ProgramError::MalformedProgram("kernel interpreter returned a non-array result".to_owned())),
            })
            .collect()
    }
}

impl<Extension> InterpretableOperation<EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>>
    for KernelCallOperation
where
    Extension: KernelExtension + InterpretableOperation<EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>>,
{
    fn interpret<D: InterpretationDriver<EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>>>(
        &self,
        context: &EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>,
        driver: &D,
        inputs: &[ArrayIrValue<Array>],
    ) -> Result<Vec<ArrayIrValue<Array>>, ProgramError> {
        self.interpret_with_budget(
            context,
            driver,
            inputs,
            DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS,
            DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_STEPS,
            None,
            None,
        )
    }
}

impl KernelCallOperation {
    /// Qualifies and executes an attached body without relying on a previously constructed definition's summary.
    fn interpret_with_budget<Extension, D>(
        &self,
        context: &EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>,
        driver: &D,
        inputs: &[ArrayIrValue<Array>],
        maximum_programs: usize,
        maximum_steps: usize,
        trace: Option<Rc<RefCell<KernelTraceState>>>,
        order: Option<&[usize]>,
    ) -> Result<Vec<ArrayIrValue<Array>>, ProgramError>
    where
        Extension:
            KernelExtension + InterpretableOperation<EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>>,
        D: InterpretationDriver<EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>>,
    {
        let body = driver.region(0)?;
        self.infer_output_types(
            &inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
            &[body.interface()],
        )?;
        if !self.prefetch_types().is_empty() {
            let ordinary_count = inputs.len() - self.prefetch_types().len();
            let prefetched = inputs[ordinary_count..]
                .iter()
                .map(|value| {
                    let ArrayIrValue::Array(array) = value else {
                        unreachable!("the validated prefetch signature contains only arrays");
                    };
                    array.clone()
                })
                .collect::<Vec<_>>();
            let definition = KernelDefinition::new(self.clone(), body.to_program()).map_err(ProgramError::custom)?;
            let specialized = definition.specialize_prefetch(&prefetched).map_err(ProgramError::custom)?;
            let regions = vec![specialized.body().clone()];
            let driver = EagerInterpretationDriver::new(&regions);
            return specialized.operation().interpret_with_budget(
                context,
                &driver,
                &inputs[..ordinary_count],
                maximum_programs,
                maximum_steps,
                trace,
                order,
            );
        }
        let extents = self
            .grid()
            .dimensions()
            .iter()
            .map(|dimension| {
                dimension.extent().value().ok_or_else(|| {
                    ProgramError::custom(KernelInitializationError::UnsupportedLaunch { boundary: "grid extents" })
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let points = self.grid().points(&extents).map_err(ProgramError::custom)?;
        if points.len() > maximum_programs {
            return Err(ProgramError::custom(KernelInitializationError::QualificationLimit {
                programs: points.len(),
                maximum: maximum_programs,
            }));
        }
        validate_kernel_initialization(body, self, maximum_programs).map_err(ProgramError::custom)?;

        let array_context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let mut input_values = inputs.iter();
        let roots = self
            .parameters()
            .iter()
            .map(|parameter| {
                let array = match parameter.access() {
                    KernelParameterAccess::WriteOnly => array_context.zero(parameter.r#type().as_ref())?,
                    KernelParameterAccess::ReadOnly | KernelParameterAccess::ReadWrite => {
                        let ArrayIrValue::Array(array) = input_values.next().unwrap() else {
                            unreachable!("the validated functional signature contains only arrays");
                        };
                        array.clone()
                    }
                };
                Ok(ArrayReference::new(array))
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;
        let remaining_steps = Rc::new(Cell::new(maximum_steps));
        let provenance = Rc::new(ProvenanceState::new());
        let mut scheduled = Vec::new();
        for coordinate in points {
            let mut validity = BTreeMap::new();
            let mut publications = Vec::new();
            let mut windows = self
                .parameters()
                .iter()
                .zip(&roots)
                .map(|(parameter, root)| {
                    let mapping_inputs = parameter
                        .mapping()
                        .program()
                        .input_types()
                        .iter()
                        .zip(&coordinate)
                        .map(|(r#type, &index)| {
                            let ArrayIrType::Dimension(r#type) = r#type else {
                                unreachable!("validated mappings have only dimension inputs");
                            };
                            DimensionValue::new(r#type.clone(), index).map_err(ProgramError::from)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let shape = parameter.r#type().static_shape().unwrap();
                    let window = parameter
                        .mapping()
                        .evaluate(&mapping_inputs, shape.dimensions())
                        .map_err(ProgramError::custom)?;
                    let source = root.with_transform(window.valid_view().clone())?;
                    let reference = if parameter.mapping().boundary_policy() == BoundaryPolicy::Masked {
                        let ArrayIrType::Reference(body_type) = parameter.body_type() else { unreachable!() };
                        let tile_type = body_type.referent();
                        let tile = ArrayReference::new(array_context.zero(tile_type)?);
                        let valid_shape = source.r#type().referent().static_shape().unwrap();
                        let selected = tile.with_transform(ArrayReferenceView::Slice {
                            axes: valid_shape
                                .dimensions()
                                .iter()
                                .map(|&extent| ArraySliceAxis::new(0, extent, 1))
                                .collect(),
                        })?;
                        if parameter.access() != KernelParameterAccess::WriteOnly {
                            Self::copy_window(&source, &selected)?;
                        }
                        let mask_type = tile_type.clone().with_data_type(DataType::Boolean);
                        let addressing = ArrayAddressing::new(mask_type.clone())?;
                        let mut index = vec![0; tile_type.rank()];
                        let mut lanes = Vec::with_capacity(addressing.element_count());
                        for _ in 0..addressing.element_count() {
                            lanes.push(
                                index.iter().zip(valid_shape.dimensions()).all(|(&index, &extent)| index < extent),
                            );
                            addressing.advance_index(&mut index);
                        }
                        validity.insert(tile.id(), ArrayReference::new(Array::from_elements(mask_type, &lanes)?));
                        if parameter.access() != KernelParameterAccess::ReadOnly {
                            publications.push((selected, source));
                        }
                        tile
                    } else {
                        source
                    };
                    let value = ArrayIrValue::Reference(reference);
                    if value.r#type().as_ref() != &parameter.body_type() {
                        return Err(ProgramError::MalformedProgram(
                            "kernel window type differs from its declared canonical body type".to_owned(),
                        ));
                    }
                    Ok(value)
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            windows.extend(
                self.coordinate_types()
                    .iter()
                    .zip(&coordinate)
                    .map(|(r#type, &index)| DimensionValue::new(r#type.clone(), index).map(ArrayIrValue::Dimension))
                    .collect::<Result<Vec<_>, _>>()?,
            );
            let qualified = QualifiedKernelContext {
                eager: *context,
                validity,
                coordinate,
                maximum_steps,
                remaining_steps: remaining_steps.clone(),
                provenance: provenance.clone(),
                trace: trace.clone(),
                pending_copies: Rc::new(RefCell::new(BTreeMap::new())),
            };
            if order.is_some() {
                scheduled.push(Some(RegionInterpreter::new(qualified, body, windows)?));
                continue;
            }
            qualified.check_step(self.name())?;
            body.interpret_in_context(&qualified, windows)?;
            let pending = qualified.pending_copies.borrow().len();
            if pending != 0 {
                return Err(ProgramError::custom(KernelInterpretationError::PendingCopies {
                    count: pending,
                    coordinate: qualified.coordinate.clone(),
                }));
            }
            for (source, destination) in publications {
                Self::copy_window(&source, &destination)?;
            }
        }
        if let Some(order) = order {
            let mut started = vec![false; scheduled.len()];
            for &program in order {
                let state = scheduled[program].take().unwrap();
                if !started[program] {
                    state.context().check_step(self.name())?;
                    started[program] = true;
                }
                scheduled[program] = Some(state.step()?);
            }
            for (program, state) in scheduled.into_iter().enumerate() {
                let state = state.unwrap();
                if !started[program] {
                    state.context().check_step(self.name())?;
                }
                let pending = state.context().pending_copies.borrow().len();
                if pending != 0 {
                    return Err(ProgramError::custom(KernelInterpretationError::PendingCopies {
                        count: pending,
                        coordinate: state.context().coordinate.clone(),
                    }));
                }
                state.finish()?;
            }
        }
        self.parameters()
            .iter()
            .zip(roots)
            .filter(|(parameter, _)| parameter.access() != KernelParameterAccess::ReadOnly)
            .map(|(_, root)| root.freeze().map(ArrayIrValue::Array))
            .collect()
    }

    /// Copies only valid selected coordinates between a logical edge tile and the original operand. Physical layout
    /// may differ because the tile is private; shape, dtype, placement, and distributed dependencies must agree.
    fn copy_window(source: &ArrayReference<Array>, destination: &ArrayReference<Array>) -> Result<(), ProgramError> {
        let value = source.read()?;
        let destination_type = destination.r#type().referent().clone();
        if value.r#type().into_owned().with_layout(None) != destination_type.clone().with_layout(None) {
            return Err(
                TypeError::invalid("kernel edge tile and operand window have incompatible logical types").into()
            );
        }
        destination.write(Array::from_logical_bytes(destination_type, &value.logical_bytes())?)
    }
}

impl<C: Domain> InterpretableOperation<C> for ScratchOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        _inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        Err(ProgramError::custom(KernelMemoryError::RequiresQualification { operation: self.name() }))
    }
}

/// Private replay authority created only after the entire attached body passes initialization qualification.
/// Portable eager rules remain unchanged; their child-region driver retains this authority across recursion.
#[derive(Clone)]
struct QualifiedKernelContext<Extension: Operation<Type = ArrayIrType>> {
    /// Canonical eager arithmetic and reference semantics.
    eager: EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>,
    /// Per-tile valid coordinates; private physical padding never counts as readable operand storage.
    validity: BTreeMap<ReferenceId, ArrayReference<Array>>,
    /// Logical program being replayed for diagnostics.
    coordinate: Vec<usize>,
    /// Invocation-wide host limit, retained for exact diagnostics.
    maximum_steps: usize,
    /// Shared across every grid point and nested region of this invocation.
    remaining_steps: Rc<Cell<usize>>,
    /// Canonical exception-safe provenance composition during ordinary program replay.
    provenance: Rc<ProvenanceState>,
    /// Source snapshots and reserved destinations awaiting their canonical completion tokens.
    pending_copies: Rc<RefCell<BTreeMap<ReferenceId, PendingCopy>>>,
    /// Optional observer shared across all points and attached regions.
    trace: Option<Rc<RefCell<KernelTraceState>>>,
}

/// One deterministic asynchronous copy: its destination remains unchanged until the matching wait succeeds.
struct PendingCopy {
    /// Initialized source elements captured when the copy started.
    source: Array,
    /// Canonical destination view reserved until completion.
    destination: ArrayReference<Array>,
}

impl<Extension: Operation<Type = ArrayIrType>> QualifiedKernelContext<Extension> {
    /// Charges one bounded replay step before any instruction or region entry can perform effects.
    fn check_step(&self, operation: &'static str) -> Result<(), ProgramError> {
        let remaining = self.remaining_steps.get();
        if remaining == 0 {
            return Err(ProgramError::custom(KernelInterpretationError::StepLimit {
                maximum_steps: self.maximum_steps,
                coordinate: self.coordinate.clone(),
                operation,
                provenance: self.provenance.current(),
            }));
        }
        self.remaining_steps.set(remaining - 1);
        Ok(())
    }

    /// Inspects ordinary values only; reference initialization and reads retain their explicit operation semantics.
    fn check_nans(
        &self,
        operation: &'static str,
        position: &'static str,
        values: &[ArrayIrValue<Array>],
    ) -> Result<(), ProgramError> {
        if !self.trace.as_ref().is_some_and(|trace| trace.borrow().check_nans) {
            return Ok(());
        }
        for (value, array) in values.iter().enumerate() {
            let ArrayIrValue::Array(array) = array else {
                continue;
            };
            let data_type = array.r#type().data_type();
            if !data_type.is_floating_point() && !data_type.is_complex() {
                continue;
            }
            // Diagnostic widening must not inherit byte strides sized for a narrower element representation.
            let array =
                Array::from_logical_bytes(array.r#type().into_owned().with_layout(None), &array.logical_bytes())?;
            let element = if data_type.is_floating_point() {
                array.converted_to(DataType::F64)?.elements::<f64>()?.iter().position(|value| value.is_nan())
            } else if data_type.is_complex() {
                array
                    .converted_to(DataType::C128)?
                    .elements::<Complex<f64>>()?
                    .iter()
                    .position(|value| value.re.is_nan() || value.im.is_nan())
            } else {
                None
            };
            if let Some(element) = element {
                return Err(ProgramError::custom(KernelInterpretationError::Nan {
                    operation,
                    position,
                    value,
                    element,
                    data_type,
                    coordinate: self.coordinate.clone(),
                    provenance: self.provenance.current(),
                }));
            }
        }
        Ok(())
    }

    /// Projects a tile's validity through the same canonical transforms as the accessed reference. This includes
    /// indexed axes and nested slices; the mask shares geometry but owns independent immutable Boolean storage.
    fn window_validity(&self, reference: &ArrayReference<Array>) -> Result<Option<Array>, ProgramError> {
        let Some(validity) = self.validity.get(&reference.id()) else {
            return Ok(None);
        };
        let mut selected = validity.clone();
        for view in reference.view().views() {
            selected = selected.with_transform(view.clone())?;
        }
        selected.read().map(Some)
    }
}

impl<Extension: Operation<Type = ArrayIrType>> Domain for QualifiedKernelContext<Extension> {
    type Type = ArrayIrType;
    type Value = ArrayIrValue<Array>;
    type Constant = ArrayIrValue<Array>;
    type Operation = KernelOperation<Extension>;
}

impl<Extension> Context for QualifiedKernelContext<Extension>
where
    Extension: KernelExtension + InterpretableOperation<EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>>,
{
    fn lift(&self, constant: Self::Constant) -> Result<Self::Value, ProgramError> {
        Ok(constant)
    }

    fn bind<O: Into<Self::Operation>, D: BindingRegionDriver<Self::Constant, Self::Operation>>(
        &self,
        operation: O,
        driver: D,
        inputs: &[Self::Value],
    ) -> Result<Vec<Self::Value>, ProgramError> {
        let operation = operation.into();
        if let Some(trace) = &self.trace {
            let mut trace = trace.borrow_mut();
            let mut accesses = Vec::new();
            for (input, mode) in operation.effects().accesses() {
                if let Some(ArrayIrValue::Reference(reference)) = inputs.get(input) {
                    let next = trace.roots.len();
                    let root = *trace.roots.entry(reference.id()).or_insert(next);
                    let view = reference.view().views().map(|view| format!("/{view:?}")).collect();
                    accesses.push(KernelTraceAccess { input, mode, root, view });
                }
            }
            let mut pending_copies = Vec::new();
            for token in self.pending_copies.borrow().keys() {
                let next = trace.roots.len();
                pending_copies.push(*trace.roots.entry(*token).or_insert(next));
            }
            pending_copies.sort_unstable();
            trace.entries.push(KernelTraceEntry {
                coordinate: self.coordinate.clone(),
                operation: operation.name(),
                provenance: self.provenance.current(),
                accesses,
                pending_copies,
            });
        }
        self.check_step(operation.name())?;
        self.check_nans(operation.name(), "input", inputs)?;
        let outputs = (|| {
            operation.validate_region_count(driver.region_count())?;
            if let KernelOperation::Scratch(scratch) = &operation {
                scratch.infer_output_types(
                    &inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
                    &[],
                )?;
                let storage = EagerContext::<Array, ArrayOperation<Array>>::new().zero(scratch.referent())?;
                return Ok(vec![ArrayIrValue::Reference(ArrayReference::new(storage))]);
            }
            let mask_index = match &operation {
                KernelOperation::MaskedLoad(_) => Some(1),
                KernelOperation::MaskedStore(_) | KernelOperation::MaskedSwap(_) => Some(2),
                _ => None,
            };
            if let Some(mask_index) = mask_index {
                operation.infer_output_types(
                    &inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
                    &driver.regions().map(|region| region.interface()).collect::<Vec<_>>(),
                )?;
                let ArrayIrValue::Reference(reference) = &inputs[0] else { unreachable!() };
                if let Some(validity) = self.window_validity(reference)? {
                    let ArrayIrValue::Array(mask) = &inputs[mask_index] else { unreachable!() };
                    let lanes = mask
                        .elements::<bool>()?
                        .into_iter()
                        .zip(validity.elements::<bool>()?)
                        .map(|(requested, valid)| requested && valid)
                        .collect::<Vec<_>>();
                    let mut inputs = inputs.to_vec();
                    inputs[mask_index] = ArrayIrValue::Array(Array::from_elements(mask.r#type().into_owned(), &lanes)?);
                    return operation.interpret(
                        &self.eager,
                        &QualifiedKernelDriver { context: self, driver: &driver, operation: operation.name() },
                        &inputs,
                    );
                }
            } else {
                // Writes may fill private padding because only valid coordinates are published. An unmasked read or
                // read-modify-write must never expose physical padding as initialized operand data.
                for (input_index, mode) in operation.effects().accesses() {
                    if mode != ReferenceAccessMode::Write
                        && let Some(ArrayIrValue::Reference(reference)) = inputs.get(input_index)
                        && let Some(validity) = self.window_validity(reference)?
                        && validity.elements::<bool>()?.iter().any(|valid| !valid)
                    {
                        return Err(ProgramError::custom(KernelMemoryError::UnmaskedWindowAccess {
                            operation: operation.name(),
                        }));
                    }
                }
            }
            match &operation {
                KernelOperation::AsyncCopy(_) => {
                    operation.infer_output_types(
                        &inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
                        &[],
                    )?;
                    let ArrayIrValue::Reference(source) = &inputs[0] else { unreachable!() };
                    let ArrayIrValue::Reference(destination) = &inputs[1] else { unreachable!() };
                    let source = source.read()?;
                    self.check_nans(operation.name(), "input", &[ArrayIrValue::Array(source.clone())])?;
                    let token = ArrayReference::new(Array::new(ArrayType::scalar(DataType::Token), vec![])?);
                    self.pending_copies
                        .borrow_mut()
                        .insert(token.id(), PendingCopy { source, destination: destination.clone() });
                    Ok(vec![ArrayIrValue::Reference(token)])
                }
                KernelOperation::Wait(_) => {
                    operation.infer_output_types(
                        &inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
                        &[],
                    )?;
                    let ArrayIrValue::Reference(token) = &inputs[0] else { unreachable!() };
                    let pending = self.pending_copies.borrow_mut().remove(&token.id()).ok_or_else(|| {
                        ProgramError::custom(KernelInterpretationError::UnknownCopyToken { token: token.id() })
                    })?;
                    let value = Array::from_logical_bytes(
                        pending.destination.r#type().referent().clone(),
                        &pending.source.logical_bytes(),
                    )?;
                    pending.destination.write(value)?;
                    token.freeze()?;
                    Ok(vec![])
                }
                _ => operation.interpret(
                    &self.eager,
                    &QualifiedKernelDriver { context: self, driver: &driver, operation: operation.name() },
                    inputs,
                ),
            }
        })()?;
        self.check_nans(operation.name(), "output", &outputs)?;
        Ok(outputs)
    }

    fn is_eager(&self) -> bool {
        true
    }

    fn provenance(&self) -> Provenance {
        self.provenance.current()
    }

    fn resolve(&self, value: &Self::Value) -> ValueResolution<Self::Constant> {
        ValueResolution::Constant(value.clone())
    }

    fn invoke_with_provenance_origin<R, F: FnOnce() -> R>(&self, origin: Provenance, function: F) -> R {
        self.provenance.invoke_with_origin(origin, function)
    }

    fn invoke_with_provenance_scope<R, F: FnOnce() -> R>(&self, scope: ProvenanceScope, function: F) -> R {
        self.provenance.invoke_with_scope(scope, function)
    }
}

/// Adapts unchanged eager higher-order rules to recursive replay in the private qualified context.
struct QualifiedKernelDriver<'r, Extension: Operation<Type = ArrayIrType>, D> {
    /// Qualification authority for this immutable invocation body.
    context: &'r QualifiedKernelContext<Extension>,
    /// Actual regions attached to the operation currently being interpreted.
    driver: &'r D,
    /// Canonical operation whose region entry consumes a replay step.
    operation: &'static str,
}

impl<Extension: Operation<Type = ArrayIrType>, D: RegionDriver<ArrayIrValue<Array>, KernelOperation<Extension>>>
    RegionDriver<ArrayIrValue<Array>, KernelOperation<Extension>> for QualifiedKernelDriver<'_, Extension, D>
{
    fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, ArrayIrValue<Array>, KernelOperation<Extension>>>
    where
        Extension: 'r,
    {
        self.driver.regions()
    }
}

impl<Extension, D> InterpretationDriver<EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>>
    for QualifiedKernelDriver<'_, Extension, D>
where
    Extension: KernelExtension + InterpretableOperation<EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>>,
    D: RegionDriver<ArrayIrValue<Array>, KernelOperation<Extension>>,
{
    fn interpret_region(
        &self,
        _context: &EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>,
        index: usize,
        inputs: Vec<ArrayIrValue<Array>>,
    ) -> Result<Vec<ArrayIrValue<Array>>, ProgramError> {
        self.context.check_step(self.operation)?;
        self.region(index)?.interpret_in_context(self.context, inputs)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayIrOperation, ArrayType, DataType, Dimension, DimensionBounds, DimensionType, Layout, Memory, StridedLayout,
    };
    use crate::contexts::Context;
    use crate::kernels::authoring::whole_array_parameter;
    use crate::kernels::calls::KernelParameter;
    use crate::kernels::grids::{Grid, GridDimension, GridExecution};
    use crate::kernels::mappings::{BlockMapping, BoundaryPolicy};
    use crate::kernels::memory::{
        AsyncCopyOperation, MaskedLoadOperation, MaskedStoreOperation, MaskedSwapOperation, WaitOperation,
    };
    use crate::kernels::operations::NoKernelExtension;
    use crate::operations::attention::{
        AttentionConfiguration, AttentionImplementation, AttentionOperandSignature, DotProductAttentionOperation,
    };
    use crate::operations::{
        AddOperation, ConditionOperation, DimensionMulOperation, DimensionToScalarOperation, ReduceOperation,
        ReductionKind, ReferenceIndexOperation, ReferenceRead, ReferenceWrite, ScaledDotOperation, WhileOperation,
    };
    use crate::parameters::Placeholder;
    use crate::programs::ProgramBuilder;
    use crate::tracing::TracingContext;

    use super::*;

    /// Builds a copy or addition definition with one-element blocks over a scalar or vector.
    fn definition(vector: bool, addition: bool) -> KernelDefinition {
        let mut mapping = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let outputs = if vector {
            vec![mapping.add_input(ArrayIrType::Dimension(DimensionType::new(
                "coordinate",
                DimensionBounds::non_negative(None).unwrap(),
            )))]
        } else {
            vec![]
        };
        let mapping = BlockMapping::new(
            mapping
                .build(outputs, vec![Placeholder; usize::from(vector)], vec![Placeholder; usize::from(vector)])
                .unwrap(),
            if vector { vec![1] } else { vec![] },
            BoundaryPolicy::InBounds,
        )
        .unwrap();
        let r#type = if vector { ArrayType::new_static(DataType::I32, [4]) } else { ArrayType::scalar(DataType::I32) };
        let mut parameters =
            vec![KernelParameter::new(r#type.clone(), KernelParameterAccess::ReadOnly, mapping.clone()).unwrap()];
        if addition {
            parameters
                .push(KernelParameter::new(r#type.clone(), KernelParameterAccess::ReadOnly, mapping.clone()).unwrap());
        }
        parameters.push(KernelParameter::new(r#type, KernelParameterAccess::WriteOnly, mapping).unwrap());
        let operation = KernelCallOperation::new(
            Grid::new(if vector {
                vec![GridDimension::new(Dimension::Static(4), GridExecution::Parallel)]
            } else {
                vec![]
            })
            .unwrap(),
            parameters,
        )
        .unwrap();
        KernelDefinition::trace(operation, |(references, _coordinates)| {
            let mut value = references[0].read()?;
            if addition {
                let second = references[1].read()?;
                value = references[0]
                    .context()
                    .bind(ArrayIrOperation::from(ArrayOperation::Add(AddOperation::new())), vec![], &[value, second])?
                    .remove(0);
            }
            references.last().unwrap().write(&value)
        })
        .unwrap()
    }

    /// Builds ordinary dimension multiplication for independent row-major tile coordinates.
    fn tiled_mapping(block_shape: Vec<usize>) -> BlockMapping {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let mut starts = Vec::new();
        for &block in &block_shape {
            let coordinate_type = DimensionType::new("coordinate", DimensionBounds::non_negative(None).unwrap());
            let coordinate = builder.add_input(coordinate_type.clone().into());
            let size = DimensionValue::constant(block).unwrap();
            let size_id = builder.add_constant(ArrayIrValue::Dimension(size.clone()));
            starts.push(
                builder
                    .add_instruction(
                        DimensionMulOperation::new(&coordinate_type, size.r#type().as_ref()).unwrap(),
                        vec![],
                        vec![coordinate, size_id],
                        None,
                    )
                    .unwrap()[0],
            );
        }
        let rank = block_shape.len();
        BlockMapping::new(
            builder.build(starts, vec![Placeholder; rank], vec![Placeholder; rank]).unwrap(),
            block_shape,
            BoundaryPolicy::Masked,
        )
        .unwrap()
    }

    /// Wraps canonical value operations with qualified kernel input/output references for conformance tests.
    fn value_definition(operation: ArrayOperation<Array>, input_types: &[ArrayType]) -> KernelDefinition {
        let output_types = operation.infer_output_types(input_types, &[]).unwrap();
        let mut parameters = input_types
            .iter()
            .map(|r#type| whole_array_parameter(r#type.clone(), KernelParameterAccess::ReadOnly).unwrap())
            .collect::<Vec<_>>();
        parameters.extend(
            output_types
                .into_iter()
                .map(|r#type| whole_array_parameter(r#type, KernelParameterAccess::WriteOnly).unwrap()),
        );
        let call = KernelCallOperation::new(Grid::new(vec![]).unwrap(), parameters).unwrap();
        KernelDefinition::trace(call, |(references, _coordinates)| {
            let inputs = references[..input_types.len()]
                .iter()
                .map(|reference| reference.read())
                .collect::<Result<Vec<_>, _>>()?;
            let outputs = references[0].context().bind(ArrayIrOperation::Array(operation), vec![], &inputs)?;
            for (reference, output) in references[input_types.len()..].iter().zip(outputs) {
                reference.write(&output)?;
            }
            Ok(())
        })
        .unwrap()
    }

    #[test]
    fn test_kernel_debug_options_default() {
        assert_eq!(
            KernelDebugOptions::default(),
            KernelDebugOptions {
                maximum_programs: DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS,
                maximum_steps: DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_STEPS,
                check_nans: false,
            },
        );
    }

    #[test]
    fn test_kernel_trace_entry_display() {
        let entry = KernelTraceEntry {
            coordinate: vec![2, 3],
            operation: "reference_read",
            provenance: Provenance::default(),
            accesses: vec![KernelTraceAccess {
                input: 0,
                mode: ReferenceAccessMode::Read,
                root: 1,
                view: "/slice[0:2]".to_owned(),
            }],
            pending_copies: vec![4],
        };
        assert_eq!(entry.to_string(), "[2, 3] reference_read input[0]=read(root[1]/slice[0:2]) pending=[4]");
    }

    #[test]
    fn test_kernel_definition_interpret() {
        let input = Array::scalar(17i32).unwrap();
        assert_eq!(definition(false, false).interpret(vec![input.clone()], 1), Ok(vec![input.clone()]));
        assert_eq!(input, Array::scalar(17i32).unwrap());
    }

    #[test]
    fn test_kernel_definition_interpret_reduction() {
        let input = Array::matrix(2, 3, vec![1i32, 2, 3, 4, 5, 6]).unwrap();
        let definition = value_definition(
            ArrayOperation::Reduce(ReduceOperation::new(vec![1], ReductionKind::Sum)),
            &[input.r#type().into_owned()],
        );
        assert_eq!(definition.interpret(vec![input], 1), Ok(vec![Array::vector(vec![6i32, 15]).unwrap()]));
        assert_eq!(definition.simplified().unwrap().operation().output_types(), definition.operation().output_types(),);
    }

    #[test]
    fn test_kernel_definition_interpret_scaled_dot() {
        let inputs = vec![
            Array::matrix(1, 4, vec![1f32, 2., 3., 4.]).unwrap(),
            Array::matrix(4, 1, vec![1f32, 1., 1., 1.]).unwrap(),
            Array::matrix(1, 2, vec![2f32, 4.]).unwrap(),
            Array::matrix(2, 1, vec![1f32, 2.]).unwrap(),
        ];
        let definition = value_definition(
            ArrayOperation::ScaledDot(ScaledDotOperation::new(
                ScaledDotOperation::default_dimensions(2).unwrap(),
                DataType::F32,
                true,
                true,
            )),
            &inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
        );
        let expected = vec![Array::matrix(1, 1, vec![62f32]).unwrap()];
        assert_eq!(definition.interpret(inputs.clone(), 1), Ok(expected.clone()));
        assert_eq!(definition.simplified().unwrap().interpret(inputs, 1), Ok(expected));
    }

    #[test]
    fn test_kernel_definition_interpret_attention() {
        // Zero logits give an exact uniform distribution, providing an independent arithmetic oracle.
        let inputs = vec![
            Array::from_elements(ArrayType::new_static(DataType::F32, [1, 1, 2]), &[0f32, 0.]).unwrap(),
            Array::from_elements(ArrayType::new_static(DataType::F32, [2, 1, 2]), &[0f32, 0., 0., 0.]).unwrap(),
            Array::from_elements(ArrayType::new_static(DataType::F32, [2, 1, 2]), &[2f32, 4., 4., 6.]).unwrap(),
        ];
        let definition = value_definition(
            ArrayOperation::DotProductAttention(DotProductAttentionOperation::new(
                AttentionConfiguration::new().with_implementation(AttentionImplementation::Portable),
                AttentionOperandSignature::new(false, false, false, false),
            )),
            &inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
        );
        let expected =
            vec![Array::from_elements(ArrayType::new_static(DataType::F32, [1, 1, 2]), &[3f32, 5.]).unwrap()];
        assert_eq!(definition.interpret(inputs.clone(), 1), Ok(expected.clone()));
        assert_eq!(definition.simplified().unwrap().interpret(inputs, 1), Ok(expected));
    }

    #[test]
    fn test_kernel_definition_interpret_vector_add() {
        assert_eq!(
            definition(true, true).interpret(
                vec![Array::vector(vec![1i32, 2, 3, 4]).unwrap(), Array::vector(vec![5i32, 6, 7, 8]).unwrap()],
                4,
            ),
            Ok(vec![Array::vector(vec![6i32, 8, 10, 12]).unwrap()]),
        );
    }

    #[test]
    fn test_kernel_definition_interpret_coordinates() {
        let mut mapping = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let coordinate = mapping.add_input(ArrayIrType::Dimension(DimensionType::new(
            "coordinate",
            DimensionBounds::non_negative(None).unwrap(),
        )));
        let mapping = BlockMapping::new(
            mapping.build(vec![coordinate], vec![Placeholder], vec![Placeholder]).unwrap(),
            vec![1],
            BoundaryPolicy::InBounds,
        )
        .unwrap();
        let operation = KernelCallOperation::new(
            Grid::new(vec![GridDimension::new(Dimension::Static(4), GridExecution::Parallel)]).unwrap(),
            vec![
                KernelParameter::new(
                    ArrayType::new_static(DataType::I64, [4]),
                    KernelParameterAccess::WriteOnly,
                    mapping,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let definition = KernelDefinition::<NoKernelExtension>::trace(operation, |(references, coordinates)| {
            let context = coordinates[0].context();
            let value = context
                .bind(ArrayIrOperation::DimensionToScalar(DimensionToScalarOperation), vec![], &coordinates)?
                .remove(0);
            let output = context.bind(ReferenceIndexOperation::new(0, 0), vec![], &references)?.remove(0);
            output.write(&value)
        })
        .unwrap();
        assert_eq!(definition.interpret(vec![], 4), Ok(vec![Array::vector(vec![0i64, 1, 2, 3]).unwrap()]));
    }

    #[test]
    fn test_kernel_definition_interpret_budget() {
        let error = definition(false, false).interpret(vec![Array::scalar(17i32).unwrap()], 0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelInitializationError>(),
            Some(&KernelInitializationError::QualificationLimit { programs: 1, maximum: 0 }),
        );
    }

    #[test]
    fn test_kernel_definition_interpret_budget_applies_to_tiled_proofs() {
        let error =
            definition(true, false).interpret(vec![Array::vector(vec![1i32, 2, 3, 4]).unwrap()], 3).unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelInitializationError>(),
            Some(&KernelInitializationError::QualificationLimit { programs: 4, maximum: 3 }),
        );
    }

    #[test]
    fn test_kernel_definition_interpret_scratch() {
        let operation = definition(false, false).operation().clone();
        let definition = KernelDefinition::<NoKernelExtension>::trace(operation, |(references, _coordinates)| {
            let scratch = references[0]
                .context()
                .bind(ScratchOperation::new(ArrayType::scalar(DataType::I32), 4).unwrap(), vec![], &[])?
                .remove(0);
            scratch.write(&references[0].read()?)?;
            references[1].write(&scratch.read()?)
        })
        .unwrap();
        assert_eq!(
            definition.interpret(vec![Array::scalar(23i32).unwrap()], 1),
            Ok(vec![Array::scalar(23i32).unwrap()]),
        );
    }

    #[test]
    fn test_kernel_definition_interpret_nested_scratch() {
        let operation = definition(false, false).operation().clone();
        let branch = KernelDefinition::<NoKernelExtension>::trace(operation.clone(), |(references, _coordinates)| {
            let scratch = references[0]
                .context()
                .bind(ScratchOperation::new(ArrayType::scalar(DataType::I32), 4).unwrap(), vec![], &[])?
                .remove(0);
            scratch.write(&references[0].read()?)?;
            references[1].write(&scratch.read()?)
        })
        .unwrap();
        let definition = KernelDefinition::<NoKernelExtension>::trace(operation, |(references, _coordinates)| {
            let context = references[0].context();
            let predicate = context.lift(ArrayIrValue::Array(Array::scalar(true).unwrap()))?;
            context.bind(
                ArrayIrOperation::Condition(ConditionOperation::new()),
                vec![branch.body().clone(), branch.body().clone()],
                &[predicate, references[0].clone(), references[1].clone()],
            )?;
            Ok(())
        })
        .unwrap();
        assert_eq!(
            definition.interpret(vec![Array::scalar(31i32).unwrap()], 1),
            Ok(vec![Array::scalar(31i32).unwrap()]),
        );
    }

    #[test]
    fn test_kernel_definition_interpret_async_copy() {
        let mut mapping = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let zero = mapping.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(0).unwrap()));
        let mapping = BlockMapping::new(
            mapping.build(vec![zero], vec![], vec![Placeholder]).unwrap(),
            vec![2],
            BoundaryPolicy::InBounds,
        )
        .unwrap();
        let input_type = ArrayType::new_static(DataType::I32, [2]);
        let output_type = input_type
            .clone()
            .with_memory(Memory::Host { pinned: false })
            .with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                KernelParameter::new(input_type, KernelParameterAccess::ReadOnly, mapping.clone()).unwrap(),
                KernelParameter::new(output_type.clone(), KernelParameterAccess::WriteOnly, mapping).unwrap(),
            ],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            let context = references[0].context();
            let token =
                context.bind(AsyncCopyOperation, vec![], &[references[0].clone(), references[1].clone()])?.remove(0);
            context.bind(WaitOperation, vec![], &[token])?;
            Ok(())
        })
        .unwrap();
        let input = Array::vector(vec![17i32, 29]).unwrap();
        let expected = vec![Array::from_elements(output_type, &[17i32, 29]).unwrap()];
        assert_eq!(definition.interpret(vec![input.clone()], 1), Ok(expected.clone()));
        let mut trace = Vec::new();
        assert_eq!(
            definition.interpret_with_trace(vec![input.clone()], &KernelDebugOptions::default(), &mut trace),
            Ok(expected),
        );
        assert_eq!(trace.iter().map(|entry| entry.operation).collect::<Vec<_>>(), vec!["async_copy", "wait"]);
        assert_eq!(trace[0].pending_copies, Vec::<usize>::new());
        assert_eq!(trace[1].pending_copies, vec![2]);
        assert_eq!(trace[1].accesses[0].mode, ReferenceAccessMode::Consume);
        assert_eq!(trace[1].accesses[0].root, 2);
        assert_eq!(input, Array::vector(vec![17i32, 29]).unwrap());
    }

    #[test]
    fn test_kernel_definition_interpret_masked_edges() {
        let mapping = tiled_mapping(vec![2, 2]);
        let r#type = ArrayType::new_static(DataType::I32, [5, 3]);
        let operation = KernelCallOperation::new(
            Grid::new(vec![
                GridDimension::new(Dimension::Static(3), GridExecution::Parallel),
                GridDimension::new(Dimension::Static(2), GridExecution::Parallel),
            ])
            .unwrap(),
            vec![
                KernelParameter::new(r#type.clone(), KernelParameterAccess::ReadOnly, mapping.clone()).unwrap(),
                KernelParameter::new(r#type.clone(), KernelParameterAccess::WriteOnly, mapping).unwrap(),
            ],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            let context = references[0].context();
            let mask = context.lift(ArrayIrValue::Array(Array::matrix(2, 2, vec![true; 4])?))?;
            let other = context.lift(ArrayIrValue::Array(Array::matrix(2, 2, vec![-7i32; 4])?))?;
            let value =
                context.bind(MaskedLoadOperation, vec![], &[references[0].clone(), mask.clone(), other])?.remove(0);
            context.bind(MaskedStoreOperation, vec![], &[references[1].clone(), value, mask])?;
            Ok(())
        })
        .unwrap();
        let input = Array::from_elements(r#type, &(0..15i32).collect::<Vec<_>>()).unwrap();
        assert_eq!(definition.interpret(vec![input.clone()], 6), Ok(vec![input.clone()]));
        assert_eq!(input.elements::<i32>().unwrap(), (0..15i32).collect::<Vec<_>>());
    }

    #[test]
    fn test_kernel_definition_interpret_masked_other() {
        let mapping = tiled_mapping(vec![4]);
        let operation = KernelCallOperation::new(
            Grid::new(vec![GridDimension::new(Dimension::Static(1), GridExecution::Parallel)]).unwrap(),
            vec![
                KernelParameter::new(
                    ArrayType::new_static(DataType::I32, [2]),
                    KernelParameterAccess::ReadOnly,
                    mapping.clone(),
                )
                .unwrap(),
                KernelParameter::new(
                    ArrayType::new_static(DataType::I32, [4]),
                    KernelParameterAccess::WriteOnly,
                    mapping,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            let context = references[0].context();
            let mask = context.lift(ArrayIrValue::Array(Array::vector(vec![true, false, true, true])?))?;
            let other = context.lift(ArrayIrValue::Array(Array::vector(vec![7i32; 4])?))?;
            let value = context.bind(MaskedLoadOperation, vec![], &[references[0].clone(), mask, other])?.remove(0);
            references[1].write(&value)
        })
        .unwrap();
        assert_eq!(
            definition.interpret(vec![Array::vector(vec![1i32, 2]).unwrap()], 1),
            Ok(vec![Array::vector(vec![1i32, 7, 7, 7]).unwrap()]),
        );
    }

    #[test]
    fn test_kernel_definition_interpret_masked_swap_preserves_inactive_storage() {
        let mapping = tiled_mapping(vec![4]);
        let r#type = ArrayType::new_static(DataType::I32, [5]);
        let operation = KernelCallOperation::new(
            Grid::new(vec![GridDimension::new(Dimension::Static(2), GridExecution::Parallel)]).unwrap(),
            vec![
                KernelParameter::new(r#type.clone(), KernelParameterAccess::ReadWrite, mapping.clone()).unwrap(),
                KernelParameter::new(r#type, KernelParameterAccess::WriteOnly, mapping).unwrap(),
            ],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            let context = references[0].context();
            let mask = context.lift(ArrayIrValue::Array(Array::vector(vec![true, false, true, true])?))?;
            let replacement = context.lift(ArrayIrValue::Array(Array::vector(vec![9i32, 10, 11, 12])?))?;
            let other = context.lift(ArrayIrValue::Array(Array::vector(vec![-1i32; 4])?))?;
            let old = context
                .bind(MaskedSwapOperation, vec![], &[references[0].clone(), replacement, mask, other])?
                .remove(0);
            references[1].write(&old)
        })
        .unwrap();
        let input = Array::vector(vec![1i32, 2, 3, 4, 5]).unwrap();
        let expected =
            vec![Array::vector(vec![9i32, 2, 11, 12, 9]).unwrap(), Array::vector(vec![1i32, -1, 3, 4, 5]).unwrap()];
        assert_eq!(definition.interpret(vec![input.clone()], 2), Ok(expected));
        assert_eq!(input.elements::<i32>().unwrap(), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_kernel_definition_interpret_with_limits() {
        let definition = definition(false, false);
        let input = Array::scalar(17i32).unwrap();
        assert_eq!(definition.interpret_with_limits(vec![input.clone()], 1, 3), Ok(vec![input.clone()]));
        let error = definition.interpret_with_limits(vec![input.clone()], 1, 2).unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelInterpretationError>(),
            Some(&KernelInterpretationError::StepLimit {
                maximum_steps: 2,
                coordinate: vec![],
                operation: "reference_write",
                provenance: Provenance::unknown(),
            }),
        );
        assert_eq!(error.to_string(), "kernel interpretation exceeded 2 steps at grid point [] in `reference_write`");
        assert_eq!(input, Array::scalar(17i32).unwrap());
    }

    #[test]
    fn test_kernel_definition_interpret_with_limits_shares_grid_budget() {
        let error = definition(true, false)
            .interpret_with_limits(vec![Array::vector(vec![1i32, 2, 3, 4]).unwrap()], 4, 5)
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelInterpretationError>(),
            Some(&KernelInterpretationError::StepLimit {
                maximum_steps: 5,
                coordinate: vec![1],
                operation: "reference_write",
                provenance: Provenance::unknown(),
            }),
        );
    }

    #[test]
    fn test_kernel_definition_interpret_with_limits_preserves_provenance() {
        let operation = definition(false, false).operation().clone();
        let scope = ProvenanceScope::new("copy output");
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            let value = references[0].read()?;
            references[1].context().invoke_with_provenance_scope(scope.clone(), || references[1].write(&value))
        })
        .unwrap();
        let error = definition.interpret_with_limits(vec![Array::scalar(3i32).unwrap()], 1, 2).unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelInterpretationError>(),
            Some(&KernelInterpretationError::StepLimit {
                maximum_steps: 2,
                coordinate: vec![],
                operation: "reference_write",
                provenance: Provenance::scope(scope, Provenance::unknown()),
            }),
        );
    }

    #[test]
    fn test_kernel_definition_interpret_with_limits_counts_empty_loop_regions() {
        let mut condition = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let predicate = condition.add_constant(ArrayIrValue::Array(Array::scalar(true).unwrap()));
        let condition = condition
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![predicate], vec![], vec![Placeholder])
            .unwrap();
        let loop_body = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![], vec![])
            .unwrap();
        let mut body = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let condition = body.import_region(condition.entry_region_ref());
        let loop_body = body.import_region(loop_body.entry_region_ref());
        body.add_instruction(
            ArrayIrOperation::While(WhileOperation::new().with_iteration_bound(100).unwrap()),
            vec![condition, loop_body],
            vec![],
            None,
        )
        .unwrap();
        let definition = KernelDefinition::new(
            KernelCallOperation::new(Grid::new(vec![]).unwrap(), vec![]).unwrap(),
            body.build(vec![], vec![], vec![]).unwrap(),
        )
        .unwrap();
        let error = definition.interpret_with_limits(vec![], 1, 4).unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelInterpretationError>(),
            Some(&KernelInterpretationError::StepLimit {
                maximum_steps: 4,
                coordinate: vec![],
                operation: "while",
                provenance: Provenance::unknown(),
            }),
        );
    }

    #[test]
    fn test_kernel_definition_interpret_with_trace() {
        let definition = definition(false, false);
        let input = Array::scalar(17i32).unwrap();
        let mut trace = Vec::new();
        assert_eq!(
            definition.interpret_with_trace(
                vec![input.clone()],
                &KernelDebugOptions { maximum_programs: 1, maximum_steps: 3, check_nans: false },
                &mut trace,
            ),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            trace.iter().map(|entry| entry.operation).collect::<Vec<_>>(),
            vec!["reference_read", "reference_write"],
        );
        assert_eq!(
            trace.iter().map(|entry| entry.coordinate.clone()).collect::<Vec<_>>(),
            vec![Vec::<usize>::new(), vec![]],
        );
        assert_eq!(trace[0].accesses[0].mode, ReferenceAccessMode::Read);
        assert_eq!(trace[0].accesses[0].root, 0);
        assert_eq!(trace[1].accesses[0].mode, ReferenceAccessMode::Write);
        assert_eq!(trace[1].accesses[0].root, 1);
        assert_eq!(trace[0].pending_copies, Vec::<usize>::new());
        let first = trace.clone();
        assert_eq!(
            definition.interpret_with_trace(
                vec![input.clone()],
                &KernelDebugOptions { maximum_programs: 1, maximum_steps: 3, check_nans: false },
                &mut trace,
            ),
            Ok(vec![input]),
        );
        assert_eq!(trace, first);
    }

    #[test]
    fn test_kernel_definition_interpret_with_trace_nan() {
        let definition = value_definition(
            ArrayOperation::Add(AddOperation::new()),
            &[ArrayType::scalar(DataType::F32), ArrayType::scalar(DataType::F32)],
        );
        let inputs = vec![Array::scalar(f32::INFINITY).unwrap(), Array::scalar(f32::NEG_INFINITY).unwrap()];
        let mut trace = Vec::new();
        let options = KernelDebugOptions { check_nans: true, ..Default::default() };
        let error = definition.interpret_with_trace(inputs.clone(), &options, &mut trace).unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelInterpretationError>(),
            Some(&KernelInterpretationError::Nan {
                operation: "add",
                position: "output",
                value: 0,
                element: 0,
                data_type: DataType::F32,
                coordinate: vec![],
                provenance: Provenance::unknown(),
            }),
        );
        assert_eq!(error.to_string(), "nan in `add` output 0 element 0 of type `f32` at grid point []");
        assert_eq!(trace.last().unwrap().operation, "add");
        let output = definition.interpret_with_trace(inputs, &KernelDebugOptions::default(), &mut trace).unwrap();
        assert!(output[0].elements::<f32>().unwrap()[0].is_nan());
        assert_eq!(trace.last().unwrap().operation, "reference_write");
    }

    #[test]
    fn test_kernel_definition_interpret_with_trace_nan_input_formats() {
        for input in [
            Array::new(ArrayType::scalar(DataType::F8E4M3FN), vec![0x7f]).unwrap(),
            Array::scalar(Complex::new(0f64, f64::NAN)).unwrap(),
        ] {
            let data_type = input.r#type().data_type();
            let definition = value_definition(
                ArrayOperation::Add(AddOperation::new()),
                &[input.r#type().into_owned(), input.r#type().into_owned()],
            );
            let mut trace = Vec::new();
            let error = definition
                .interpret_with_trace(
                    vec![input.clone(), input],
                    &KernelDebugOptions { check_nans: true, ..Default::default() },
                    &mut trace,
                )
                .unwrap_err();
            assert_eq!(
                error.downcast_custom::<KernelInterpretationError>(),
                Some(&KernelInterpretationError::Nan {
                    operation: "reference_read",
                    position: "output",
                    value: 0,
                    element: 0,
                    data_type,
                    coordinate: vec![],
                    provenance: Provenance::unknown(),
                }),
            );
            assert_eq!(trace.len(), 1);
        }
    }

    #[test]
    fn test_kernel_definition_interpret_with_trace_failure() {
        let definition = definition(false, false);
        let input = Array::scalar(17i32).unwrap();
        let mut trace = Vec::new();
        let error = definition
            .interpret_with_trace(
                vec![input.clone()],
                &KernelDebugOptions { maximum_programs: 1, maximum_steps: 2, check_nans: false },
                &mut trace,
            )
            .unwrap_err();
        assert_eq!(error.to_string(), "kernel interpretation exceeded 2 steps at grid point [] in `reference_write`");
        assert_eq!(
            trace.iter().map(|entry| entry.operation).collect::<Vec<_>>(),
            vec!["reference_read", "reference_write"],
        );
        let error = definition
            .interpret_with_trace(
                vec![input],
                &KernelDebugOptions { maximum_programs: 0, maximum_steps: 2, check_nans: false },
                &mut trace,
            )
            .unwrap_err();
        assert_eq!(error.to_string(), "kernel qualification requires 1 programs, exceeding the limit 0");
        assert_eq!(trace, Vec::<KernelTraceEntry>::new());
    }

    #[test]
    fn test_scratch_operation_interpret() {
        let context = EagerContext::<ArrayIrValue<Array>, KernelOperation>::new();
        let error = context
            .bind(ScratchOperation::new(ArrayType::scalar(DataType::I32), 4).unwrap(), vec![], &[])
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelMemoryError>(),
            Some(&KernelMemoryError::RequiresQualification { operation: "scratch" }),
        );
    }

    #[test]
    fn test_qualified_kernel_context_async_completion() {
        // Construct the private context directly to verify the runtime defense independently of static rejection.
        let context = QualifiedKernelContext::<NoKernelExtension> {
            eager: EagerContext::new(),
            validity: BTreeMap::new(),
            coordinate: vec![],
            maximum_steps: 10,
            remaining_steps: Rc::new(Cell::new(10)),
            provenance: Rc::new(ProvenanceState::new()),
            trace: None,
            pending_copies: Rc::new(RefCell::new(BTreeMap::new())),
        };
        let source = ArrayReference::new(Array::vector(vec![11i32, 13]).unwrap());
        let destination = ArrayReference::new(Array::vector(vec![0i32, 0]).unwrap());
        let token = context
            .bind(
                AsyncCopyOperation,
                vec![],
                &[ArrayIrValue::Reference(source), ArrayIrValue::Reference(destination.clone())],
            )
            .unwrap()
            .remove(0);
        assert_eq!(destination.read(), Ok(Array::vector(vec![0i32, 0]).unwrap()));
        assert_eq!(context.bind(WaitOperation, vec![], &[token.clone()]), Ok(vec![]));
        assert_eq!(destination.read(), Ok(Array::vector(vec![11i32, 13]).unwrap()));
        let ArrayIrValue::Reference(reference) = &token else { unreachable!() };
        let error = context.bind(WaitOperation, vec![], &[token.clone()]).unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelInterpretationError>(),
            Some(&KernelInterpretationError::UnknownCopyToken { token: reference.id() }),
        );
        assert_eq!(error.to_string(), "async copy completion token is unknown or already consumed");
    }

    #[test]
    fn test_kernel_call_operation_interpret() {
        let definition = definition(true, true);
        let (_, program) = TracingContext::<ArrayIrValue<Array>, KernelOperation>::trace(
            |inputs: Vec<_>| {
                inputs[0].context().bind(definition.operation().clone(), vec![definition.body().clone()], &inputs)
            },
            definition.operation().input_types(),
        )
        .unwrap();
        let inputs = vec![
            ArrayIrValue::Array(Array::vector(vec![1i32, 2, 3, 4]).unwrap()),
            ArrayIrValue::Array(Array::vector(vec![5i32, 6, 7, 8]).unwrap()),
        ];
        let expected = vec![ArrayIrValue::Array(Array::vector(vec![6i32, 8, 10, 12]).unwrap())];
        assert_eq!(program.interpret(inputs.clone()), Ok(expected.clone()));
        let (_, replayed) = TracingContext::<ArrayIrValue<Array>, KernelOperation>::trace(
            |inputs: Vec<_>| {
                let context = inputs[0].context().clone();
                program.interpret_in_context(&context, inputs)
            },
            definition.operation().input_types(),
        )
        .unwrap();
        assert_eq!(replayed.to_string(), program.to_string());
        assert_eq!(replayed.interpret(inputs), Ok(expected));
    }
}
