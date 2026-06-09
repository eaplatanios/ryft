//! User-facing XLA compile-and-execute API.
//!
//! [`compile`] is the `ryft` analogue of `jax.jit`: it traces a closure over staged
//! tracers into an XLA program, compiles it via PJRT, and returns a runtime handle that
//! executes the compiled program against [`Array`] inputs. The trace borrows the execution domain for the duration of
//! one trace and carries an active runtime-capture table, so staged compiled functions can register captures in the
//! same captured program without embedding runtime arrays in the IR. The resulting
//! [`Program`](ryft_core::programs::Program) is then compiled and executed via the supplied domain's internal cache.
//!
//! New backend-agnostic code should prefer the core pipeline at
//! [`ryft_core::compilation::compile_with_options`].

use std::cell::RefCell;
use std::rc::Rc;

use ryft_core::compilation::context::CapturingContext;
use ryft_core::compilation::{CapturedProgram, CompilationDomain, CompilationOptions, FunctionFingerprint};
use ryft_core::contexts::StagingContext;
use ryft_core::parameters::{ParameterError, Parameterized, ParameterizedFamily};
use ryft_core::programs::{ProgramBuilder, ProgramError};
use ryft_core::sharding::{DeviceMesh, Sharding};
use ryft_core::tracing::{DomainTracer, Tracer, TracingContext};
use ryft_core::tracing_v2::{BatchContext, DifferentiationContext, LinearizationContext};
use ryft_core::types::{ArrayType, Typed};

use crate::Array;
use crate::experimental::domains::{XlaCompiledProgram, XlaDomain, XlaDomainError, XlaOptions};
use crate::experimental::ops::{JitCallOperation, XlaConstant, XlaOperation};

/// Tracer leaf used while tracing an XLA compilation.
type XlaCompileTracer<'domain, 'c> = DomainTracer<'domain, XlaDomain<'c>>;

/// Captured-constant output tree produced by tracing an XLA closure.
type XlaSourceProgramOutput<Out> = <Out as Parameterized<ArrayType>>::To<XlaConstant>;

/// Tracer leaf used while linearizing a compiled XLA function inside a compile trace.
type XlaCompileLinearizationTracer<'domain, 'c> = Tracer<
    LinearizationContext<'domain, TracingContext<'domain, XlaDomain<'c>>, TracingContext<'domain, XlaDomain<'c>>>,
>;

/// Just-in-time compiled function handle. Returned by [`compile`] and
/// [`compile_with_options`].
///
/// Holds the cached PJRT-backed [`XlaCompiledProgram`] plus the input / output type metadata
/// needed to marshal a [`Parameterized`] tree of [`Array`]s into the executable and reassemble
/// the outputs back into the user's expected output tree shape.
///
/// Also retains the **source [`Program`]** that the execution domain compiled into [`Self::program`]. Useful for
/// diagnostics (printing the traced IR, instruction counts, graph rendering), for outer transforms, and for inner
/// staging via [`Self::stage`] with trace inputs, which emits a `jit_call` boundary carrying the source program into
/// the active outer trace context.
pub struct CompiledXlaFunction<
    'c,
    In: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<XlaConstant>>,
    Out: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<XlaConstant>>,
> {
    /// Compiled XLA program. Carries the loaded PJRT executable plus per-call state baked at
    /// compile time (output types, donation flags, expected input shardings, mesh).
    program: XlaCompiledProgram<'c>,

    /// Source [`Program`] that produced [`Self::program`]. Runtime [`Array`] buffers captured by the trace are stored
    /// in the surrounding capture table and supplied as hidden executable arguments at execution.
    source_program:
        CapturedProgram<ArrayType, Array<'c>, XlaOperation, In::To<XlaConstant>, XlaSourceProgramOutput<Out>>,

    /// PyTree shape of the output. Used by [`Self::interpret`] to reassemble the executor's flat
    /// output buffer list back into the user's expected output tree.
    output_structure: Out::ParameterStructure,

    /// XLA domain used to execute the compiled program. Cloned from the execution domain so
    /// the compiled function isn't tied to the context's borrow scope.
    domain: XlaDomain<'c>,
}

impl<
    'c,
    In: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<XlaConstant>>,
    Out: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<XlaConstant>>,
> Clone for CompiledXlaFunction<'c, In, Out>
{
    fn clone(&self) -> Self {
        Self {
            program: self.program.clone(),
            source_program: self.source_program.clone(),
            output_structure: self.output_structure.clone(),
            domain: self.domain.clone(),
        }
    }
}

impl<
    'c,
    In: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<XlaConstant>>,
    Out: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<XlaConstant>>,
> CompiledXlaFunction<'c, In, Out>
{
    /// Returns the flat output [`ArrayType`]s in the order the executor produces them.
    #[inline]
    pub fn output_types(&self) -> &[ArrayType] {
        self.program.output_types()
    }

    /// Returns the source [`Program`] that produced the compiled artifact. Useful for outer transforms (`grad` / `jvp`
    /// / `vjp` / `batch`), staged `jit_call` payloads, and diagnostics (printing the traced IR, instruction counts,
    /// graph rendering).
    #[inline]
    pub fn source_program(
        &self,
    ) -> &CapturedProgram<ArrayType, Array<'c>, XlaOperation, In::To<XlaConstant>, XlaSourceProgramOutput<Out>> {
        &self.source_program
    }

    /// Returns the device mesh the compiled program runs against. Delegates to the cached
    /// [`XlaCompiledProgram::mesh`](crate::experimental::domains::XlaCompiledProgram::mesh).
    #[inline]
    pub fn mesh(&self) -> &DeviceMesh {
        self.program.mesh()
    }

    /// Reconstructs the structured input parameter tree this function was compiled for, by
    /// reading each input atom's [`ArrayType`] from the retained source program and reassembling
    /// them under the program's `input_structure`. Used internally by transformed compiles.
    fn input_signature(&self) -> Result<In, ParameterError>
    where
        In: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType, To = In>, To<ArrayType> = In>,
    {
        let program = self.source_program.program();
        let structure = program.input_structure().clone();
        let atoms = program.input_ids().iter().map(|id| program.atoms()[id.index()].r#type().into_owned());
        In::To::<ArrayType>::from_parameters(structure, atoms)
    }

    /// Executes this compiled function on concrete [`Array`] inputs.
    ///
    /// This runs the cached compiled artifact and reassembles the executor's flat outputs under this function's
    /// output parameter structure.
    #[inline]
    pub fn interpret(&self, inputs: In::To<Array<'c>>) -> Result<Out::To<Array<'c>>, XlaDomainError>
    where
        In: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<Array<'c>>>,
        In::To<Array<'c>>: Parameterized<Array<'c>>,
        Out: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<Array<'c>>>,
        Out::To<Array<'c>>:
            Parameterized<Array<'c>, Family = Out::Family, ParameterStructure = Out::ParameterStructure>,
    {
        let inputs_vec: Vec<Array<'c>> = inputs.into_parameters().collect();
        let outputs = self.domain.execute_with_captures(&self.program, self.source_program.captures(), inputs_vec)?;
        Out::To::<Array<'c>>::from_parameters(self.output_structure.clone(), outputs)
            .map_err(|error| XlaDomainError::Array(error.into()))
    }

    /// Stages this compiled function into an active trace as a `jit_call` operation.
    ///
    /// This does not execute the compiled artifact. It records a trace boundary carrying this function's retained
    /// source program so enclosing transforms can rewrite the boundary through the ordinary XLA operation rules.
    #[inline]
    pub fn stage<
        C: StagingContext<Type = ArrayType, Constant = XlaConstant, Operation = XlaOperation>
            + CapturingContext<Array<'c>>,
    >(
        &self,
        inputs: In::To<Tracer<C>>,
    ) -> Out::To<Tracer<C>>
    where
        In: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<Tracer<C>>>,
        In::To<Tracer<C>>: Parameterized<Tracer<C>>,
        Out: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<Tracer<C>>>,
        Out::To<Tracer<C>>:
            Parameterized<Tracer<C>, Family = Out::Family, ParameterStructure = Out::ParameterStructure>,
    {
        let inputs = inputs.into_parameters().collect::<Vec<_>>();
        let context = inputs
            .first()
            .expect("staging a well-formed jitted call requires at least one input")
            .context()
            .clone();
        let capture_references = self
            .source_program
            .captures()
            .iter()
            .cloned()
            .map(|value| context.capture(value))
            .collect::<Result<Vec<_>, _>>()
            .expect("staging a well-formed jitted call should register captures in the outer trace");
        let outputs = self
            .stage_with_flat_capture_references(capture_references.as_slice(), inputs)
            .expect("staging a well-formed jitted call into a compatible outer trace should not fail");
        Out::To::<Tracer<C>>::from_parameters(self.output_structure.clone(), outputs)
            .expect("jitted call output structure should match the compiled function")
    }

    /// Stages this compiled function with flat explicit capture references.
    fn stage_with_flat_capture_references<
        C: StagingContext<Type = ArrayType, Constant = XlaConstant, Operation = XlaOperation>,
    >(
        &self,
        capture_references: &[XlaConstant],
        inputs: Vec<Tracer<C>>,
    ) -> Result<Vec<Tracer<C>>, ProgramError> {
        self.source_program.validate_capture_inputs(capture_references)?;
        let context = inputs
            .first()
            .map(|input| input.context().clone())
            .ok_or(ProgramError::InvalidInputCount { expected: 1, got: 0 })?;
        let capture_tracers = capture_references.iter().cloned().map(|capture| context.constant(capture));
        let full_inputs = capture_tracers.chain(inputs).collect::<Vec<_>>();
        context.stage_operation(
            XlaOperation::Extension(crate::experimental::ops::XlaOperationExtension::JitCall(Box::new(
                JitCallOperation::new(self.source_program.open_captures_as_inputs()?),
            ))),
            full_inputs.as_slice(),
        )
    }
}

/// Reverse-mode AD: compiles a new function that computes the gradient of a scalar-valued compiled function with
/// respect to its inputs. The original closure is never re-executed; [`Self::stage`] emits a `jit_call` boundary, and
/// the active transform rewrites that operation through ordinary JVP and transpose rules.
impl<
    'c,
    In: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType, To = In> + ParameterizedFamily<XlaConstant>,
            To<ArrayType> = In,
            ParameterStructure: std::fmt::Debug + std::hash::Hash + PartialEq,
        >,
> CompiledXlaFunction<'c, In, ArrayType>
{
    /// Returns a new compiled function that computes the reverse-mode gradient of `self` with
    /// respect to its input. Mirrors `jax.grad(jax.jit(f))`.
    ///
    /// `self` must produce a single rank-0 scalar output (encoded by the `Out = ArrayType`
    /// impl-block constraint above). The returned compiled function has the same input shape
    /// and produces an output whose leaves carry the partial derivative at each input leaf.
    #[track_caller]
    pub fn value_and_gradient<'domain>(&'domain self) -> Result<CompiledXlaFunction<'c, In, In>, XlaDomainError>
    where
        'c: 'domain,
        In::Family: ParameterizedFamily<XlaCompileTracer<'domain, 'c>>
            + ParameterizedFamily<XlaCompileLinearizationTracer<'domain, 'c>>,
        In::To<XlaCompileTracer<'domain, 'c>>: Parameterized<
                XlaCompileTracer<'domain, 'c>,
                To<XlaCompileTracer<'domain, 'c>> = In::To<XlaCompileTracer<'domain, 'c>>,
                To<ArrayType> = In,
            >,
        In::To<XlaCompileTracer<'domain, 'c>>: Parameterized<
                XlaCompileTracer<'domain, 'c>,
                To<XlaCompileLinearizationTracer<'domain, 'c>> = In::To<XlaCompileLinearizationTracer<'domain, 'c>>,
            >,
        In::To<XlaCompileLinearizationTracer<'domain, 'c>>: Parameterized<XlaCompileLinearizationTracer<'domain, 'c>>,
    {
        let function = self;
        let input_signature = function.input_signature().map_err(|error| XlaDomainError::Array(error.into()))?;
        let mesh = function.mesh().clone();
        let captures = function.source_program.captures().to_vec();
        compile_with_flat_captures(
            move |capture_references, _, tracers| {
                let context = tracers
                    .parameters()
                    .next()
                    .expect("compiled value_and_gradient requires at least one input")
                    .context()
                    .clone();
                context
                    .value_and_gradient(
                        move |y| {
                            let outputs = function
                                .stage_with_flat_capture_references(
                                    capture_references.as_slice(),
                                    y.into_parameters().collect(),
                                )
                                .expect("compiled value_and_gradient call staging should succeed");
                            let output: XlaCompileLinearizationTracer<'domain, 'c> =
                                Parameterized::from_parameters(function.output_structure.clone(), outputs)
                                    .expect("compiled value_and_gradient output reassembly should succeed");
                            output
                        },
                        tracers,
                    )
                    .unwrap()
            },
            captures,
            input_signature,
            &function.domain,
            CompilationOptions::new(XlaOptions::new(mesh)),
        )
    }
}

/// Forward-mode JVP packaged as a method. Mirrors `jax.jvp(jax.jit(f))`.
impl<
    'c,
    In: Clone
        + Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType, To = In> + ParameterizedFamily<XlaConstant>,
            To<ArrayType> = In,
            ParameterStructure: std::fmt::Debug + std::hash::Hash + PartialEq,
        >,
    Out: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<XlaConstant>>,
> CompiledXlaFunction<'c, In, Out>
{
    /// Returns a new compiled function that computes the forward-mode JVP of `self`. Mirrors
    /// `jax.jvp(f, primals, tangents)` packaged into one compiled function: the returned handle
    /// takes `(primals, tangents)` and returns `(primal_out, tangent_out)`.
    ///
    /// The implementation stages a `jit_call` operation and lets the ordinary JVP rule for that
    /// call operation build the tangent call boundary.
    #[track_caller]
    pub fn jvp<'domain>(&'domain self) -> Result<CompiledXlaFunction<'c, (In, In), (Out, Out)>, XlaDomainError>
    where
        'c: 'domain,
        In::Family: ParameterizedFamily<XlaCompileTracer<'domain, 'c>, To = In::To<XlaCompileTracer<'domain, 'c>>>,
        In::To<XlaCompileTracer<'domain, 'c>>: Parameterized<
                XlaCompileTracer<'domain, 'c>,
                Family = In::Family,
                ParameterStructure = In::ParameterStructure,
            >,
        Out::Family: ParameterizedFamily<XlaCompileTracer<'domain, 'c>, To = Out::To<XlaCompileTracer<'domain, 'c>>>,
        Out::To<XlaCompileTracer<'domain, 'c>>: Parameterized<
                XlaCompileTracer<'domain, 'c>,
                Family = Out::Family,
                ParameterStructure = Out::ParameterStructure,
                To<ArrayType> = Out,
            >,
    {
        let function = self;
        let input_signature = function.input_signature().map_err(|error| XlaDomainError::Array(error.into()))?;
        let primals_and_tangents = (input_signature.clone(), input_signature);
        let mesh = function.mesh().clone();
        let captures = function.source_program.captures().to_vec();
        compile_with_flat_captures(
            move |capture_references, _, inputs| {
                let (primals, tangents) = inputs;
                let output_structure = function.output_structure.clone();
                let primals = primals.into_parameters().collect::<Vec<_>>();
                let tangents = tangents.into_parameters().collect::<Vec<_>>();
                let context = primals.first().expect("jvp requires at least one input tracer").context().clone();
                let (primal_outputs, tangent_outputs): (
                    Vec<XlaCompileTracer<'domain, 'c>>,
                    Vec<XlaCompileTracer<'domain, 'c>>,
                ) = context
                    .jvp(
                        move |inputs| {
                            function
                                .stage_with_flat_capture_references(capture_references.as_slice(), inputs)
                                .expect("compiled-function jvp call staging should succeed")
                        },
                        primals,
                        tangents,
                    )
                    .expect("compiled-function jvp should stage successfully");
                let primal_tree =
                    Out::To::<_>::from_parameters(output_structure.clone(), primal_outputs).expect("primal reassembly");
                let tangent_tree =
                    Out::To::<_>::from_parameters(output_structure, tangent_outputs).expect("tangent reassembly");
                (primal_tree, tangent_tree)
            },
            captures,
            primals_and_tangents,
            &function.domain,
            CompilationOptions::new(XlaOptions::new(mesh)),
        )
    }

    /// Returns a new compiled function that runs `self` in parallel over `axis_size` lanes
    /// along a new leading axis of each input and output. Mirrors `jax.vmap(f)` with default
    /// `in_axes=0` / `out_axes=0`. Every input leaf gets a new leading axis of size `axis_size`;
    /// every output leaf is materialized with the batched axis at position 0. The batched
    /// leading axis is replicated for now.
    ///
    /// The implementation stages a `jit_call` operation and lets the ordinary batching rule for that
    /// call operation build the batched call boundary.
    ///
    /// # Limitation
    ///
    /// Programs that use `shard_map`, `linear_shard_map`, or `with_sharding_constraint` will
    /// surface [`BatchingError::MissingBatchingRule`](ryft_core::tracing_v2::batching::BatchingError)
    /// at batch time — the batching rules for those XLA-specific extension variants are not yet
    /// implemented. Non-shard-map ops batch correctly through the per-op rules.
    #[track_caller]
    pub fn batch<'domain>(&'domain self, axis_size: usize) -> Result<CompiledXlaFunction<'c, In, Out>, XlaDomainError>
    where
        'c: 'domain,
        In::Family: ParameterizedFamily<XlaCompileTracer<'domain, 'c>, To = In::To<XlaCompileTracer<'domain, 'c>>>,
        In::To<XlaCompileTracer<'domain, 'c>>: Parameterized<
                XlaCompileTracer<'domain, 'c>,
                Family = In::Family,
                ParameterStructure = In::ParameterStructure,
            >,
        Out::Family: ParameterizedFamily<XlaCompileTracer<'domain, 'c>, To = Out::To<XlaCompileTracer<'domain, 'c>>>,
        Out::To<XlaCompileTracer<'domain, 'c>>: Parameterized<
                XlaCompileTracer<'domain, 'c>,
                Family = Out::Family,
                ParameterStructure = Out::ParameterStructure,
                To<ArrayType> = Out,
            >,
    {
        let function = self;
        let unbatched_signature = function.input_signature().map_err(|error| XlaDomainError::Array(error.into()))?;
        let input_structure = unbatched_signature.parameter_structure();
        let batched_leaves: Vec<ArrayType> = unbatched_signature
            .into_parameters()
            .map(|t| add_leading_batch_dim(t, axis_size))
            .collect::<Result<Vec<_>, _>>()?;
        let batched_input = In::from_parameters(input_structure, batched_leaves)
            .map_err(|error| XlaDomainError::Array(error.into()))?;
        let mesh = function.mesh().clone();
        let captures = function.source_program.captures().to_vec();
        compile_with_flat_captures(
            move |capture_references, _, batched_tracers| {
                let output_structure = function.output_structure.clone();
                let output_count = function.output_types().len();
                let batched_tracers = batched_tracers.into_parameters().collect::<Vec<_>>();
                let input_count = batched_tracers.len();
                let context = batched_tracers.first().expect("batch requires at least one input").context().clone();
                let flat_outputs: Vec<XlaCompileTracer<'domain, 'c>> = context
                    .batch(
                        move |inputs| {
                            function.stage_with_flat_capture_references(capture_references.as_slice(), inputs)
                        },
                        batched_tracers,
                        vec![Some(0_usize); input_count],
                        vec![Some(0_usize); output_count],
                        Some(axis_size),
                    )
                    .expect("compiled-function batch should stage successfully");
                Out::To::<_>::from_parameters(output_structure, flat_outputs).expect("batch output reassembly")
            },
            captures,
            batched_input,
            &function.domain,
            CompilationOptions::new(XlaOptions::new(mesh)),
        )
    }
}

/// Adds a leading axis of `size` to `array_type`, replicated. Used by [`CompiledXlaFunction::batch`] to
/// construct the batched input signature. If `array_type` carried a sharding, the returned [`ArrayType`]
/// gets the same sharding with a leading [`ShardingDimension::replicated`] prepended so that
/// the sharding rank still matches the shape rank.
fn add_leading_batch_dim(array_type: ArrayType, size: usize) -> Result<ArrayType, XlaDomainError> {
    let mut dims: Vec<ryft_core::Size> = vec![ryft_core::Size::Static(size)];
    dims.extend(array_type.shape().dimensions().iter().copied());
    let sharding = match array_type.sharding() {
        Some(existing) => {
            let mut extended_dims = vec![ryft_core::sharding::ShardingDimension::replicated()];
            extended_dims.extend(existing.dimensions().iter().cloned());
            Some(
                ryft_core::sharding::Sharding::with_manual_axes(
                    existing.mesh().clone(),
                    extended_dims,
                    existing.unreduced_axes().clone(),
                    existing.reduced_manual_axes().clone(),
                    existing.varying_manual_axes().clone(),
                )
                .map_err(|error| XlaDomainError::Array(error.into()))?,
            )
        }
        None => None,
    };
    ArrayType::new(array_type.data_type(), ryft_core::Shape::new(dims), array_type.layout().cloned(), sharding)
        .map_err(|error| XlaDomainError::Array(error.into()))
}

/// Compiles `function` once and returns a [`CompiledXlaFunction`] that executes it on subsequent
/// calls. Mirrors `jax.jit`.
///
/// Equivalent to [`compile_with_options`] called with [`CompilationOptions::new`]
/// wrapping [`XlaOptions::new(mesh)`](XlaOptions::new).
///
/// The function is traced against the supplied `domain`, which lets nested compiled functions register runtime
/// captures in the same active trace. The resulting program is then compiled and executed against `domain`, sharing its
/// [`CompilationContext`](ryft_core::compilation::CompilationContext) cache across repeat
/// invocations at the same source line.
#[track_caller]
pub fn compile<
    'domain,
    'c: 'domain,
    F: FnOnce(In::To<XlaCompileTracer<'domain, 'c>>) -> Out::To<XlaCompileTracer<'domain, 'c>>,
    In: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<XlaConstant>
                        + ParameterizedFamily<XlaCompileTracer<'domain, 'c>>,
            ParameterStructure: std::hash::Hash,
        >,
    Out: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<XlaConstant>
                        + ParameterizedFamily<XlaCompileTracer<'domain, 'c>>,
            To<XlaCompileTracer<'domain, 'c>>: Parameterized<XlaCompileTracer<'domain, 'c>, To<ArrayType> = Out>,
        >,
>(
    function: F,
    input_types: In,
    domain: &'domain XlaDomain<'c>,
    mesh: DeviceMesh,
) -> Result<CompiledXlaFunction<'c, In, Out>, XlaDomainError> {
    compile_with_options(function, input_types, domain, CompilationOptions::new(XlaOptions::new(mesh)))
}

/// Same as [`compile`] but makes runtime arrays explicit captures of the compiled program.
///
/// The closure receives capture tracers first and ordinary input tracers second. Captures are compiled as hidden
/// executable arguments and are supplied from the returned [`CompiledXlaFunction`] at execution time, so callers of the
/// compiled function still pass only `In` inputs.
#[track_caller]
pub fn compile_with_captures<
    'domain,
    'c: 'domain,
    F: FnOnce(
        Vec<XlaCompileTracer<'domain, 'c>>,
        In::To<XlaCompileTracer<'domain, 'c>>,
    ) -> Out::To<XlaCompileTracer<'domain, 'c>>,
    In: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<XlaConstant>
                        + ParameterizedFamily<XlaCompileTracer<'domain, 'c>>,
            ParameterStructure: std::hash::Hash,
        >,
    Out: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<XlaConstant>
                        + ParameterizedFamily<XlaCompileTracer<'domain, 'c>>,
            To<XlaCompileTracer<'domain, 'c>>: Parameterized<XlaCompileTracer<'domain, 'c>, To<ArrayType> = Out>,
        >,
>(
    function: F,
    captures: Vec<Array<'c>>,
    input_types: In,
    domain: &'domain XlaDomain<'c>,
    mesh: DeviceMesh,
) -> Result<CompiledXlaFunction<'c, In, Out>, XlaDomainError> {
    compile_with_flat_captures(
        |_, capture_tracers, inputs| function(capture_tracers, inputs),
        captures,
        input_types,
        domain,
        CompilationOptions::new(XlaOptions::new(mesh)),
    )
}

/// Same as [`compile`] but accepts a full [`CompilationOptions`] payload for
/// JAX-style configuration: structural call-site fingerprinting plus the XLA-specific [`XlaOptions`] (mesh, sharding
/// overrides, per-input buffer donation flags).
#[track_caller]
pub fn compile_with_options<
    'domain,
    'c: 'domain,
    F: FnOnce(In::To<XlaCompileTracer<'domain, 'c>>) -> Out::To<XlaCompileTracer<'domain, 'c>>,
    In: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<XlaConstant>
                        + ParameterizedFamily<XlaCompileTracer<'domain, 'c>>,
            ParameterStructure: std::hash::Hash,
        >,
    Out: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<XlaConstant>
                        + ParameterizedFamily<XlaCompileTracer<'domain, 'c>>,
            To<XlaCompileTracer<'domain, 'c>>: Parameterized<XlaCompileTracer<'domain, 'c>, To<ArrayType> = Out>,
        >,
>(
    function: F,
    input_types: In,
    domain: &'domain XlaDomain<'c>,
    options: CompilationOptions<XlaDomain<'c>>,
) -> Result<CompiledXlaFunction<'c, In, Out>, XlaDomainError> {
    compile_with_flat_captures(|_, _, inputs| function(inputs), Vec::new(), input_types, domain, options)
}

#[track_caller]
fn compile_with_flat_captures<
    'domain,
    'c: 'domain,
    F: FnOnce(
        Vec<XlaConstant>,
        Vec<XlaCompileTracer<'domain, 'c>>,
        In::To<XlaCompileTracer<'domain, 'c>>,
    ) -> Out::To<XlaCompileTracer<'domain, 'c>>,
    In: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<XlaConstant>
                        + ParameterizedFamily<XlaCompileTracer<'domain, 'c>>,
            ParameterStructure: std::hash::Hash,
        >,
    Out: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<XlaConstant>
                        + ParameterizedFamily<XlaCompileTracer<'domain, 'c>>,
            To<XlaCompileTracer<'domain, 'c>>: Parameterized<XlaCompileTracer<'domain, 'c>, To<ArrayType> = Out>,
        >,
>(
    function: F,
    captures: Vec<Array<'c>>,
    input_types: In,
    domain: &'domain XlaDomain<'c>,
    options: CompilationOptions<XlaDomain<'c>>,
) -> Result<CompiledXlaFunction<'c, In, Out>, XlaDomainError> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Capture the call site BEFORE doing anything else, so `#[track_caller]` propagates correctly,
    // and fold in a hash of the input tree's structure. Treedef-only fields (non-`Parameter`
    // struct fields like `batch_size: usize`, mode flags, hyperparameters) partition the cache
    // here so that repeat invocations at the same source line with structurally-different
    // inputs get distinct compiled artifacts.
    let base_fingerprint = FunctionFingerprint::from_caller();
    let mut structure_hasher = DefaultHasher::new();
    input_types.parameter_structure().hash(&mut structure_hasher);
    let function_fingerprint =
        FunctionFingerprint::Composite { base: Box::new(base_fingerprint), extra: structure_hasher.finish() };

    let xla_options = options.options;

    // Apply the in-shardings override (if any) before tracing — the override changes the input
    // ArrayTypes that the SPMD lowering will see.
    let input_types = if let Some(in_shardings) = &xla_options.in_shardings {
        apply_in_shardings_override(input_types, in_shardings)?
    } else {
        input_types
    };

    let (output_types_tree, program, captures) =
        trace_with_flat_captures::<F, In, Out>(function, captures, input_types, domain)
            .map_err(XlaDomainError::from)?;
    let output_structure = output_types_tree.parameter_structure();
    let output_count = output_types_tree.parameter_count();

    // Validate the out-shardings override before cache lookup so malformed options fail even on cache hits. The
    // compile boundary applies the actual output type rewrite while lowering.
    if let Some(ref out_shardings) = xla_options.out_shardings {
        if out_shardings.len() != output_count {
            return Err(XlaDomainError::InvalidCompilationOptions {
                reason: format!(
                    "out_shardings has {} entries but the function has {} flat output(s)",
                    out_shardings.len(),
                    output_count,
                ),
            });
        }
    }

    // Build the input type list from the traced program's atoms (these reflect any
    // in-shardings override that was applied above).
    let input_types_vec: Vec<ArrayType> = program
        .input_ids()
        .iter()
        .map(|atom_id| program.atoms()[atom_id.index()].r#type().into_owned())
        .collect();

    // Validate donation arity. Empty `donation_flags` means the user did not declare donation;
    // any other length must match the function's flat input arity.
    if !xla_options.donation_flags.is_empty() && xla_options.donation_flags.len() != input_types_vec.len() {
        return Err(XlaDomainError::InvalidCompilationOptions {
            reason: format!(
                "donation_flags has {} entries but the function has {} flat input(s)",
                xla_options.donation_flags.len(),
                input_types_vec.len(),
            ),
        });
    }

    let source_program = CapturedProgram::new(program, captures);
    source_program.validate_capture_references().map_err(XlaDomainError::from)?;
    let capture_types =
        source_program.captures().iter().map(|capture| capture.r#type().into_owned()).collect::<Vec<_>>();
    let cache_argument_types = capture_types.iter().cloned().chain(input_types_vec.iter().cloned()).collect::<Vec<_>>();

    // Cache key derived from the execution domain. Hidden captures are part of the executable signature, so their
    // types participate in the key while their runtime values stay on the returned function handle.
    let cache_key = domain.compilation_key(&function_fingerprint, &cache_argument_types, &xla_options);

    // Cache lookup / on-miss compile.
    let cache = domain.cache().expect("XlaDomain always exposes a compile cache");
    let compiled: XlaCompiledProgram<'c> =
        cache.get_or_compile(domain, cache_key, || -> Result<XlaCompiledProgram<'c>, XlaDomainError> {
            domain.compile_program_with_captures(source_program.program(), capture_types.as_slice(), &xla_options)
        })?;

    Ok(CompiledXlaFunction { program: compiled, source_program, output_structure, domain: domain.clone() })
}

fn trace_with_flat_captures<
    'domain,
    'c: 'domain,
    F: FnOnce(
        Vec<XlaConstant>,
        Vec<XlaCompileTracer<'domain, 'c>>,
        In::To<XlaCompileTracer<'domain, 'c>>,
    ) -> Out::To<XlaCompileTracer<'domain, 'c>>,
    In: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<XlaConstant>
                        + ParameterizedFamily<XlaCompileTracer<'domain, 'c>>,
        >,
    Out: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<XlaConstant>
                        + ParameterizedFamily<XlaCompileTracer<'domain, 'c>>,
            To<XlaCompileTracer<'domain, 'c>>: Parameterized<XlaCompileTracer<'domain, 'c>, To<ArrayType> = Out>,
        >,
>(
    function: F,
    captures: Vec<Array<'c>>,
    input_types: In,
    domain: &'domain XlaDomain<'c>,
) -> Result<
    (Out, crate::experimental::ops::XlaProgram<In::To<XlaConstant>, XlaSourceProgramOutput<Out>>, Vec<Array<'c>>),
    ProgramError,
> {
    let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
    let capture_table = Rc::new(RefCell::new(Vec::new()));
    let context: TracingContext<'domain, XlaDomain<'c>> =
        TracingContext::new_with_captures(domain, builder.clone(), capture_table.clone());
    let capture_references = captures.into_iter().map(|value| context.capture(value)).collect::<Result<Vec<_>, _>>()?;
    let capture_tracers = capture_references.iter().cloned().map(|capture| context.constant(capture)).collect();
    let input_structure = input_types.parameter_structure();
    let inputs = input_types.map_parameters(|input_type| context.input(input_type))?;
    let outputs = function(capture_references, capture_tracers, inputs);
    if let Some(error) = builder.borrow().error().cloned() {
        return Err(error);
    }
    let output_structure = outputs.parameter_structure();
    let output_ids = outputs.parameters().map(Tracer::atom_id).collect::<Result<Vec<_>, _>>()?;
    let output_types = outputs.map_parameters(|output| output.r#type().into_owned())?;
    drop(context);
    let captures = Rc::try_unwrap(capture_table).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
    let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
    let program = builder.build(output_ids, input_structure, output_structure)?;
    Ok((output_types, program, captures))
}

/// Traces `function` against `input_types` and returns the abstract output type tree, without
/// lowering or compiling. Mirrors `jax.eval_shape`.
#[track_caller]
pub fn infer_output_types<
    F: FnOnce(In::To<XlaCompileTracer<'static, 'static>>) -> Out::To<XlaCompileTracer<'static, 'static>>,
    In: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<XlaConstant>
                        + ParameterizedFamily<XlaCompileTracer<'static, 'static>>,
        >,
    Out: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<XlaConstant>
                        + ParameterizedFamily<XlaCompileTracer<'static, 'static>>,
            To<XlaCompileTracer<'static, 'static>>: Parameterized<
                XlaCompileTracer<'static, 'static>,
                To<ArrayType> = Out,
            >,
        >,
>(
    function: F,
    input_types: In,
) -> Result<Out, ProgramError> {
    let token: &'static XlaDomain<'static> = XlaDomain::token();
    TracingContext::infer_output_type(
        token,
        |tracers: In::To<XlaCompileTracer<'static, 'static>>| Ok(function(tracers)),
        input_types,
    )
}

/// Replaces the [`Sharding`] metadata on every [`ArrayType`] leaf of `input_types`.
fn apply_in_shardings_override<In: Parameterized<ArrayType>>(
    input_types: In,
    in_shardings: &[Sharding],
) -> Result<In, XlaDomainError> {
    let structure = input_types.parameter_structure();
    let flat: Vec<ArrayType> = input_types.into_parameters().collect();
    if flat.len() != in_shardings.len() {
        return Err(XlaDomainError::InvalidCompilationOptions {
            reason: format!(
                "in_shardings has {} entries but the function has {} flat input(s)",
                in_shardings.len(),
                flat.len(),
            ),
        });
    }
    let overridden = flat
        .into_iter()
        .zip(in_shardings)
        .map(|(array_type, sharding)| {
            ArrayType::new(
                array_type.data_type(),
                array_type.shape().clone(),
                array_type.layout().cloned(),
                Some(sharding.clone()),
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| XlaDomainError::Array(error.into()))?;
    In::from_parameters(structure, overridden).map_err(|error| XlaDomainError::Array(error.into()))
}

#[cfg(test)]
mod tests {
    use ryft_core::operations::trigonometric::Sin;
    use ryft_core::sharding::{Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, Sharding};
    use ryft_core::tracing_v2::DifferentiationContext;
    use ryft_core::types::data_types::DataType;
    use ryft_core::types::{ArrayType, Shape, Size};
    use ryft_pjrt::{ClientOptions, CpuClientOptions, load_cpu_plugin};

    use ryft_core::compilation::CompilationOptions;

    use crate::experimental::domains::{XlaDomain, XlaDomainError, XlaOptions};
    use crate::experimental::ops::{XlaOperation, XlaOperationExtension};
    use crate::tests::{values_from_bytes, values_to_bytes};
    use crate::{
        Array, CompiledXlaFunction, FromPjrt, compile, compile_with_captures, compile_with_options, infer_output_types,
    };

    fn single_device_mesh(client: &ryft_pjrt::Client<'_>) -> DeviceMesh {
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![device],
        )
        .unwrap()
    }

    fn two_device_mesh(client: &ryft_pjrt::Client<'_>) -> DeviceMesh {
        let devices: Vec<Device> = client
            .addressable_devices()
            .unwrap()
            .iter()
            .take(2)
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect();
        DeviceMesh::new(LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(), devices)
            .unwrap()
    }

    fn read_f32_array(client: &ryft_pjrt::Client<'_>, array: &Array<'_>) -> Vec<f32> {
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let shard_bytes = array
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        values_from_bytes::<f32>(shard_bytes.as_slice())
    }

    #[test]
    fn test_jit_unary_function_runs_end_to_end() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(4)]),
            None,
            Some(Sharding::replicated(mesh.logical_mesh().clone(), 1)),
        )
        .unwrap();
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin(), input_type.clone(), &engine, mesh.clone()).unwrap();

        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();
        let output = compiled.interpret(source).unwrap();

        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let shard_bytes = output
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        let observed = values_from_bytes::<f32>(shard_bytes.as_slice());
        for (got, &input) in observed.iter().zip(values.iter()) {
            assert!((got - input.sin()).abs() < 1e-5, "got {got}, expected ~{}", input.sin());
        }
    }

    /// Smoke test: after `compile` runs, the returned handle exposes the source
    /// [`Program`] that was traced. This is the foundation for diagnostics (printing the IR)
    /// and for transform composition / inner staging via
    /// [`CompiledXlaFunction::stage`].
    #[test]
    fn test_compiled_function_retains_source_program() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(4)]),
            None,
            Some(Sharding::replicated(mesh.logical_mesh().clone(), 1)),
        )
        .unwrap();
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin(), input_type.clone(), &engine, mesh.clone()).unwrap();

        // For `|x| x.sin()` with a single F32[4] input and a single F32[4] output, the program
        // should carry one input atom, one output atom, and at least one instruction (the sin).
        let source = compiled.source_program().program();
        assert_eq!(source.input_ids().len(), 1, "expected one program input for the unary closure");
        assert_eq!(source.output_ids().len(), 1, "expected one program output for the unary closure");
        assert!(
            !source.instructions().is_empty(),
            "traced program should carry at least one instruction (the body of x.sin())",
        );
    }

    /// Inner-composition smoke test: a compiled function can be staged into another
    /// `compile` closure as a sub-routine, producing the same result as if the
    /// whole computation were a single closure. Mirrors JAX's
    /// `jit(lambda x: jit(f)(x).cos())` pattern.
    ///
    /// Exercises [`CompiledXlaFunction::stage`], which stages the retained source program behind a
    /// `jit_call` boundary in the active outer trace.
    #[test]
    fn test_compiled_function_staged_inside_compile() {
        use ryft_core::operations::trigonometric::Cos;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(4)]),
            None,
            Some(Sharding::replicated(mesh.logical_mesh().clone(), 1)),
        )
        .unwrap();

        // Inner: compile `f = |x| x.sin()`.
        let inner: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin(), input_type.clone(), &engine, mesh.clone()).unwrap();

        // Outer: compile `g = |x| cos(inner(x))` by staging `inner` as one `jit_call` and applying `cos` to its
        // output.
        let outer: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(move |x| inner.stage(x).cos(), input_type.clone(), &engine, mesh.clone()).unwrap();
        let jit_call_count = outer
            .source_program()
            .program()
            .instructions()
            .iter()
            .filter(|instruction| {
                matches!(instruction.operation(), XlaOperation::Extension(XlaOperationExtension::JitCall(_)))
            })
            .count();
        let inlined_sin_count = outer
            .source_program()
            .program()
            .instructions()
            .iter()
            .filter(|instruction| matches!(instruction.operation(), XlaOperation::Sin))
            .count();
        assert_eq!(jit_call_count, 1, "inner compiled function should stage as one jit_call");
        assert_eq!(inlined_sin_count, 0, "inner function body should not be inlined into the outer trace");

        // Execute and compare against the mathematical reference.
        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source = Array::from_host_buffer(
            &client,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();
        let output = outer.interpret(source).unwrap();

        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let shard_bytes = output
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        let observed = values_from_bytes::<f32>(shard_bytes.as_slice());
        for (got, &input) in observed.iter().zip(values.iter()) {
            let expected = input.sin().cos();
            assert!((got - expected).abs() < 1e-5, "got {got}, expected ~{expected}");
        }
    }

    #[test]
    fn test_compile_with_captures_runs_with_hidden_capture_argument() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(4)]),
            None,
            Some(Sharding::replicated(mesh.logical_mesh().clone(), 1)),
        )
        .unwrap();
        let bias = Array::from_host_buffer(
            &client,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[2.0, 2.0, 2.0, 2.0]).as_slice(),
        )
        .unwrap();

        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x + captures[0].clone(),
            vec![bias],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        assert_eq!(compiled.source_program().captures().len(), 1);
        assert_eq!(compiled.source_program().open_captures_as_inputs().unwrap().input_ids().len(), 2);

        let input = Array::from_host_buffer(
            &client,
            input_type,
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let output = compiled.interpret(input).unwrap();

        assert_eq!(read_f32_array(&client, &output), vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_captured_compiled_function_stages_inside_ordinary_compile() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(4)]),
            None,
            Some(Sharding::replicated(mesh.logical_mesh().clone(), 1)),
        )
        .unwrap();
        let bias = Array::from_host_buffer(
            &client,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[2.0, 2.0, 2.0, 2.0]).as_slice(),
        )
        .unwrap();
        let inner: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x + captures[0].clone(),
            vec![bias],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let outer: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(move |x| inner.stage(x).sin(), input_type.clone(), &engine, mesh.clone()).unwrap();

        assert_eq!(outer.source_program().captures().len(), 1);

        let input = Array::from_host_buffer(
            &client,
            input_type,
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let output = outer.interpret(input).unwrap();
        let observed = read_f32_array(&client, &output);
        for (got, expected) in observed.iter().zip([3.0f32.sin(), 4.0f32.sin(), 5.0f32.sin(), 6.0f32.sin()]) {
            assert!((got - expected).abs() < 1e-5, "got {got}, expected ~{expected}");
        }
    }

    #[test]
    fn test_multiple_captured_compiled_functions_stage_inside_ordinary_compile() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(4)]),
            None,
            Some(Sharding::replicated(mesh.logical_mesh().clone(), 1)),
        )
        .unwrap();
        let left_bias = Array::from_host_buffer(
            &client,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[2.0, 2.0, 2.0, 2.0]).as_slice(),
        )
        .unwrap();
        let right_bias = Array::from_host_buffer(
            &client,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[10.0, 10.0, 10.0, 10.0]).as_slice(),
        )
        .unwrap();
        let left: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x + captures[0].clone(),
            vec![left_bias],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let right: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x + captures[0].clone(),
            vec![right_bias],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let outer: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(move |x| left.stage(x.clone()) + right.stage(x), input_type.clone(), &engine, mesh.clone())
                .unwrap();

        assert_eq!(outer.source_program().captures().len(), 2);

        let input = Array::from_host_buffer(
            &client,
            input_type,
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let output = outer.interpret(input).unwrap();
        let observed = read_f32_array(&client, &output);
        for (got, expected) in observed.iter().zip([14.0, 16.0, 18.0, 20.0]) {
            assert!((got - expected).abs() < 1e-5, "got {got}, expected ~{expected}");
        }
    }

    #[test]
    fn test_jvp_method_preserves_compiled_function_captures() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new()), None, Some(sharding)).unwrap();
        let bias = Array::from_host_buffer(
            &client,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[2.0]).as_slice(),
        )
        .unwrap();
        let inner: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x + captures[0].clone(),
            vec![bias],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let jvp_compiled: CompiledXlaFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> =
            inner.jvp().unwrap();

        assert_eq!(jvp_compiled.source_program().captures().len(), 1);

        let primal = Array::from_host_buffer(
            &client,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[3.0]).as_slice(),
        )
        .unwrap();
        let tangent =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f32>(&[4.0]).as_slice())
                .unwrap();
        let (primal_output, tangent_output) = jvp_compiled.interpret((primal, tangent)).unwrap();

        assert_eq!(read_f32_array(&client, &primal_output), vec![5.0]);
        assert_eq!(read_f32_array(&client, &tangent_output), vec![4.0]);
    }

    #[test]
    fn test_value_and_gradient_method_preserves_compiled_function_captures() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new()), None, Some(sharding)).unwrap();
        let bias = Array::from_host_buffer(
            &client,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[2.0]).as_slice(),
        )
        .unwrap();
        let inner: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x + captures[0].clone(),
            vec![bias],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let grad_compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> = inner.value_and_gradient().unwrap();

        assert_eq!(grad_compiled.source_program().captures().len(), 1);

        let input =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f32>(&[3.0]).as_slice())
                .unwrap();
        let output = grad_compiled.interpret(input).unwrap();

        assert_eq!(read_f32_array(&client, &output), vec![1.0]);
    }

    #[test]
    fn test_compile_can_stage_value_and_gradient_of_captured_compiled_function() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new()), None, Some(sharding)).unwrap();
        let bias = Array::from_host_buffer(
            &client,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[0.25]).as_slice(),
        )
        .unwrap();
        let inner: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile_with_captures(
            |captures, x| (x + captures[0].clone()).sin(),
            vec![bias],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let gradient: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile(
            move |x| {
                let context = x.context().clone();
                context
                    .value_and_gradient(move |y| inner.stage(y), x)
                    .expect("nested captured grad(jit) should stage")
            },
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        assert_eq!(gradient.source_program().captures().len(), 1);

        for &point in &[0.0f32, 0.25, 0.5, 1.0] {
            let input = Array::from_host_buffer(
                &client,
                input_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>(&[point]).as_slice(),
            )
            .unwrap();
            let output = gradient.interpret(input).unwrap();
            let observed = read_f32_array(&client, &output);
            assert_eq!(observed.len(), 1);
            let expected = (point + 0.25).cos();
            assert!(
                (observed[0] - expected).abs() < 1e-5,
                "grad(sin(x + bias))({point}) expected ~{expected}, got {}",
                observed[0],
            );
        }
    }

    #[test]
    fn test_compile_can_stage_jvp_of_captured_compiled_function() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new()), None, Some(sharding)).unwrap();
        let bias = Array::from_host_buffer(
            &client,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[2.0]).as_slice(),
        )
        .unwrap();
        let inner: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x + captures[0].clone(),
            vec![bias],
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let jvp_compiled: CompiledXlaFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> = compile(
            move |(primal, tangent)| {
                let context = primal.context().clone();
                context
                    .jvp(move |x| inner.stage(x), primal, tangent)
                    .expect("nested captured jvp(jit) should stage")
            },
            (input_type.clone(), input_type.clone()),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        assert_eq!(jvp_compiled.source_program().captures().len(), 1);

        let primal = Array::from_host_buffer(
            &client,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[3.0]).as_slice(),
        )
        .unwrap();
        let tangent =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f32>(&[4.0]).as_slice())
                .unwrap();
        let (primal_output, tangent_output) = jvp_compiled.interpret((primal, tangent)).unwrap();

        assert_eq!(read_f32_array(&client, &primal_output), vec![5.0]);
        assert_eq!(read_f32_array(&client, &tangent_output), vec![4.0]);
    }

    #[test]
    fn test_batch_method_preserves_compiled_function_captures() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let scalar_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let scalar_type = ArrayType::new(DataType::F32, Shape::new(Vec::new()), None, Some(scalar_sharding)).unwrap();
        let bias = Array::from_host_buffer(
            &client,
            scalar_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[2.0]).as_slice(),
        )
        .unwrap();
        let inner: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile_with_captures(
            |captures, x| x + captures[0].clone(),
            vec![bias],
            scalar_type,
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let batched: CompiledXlaFunction<'_, ArrayType, ArrayType> = inner.batch(4).unwrap();

        assert_eq!(batched.source_program().captures().len(), 1);

        let batched_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let batched_input_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]), None, Some(batched_sharding)).unwrap();
        let input = Array::from_host_buffer(
            &client,
            batched_input_type,
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let output = batched.interpret(input).unwrap();

        assert_eq!(read_f32_array(&client, &output), vec![3.0, 4.0, 5.0, 6.0]);
    }

    /// Outer-transform smoke test: applying `grad` to a `CompiledXlaFunction` produces a new compiled function that
    /// computes `d/dx f(x)`. This mirrors JAX's `grad(jit(f))` idiom.
    ///
    /// The mechanism: [`CompiledXlaFunction::stage`] stages a `jit_call` operation inside a `grad` trace. The outer
    /// transform rewrites that operation through the same [`DifferentiableOperation`](ryft_core::tracing_v2::DifferentiableOperation)
    /// machinery as primitive ops.
    #[test]
    fn test_value_and_gradient_of_compiled_function_round_trips() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new()), None, Some(sharding)).unwrap();

        // Compile `f = |x| x.sin()`.
        let inner: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin(), input_type.clone(), &engine, mesh.clone()).unwrap();

        let grad_compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> = inner.value_and_gradient().unwrap();

        // d/dx sin(x) = cos(x). Verify at a few points.
        for &point in &[0.0f32, 0.25, 0.5, 1.0] {
            let source = Array::from_host_buffer(
                &client,
                input_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>([point].as_slice()).as_slice(),
            )
            .unwrap();
            let output = grad_compiled.interpret(source).unwrap();
            let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
            let shard_bytes = output
                .device_shard(device_id)
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            let observed = values_from_bytes::<f32>(shard_bytes.as_slice())[0];
            let expected = point.cos();
            assert!((observed - expected).abs() < 1e-5, "grad(sin)({point}) expected ~{expected}, got {observed}",);
        }
    }

    /// Equivalent to [`test_value_and_gradient_of_compiled_function_round_trips`] but through the
    /// [`CompiledXlaFunction::value_and_gradient`] method. Verifies that `inner.value_and_gradient()` produces a
    /// compiled function whose `.interpret(point)` matches `cos(point)` for sample points.
    #[test]
    fn test_value_and_gradient_method_matches_in_trace_value_and_gradient() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new()), None, Some(sharding)).unwrap();

        let inner: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin(), input_type.clone(), &engine, mesh.clone()).unwrap();
        let grad_compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> = inner.value_and_gradient().unwrap();

        for &point in &[0.0f32, 0.25, 0.5, 1.0] {
            let source = Array::from_host_buffer(
                &client,
                input_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>([point].as_slice()).as_slice(),
            )
            .unwrap();
            let output = grad_compiled.interpret(source).unwrap();
            let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
            let shard_bytes = output
                .device_shard(device_id)
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            let observed = values_from_bytes::<f32>(shard_bytes.as_slice())[0];
            let expected = point.cos();
            assert!(
                (observed - expected).abs() < 1e-5,
                "f.value_and_gradient()({point}) expected ~{expected}, got {observed}",
            );
        }
    }

    /// Distinct `inner.value_and_gradient()` call sites must yield distinct cache entries (confirming
    /// `#[track_caller]` propagation through the method). Same-line repeats share one entry.
    #[test]
    fn test_value_and_gradient_method_partitions_cache_by_call_site() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new()), None, Some(sharding)).unwrap();

        let inner: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin(), input_type.clone(), &engine, mesh.clone()).unwrap();
        let baseline = engine.cache_size();

        // Two `value_and_gradient` calls at DIFFERENT source lines: must produce two distinct cache entries.
        let _grad_a: CompiledXlaFunction<'_, ArrayType, ArrayType> = inner.value_and_gradient().unwrap();
        let _grad_b: CompiledXlaFunction<'_, ArrayType, ArrayType> = inner.value_and_gradient().unwrap();
        assert_eq!(engine.cache_size(), baseline + 2);

        // Multiple `value_and_gradient` calls at the SAME source line: must share one cache entry.
        let baseline = engine.cache_size();
        for _ in 0..3 {
            let _: CompiledXlaFunction<'_, ArrayType, ArrayType> = inner.value_and_gradient().unwrap();
        }
        assert_eq!(engine.cache_size(), baseline + 1);
    }

    /// For `f = |x| x.sin()`, `f.jvp()` produces a compiled function whose
    /// `.interpret((primal, tangent))` returns `(sin(primal), cos(primal) * tangent)` within `1e-5`.
    #[test]
    fn test_jvp_method_returns_primal_and_tangent() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new()), None, Some(sharding)).unwrap();

        let inner: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin(), input_type.clone(), &engine, mesh.clone()).unwrap();
        let jvp_compiled: CompiledXlaFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> =
            inner.jvp().unwrap();
        let jit_call_count = jvp_compiled
            .source_program()
            .program()
            .instructions()
            .iter()
            .filter(|instruction| {
                matches!(instruction.operation(), XlaOperation::Extension(XlaOperationExtension::JitCall(_)))
            })
            .count();
        let inlined_sin_count = jvp_compiled
            .source_program()
            .program()
            .instructions()
            .iter()
            .filter(|instruction| matches!(instruction.operation(), XlaOperation::Sin))
            .count();
        assert_eq!(jit_call_count, 2, "jvp(jit(f)) should stage separate primal and pushforward jit_call boundaries",);
        assert_eq!(inlined_sin_count, 0, "jvp(jit(f)) should not inline the callee body");

        for &(primal, tangent) in &[(0.0f32, 1.0f32), (0.25, 2.0), (0.5, -0.5), (1.0, 0.7)] {
            let primal_array = Array::from_host_buffer(
                &client,
                input_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>([primal].as_slice()).as_slice(),
            )
            .unwrap();
            let tangent_array = Array::from_host_buffer(
                &client,
                input_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>([tangent].as_slice()).as_slice(),
            )
            .unwrap();
            let (primal_out, tangent_out) = jvp_compiled.interpret((primal_array, tangent_array)).unwrap();
            let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
            let primal_observed = {
                let shard_bytes = primal_out
                    .device_shard(device_id)
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .copy_to_host(None)
                    .unwrap()
                    .r#await()
                    .unwrap();
                values_from_bytes::<f32>(shard_bytes.as_slice())[0]
            };
            let tangent_observed = {
                let shard_bytes = tangent_out
                    .device_shard(device_id)
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .copy_to_host(None)
                    .unwrap()
                    .r#await()
                    .unwrap();
                values_from_bytes::<f32>(shard_bytes.as_slice())[0]
            };
            let expected_primal = primal.sin();
            let expected_tangent = primal.cos() * tangent;
            assert!(
                (primal_observed - expected_primal).abs() < 1e-5,
                "jvp primal at (primal={primal}, tangent={tangent}): expected ~{expected_primal}, got {primal_observed}",
            );
            assert!(
                (tangent_observed - expected_tangent).abs() < 1e-5,
                "jvp tangent at (primal={primal}, tangent={tangent}): expected ~{expected_tangent}, got {tangent_observed}",
            );
        }
    }

    /// For scalar `f = |x| x.sin()`, `f.batch(4)?.interpret(batched_array_of_4)` returns
    /// `[sin(x[0]), ..., sin(x[3])]` within `1e-5`.
    #[test]
    fn test_batch_method_batches_leading_axis() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let scalar_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let scalar_input_type =
            ArrayType::new(DataType::F32, Shape::new(Vec::new()), None, Some(scalar_sharding)).unwrap();

        let inner: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin(), scalar_input_type, &engine, mesh.clone()).unwrap();
        let batched: CompiledXlaFunction<'_, ArrayType, ArrayType> = inner.batch(4).unwrap();
        let jit_call_count = batched
            .source_program()
            .program()
            .instructions()
            .iter()
            .filter(|instruction| {
                matches!(instruction.operation(), XlaOperation::Extension(XlaOperationExtension::JitCall(_)))
            })
            .count();
        let inlined_sin_count = batched
            .source_program()
            .program()
            .instructions()
            .iter()
            .filter(|instruction| matches!(instruction.operation(), XlaOperation::Sin))
            .count();
        assert_eq!(jit_call_count, 1, "batch(jit(f)) should stage one batched jit_call boundary");
        assert_eq!(inlined_sin_count, 0, "batch(jit(f)) should not inline the callee body");

        let batched_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let batched_input_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]), None, Some(batched_sharding)).unwrap();
        let inputs = [0.0f32, 0.5, 1.0, 1.5];
        let source = Array::from_host_buffer(
            &client,
            batched_input_type,
            mesh.clone(),
            values_to_bytes::<f32>(&inputs).as_slice(),
        )
        .unwrap();
        let output = batched.interpret(source).unwrap();

        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let shard_bytes = output
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        let observed = values_from_bytes::<f32>(shard_bytes.as_slice());
        assert_eq!(observed.len(), 4, "expected 4 lane outputs");
        for (got, &input) in observed.iter().zip(inputs.iter()) {
            let expected = input.sin();
            assert!((got - expected).abs() < 1e-5, "batch(sin)({input}) expected ~{expected}, got {got}");
        }
    }

    #[test]
    fn test_jvp_and_batch_methods_compose_around_jit_call() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let scalar_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let scalar_input_type =
            ArrayType::new(DataType::F32, Shape::new(Vec::new()), None, Some(scalar_sharding)).unwrap();
        let batched_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let batched_input_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]), None, Some(batched_sharding)).unwrap();

        let inner: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin(), scalar_input_type, &engine, mesh.clone()).unwrap();
        let jvp_inner: CompiledXlaFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> = inner.jvp().unwrap();
        let batch_jvp: CompiledXlaFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> =
            jvp_inner.batch(4).unwrap();
        let batch_inner: CompiledXlaFunction<'_, ArrayType, ArrayType> = inner.batch(4).unwrap();
        let jvp_batch: CompiledXlaFunction<'_, (ArrayType, ArrayType), (ArrayType, ArrayType)> =
            batch_inner.jvp().unwrap();

        let primals = [0.0f32, 0.25, 0.5, 1.0];
        let tangents = [1.0f32, 2.0, -0.5, 0.7];
        let make_array = |values: &[f32]| {
            Array::from_host_buffer(
                &client,
                batched_input_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>(values).as_slice(),
            )
            .unwrap()
        };

        for (label, compiled) in [("batch(jvp(jit(f)))", batch_jvp), ("jvp(batch(jit(f)))", jvp_batch)] {
            let (primal_output, tangent_output) =
                compiled.interpret((make_array(&primals), make_array(&tangents))).unwrap();
            let primal_observed = read_f32_array(&client, &primal_output);
            let tangent_observed = read_f32_array(&client, &tangent_output);
            for (index, (&primal, &tangent)) in primals.iter().zip(tangents.iter()).enumerate() {
                let expected_primal = primal.sin();
                let expected_tangent = primal.cos() * tangent;
                assert!(
                    (primal_observed[index] - expected_primal).abs() < 1e-5,
                    "{label} primal lane {index}: expected ~{expected_primal}, got {}",
                    primal_observed[index],
                );
                assert!(
                    (tangent_observed[index] - expected_tangent).abs() < 1e-5,
                    "{label} tangent lane {index}: expected ~{expected_tangent}, got {}",
                    tangent_observed[index],
                );
            }
        }
    }

    #[test]
    fn test_jit_binary_function_with_tuple_input() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let shape = Shape::new(vec![Size::Static(3)]);
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let input_type = ArrayType::new(DataType::F32, shape.clone(), None, Some(sharding.clone())).unwrap();
        let compiled: CompiledXlaFunction<'_, (ArrayType, ArrayType), ArrayType> =
            compile(|(a, b)| a + b, (input_type.clone(), input_type.clone()), &engine, mesh.clone()).unwrap();

        let a_values = [10.0f32, 20.0, 30.0];
        let b_values = [1.0f32, 2.0, 3.0];
        let a = Array::from_host_buffer(
            &client,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&a_values).as_slice(),
        )
        .unwrap();
        let b =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f32>(&b_values).as_slice())
                .unwrap();
        let output = compiled.interpret((a, b)).unwrap();

        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let shard_bytes = output
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        let observed = values_from_bytes::<f32>(shard_bytes.as_slice());
        assert_eq!(observed, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_jit_cache_hit_on_repeated_call_site() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(4)]),
            None,
            Some(Sharding::replicated(mesh.logical_mesh().clone(), 1)),
        )
        .unwrap();

        assert_eq!(engine.cache_size(), 0);
        // Two `compile` invocations on the same source line (inside the loop body) share a call-site
        // fingerprint, so the second invocation hits the cache instead of compiling again.
        for _ in 0..2 {
            let _: CompiledXlaFunction<'_, ArrayType, ArrayType> =
                compile(|x| x.sin(), input_type.clone(), &engine, mesh.clone()).unwrap();
        }
        assert_eq!(engine.cache_size(), 1, "repeat compile at the same call site should hit the cache");
    }

    #[test]
    fn test_jit_distinct_call_sites_use_distinct_cache_entries() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(4)]),
            None,
            Some(Sharding::replicated(mesh.logical_mesh().clone(), 1)),
        )
        .unwrap();

        assert_eq!(engine.cache_size(), 0);
        // Two `compile` invocations at distinct source lines populate two cache entries even when
        // the closure and inputs are identical, mirroring the way JAX's compile cache keys on
        // function identity (which differs per Python `id()` even for source-equivalent
        // lambdas).
        let _: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin(), input_type.clone(), &engine, mesh.clone()).unwrap();
        let _: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile(|x| x.sin(), input_type, &engine, mesh).unwrap();
        assert_eq!(engine.cache_size(), 2);
    }

    #[test]
    fn test_compile_with_options_donates_argument() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(3)]),
            None,
            Some(Sharding::replicated(mesh.logical_mesh().clone(), 1)),
        )
        .unwrap();
        let options = CompilationOptions::new(XlaOptions::new(mesh.clone()).with_donate(true));
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile_with_options(|x| x.sin(), input_type.clone(), &engine, options).unwrap();

        let values = [0.0f32, 0.5, 1.0];
        let source =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();
        let output = compiled.interpret(source).unwrap();

        // The output remains independently readable after the donating call returns. Donation
        // is opaque from the host side — PJRT may reuse the input's device buffer for the
        // output, but the public API only observes the resulting `Array`.
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let shard_bytes = output
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        let observed = values_from_bytes::<f32>(shard_bytes.as_slice());
        for (got, &input) in observed.iter().zip(values.iter()) {
            assert!((got - input.sin()).abs() < 1e-5, "got {got}, expected ~{}", input.sin());
        }
    }

    #[test]
    fn test_compile_with_options_rejects_donation_arity_mismatch() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(2)]),
            None,
            Some(Sharding::replicated(mesh.logical_mesh().clone(), 1)),
        )
        .unwrap();
        // Build a `donation_flags` vec whose length does not match the function's flat input
        // arity. `with_donate` enforces matching shape via the `Parameterized<bool>` bound on
        // its argument, so producing an arity mismatch requires setting `donation_flags`
        // directly on `XlaOptions`.
        let mut xla_options = XlaOptions::new(mesh.clone());
        xla_options.donation_flags = vec![true, false, false]; // 3 entries, 1 input
        let options = CompilationOptions::new(xla_options);
        let result: Result<CompiledXlaFunction<'_, ArrayType, ArrayType>, XlaDomainError> =
            compile_with_options(|x| x.sin(), input_type, &engine, options);
        assert!(matches!(result, Err(XlaDomainError::InvalidCompilationOptions { .. })));
    }

    /// Two compiles at the same source line with the SAME array leaf but DIFFERENT
    /// non-`Parameter` field on the input tree (`batch_size: usize`) must each populate their
    /// own cache entry. The cache key folds in a hash of `Input::ParameterStructure`, so any
    /// treedef-only difference (hyperparameters, mode flags, ...) partitions automatically.
    #[test]
    fn test_compile_partitions_cache_by_input_tree_structure() {
        use ryft_core::parameters::Parameter;
        use ryft_macros::Parameterized;

        #[derive(Parameterized, Debug, Clone, PartialEq, Eq, Hash)]
        #[ryft(crate = "ryft_core")]
        struct HyperparamInput<P: Parameter> {
            array: P,
            batch_size: usize,
        }

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);
        let array_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(4)]),
            None,
            Some(Sharding::replicated(mesh.logical_mesh().clone(), 1)),
        )
        .unwrap();

        for batch_size in [32usize, 64usize] {
            let input = HyperparamInput { array: array_type.clone(), batch_size };
            let _: CompiledXlaFunction<'_, HyperparamInput<ArrayType>, ArrayType> =
                compile(|input| input.array.sin(), input, &engine, mesh.clone()).unwrap();
        }
        assert_eq!(engine.cache_size(), 2);
    }

    #[test]
    fn test_compile_with_options_in_shardings_override_replaces_input_sharding() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = two_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        // input_type carries the abstract shape & dtype but a "wrong" sharding (replicated). The
        // `in_shardings` override replaces it with a 2-way shard along "x" before tracing, so
        // the compiled program shards the input across the 2-device mesh.
        let shape = Shape::new(vec![Size::Static(4)]);
        let abstract_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let abstract_input_type = ArrayType::new(DataType::F32, shape.clone(), None, Some(abstract_sharding)).unwrap();
        let sharded =
            Sharding::new(mesh.logical_mesh().clone(), vec![ryft_core::sharding::ShardingDimension::sharded(["x"])])
                .unwrap();
        let xla_options = XlaOptions::new(mesh.clone()).with_in_shardings(vec![sharded.clone()]);
        let options = CompilationOptions::new(xla_options);
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile_with_options(|x| x.sin(), abstract_input_type, &engine, options).unwrap();

        // Build the input array under the overridden sharding so it matches the executable's
        // expected layout.
        let input_type = ArrayType::new(DataType::F32, shape, None, Some(sharded)).unwrap();
        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();
        let output = compiled.interpret(source).unwrap();

        // Reassemble values from both shards in device order.
        let mut observed: Vec<f32> = Vec::with_capacity(values.len());
        for device in client.addressable_devices().unwrap().iter().take(2) {
            let device_id = device.id().unwrap();
            let shard_bytes = output
                .device_shard(device_id)
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            observed.extend(values_from_bytes::<f32>(shard_bytes.as_slice()));
        }
        for (got, &input) in observed.iter().zip(values.iter()) {
            assert!((got - input.sin()).abs() < 1e-5, "got {got}, expected ~{}", input.sin());
        }
    }

    #[test]
    fn test_compile_with_options_rejects_in_shardings_arity_mismatch() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(2)]),
            None,
            Some(Sharding::replicated(mesh.logical_mesh().clone(), 1)),
        )
        .unwrap();
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        // Two shardings for one flat input — should fail.
        let xla_options = XlaOptions::new(mesh).with_in_shardings(vec![sharding.clone(), sharding]);
        let options = CompilationOptions::new(xla_options);
        let result: Result<CompiledXlaFunction<'_, ArrayType, ArrayType>, XlaDomainError> =
            compile_with_options(|x| x.sin(), input_type, &engine, options);
        assert!(matches!(result, Err(XlaDomainError::InvalidCompilationOptions { .. })));
    }

    #[test]
    fn test_compile_with_options_out_shardings_override_propagates_to_output_array() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = two_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let shape = Shape::new(vec![Size::Static(4)]);
        let sharded =
            Sharding::new(mesh.logical_mesh().clone(), vec![ryft_core::sharding::ShardingDimension::sharded(["x"])])
                .unwrap();
        let input_type = ArrayType::new(DataType::F32, shape.clone(), None, Some(sharded.clone())).unwrap();
        // Override the output sharding to the same 2-way shard along "x" so the partitioner
        // emits a fully-sharded output and `Array`'s sharding metadata matches.
        let xla_options = XlaOptions::new(mesh.clone()).with_out_shardings(vec![sharded.clone()]);
        let options = CompilationOptions::new(xla_options);
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile_with_options(|x| x.sin(), input_type.clone(), &engine, options).unwrap();

        // The returned Array should carry the overridden sharding.
        assert_eq!(compiled.output_types()[0].sharding(), Some(&sharded));

        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();
        let output = compiled.interpret(source).unwrap();
        assert_eq!(output.sharding(), &sharded);

        let mut observed: Vec<f32> = Vec::with_capacity(values.len());
        for device in client.addressable_devices().unwrap().iter().take(2) {
            let device_id = device.id().unwrap();
            let shard_bytes = output
                .device_shard(device_id)
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            observed.extend(values_from_bytes::<f32>(shard_bytes.as_slice()));
        }
        for (got, &input) in observed.iter().zip(values.iter()) {
            assert!((got - input.sin()).abs() < 1e-5, "got {got}, expected ~{}", input.sin());
        }
    }

    #[test]
    fn test_jit_implicitly_reshards_mismatched_inputs() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = two_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        // The executable expects a 2-way shard along "x", but the caller will pass a fully
        // replicated array. `CompiledXlaFunction::interpret` should silently reshard before executing.
        let shape = Shape::new(vec![Size::Static(4)]);
        let sharded =
            Sharding::new(mesh.logical_mesh().clone(), vec![ryft_core::sharding::ShardingDimension::sharded(["x"])])
                .unwrap();
        let input_type = ArrayType::new(DataType::F32, shape.clone(), None, Some(sharded.clone())).unwrap();
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin(), input_type.clone(), &engine, mesh.clone()).unwrap();

        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let replicated_input_type = ArrayType::new(DataType::F32, shape, None, Some(replicated)).unwrap();
        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source = Array::from_host_buffer(
            &client,
            replicated_input_type,
            mesh.clone(),
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();
        // Calling with a replicated source against a sharded-expecting executable would error
        // without implicit reshard. With reshard it should succeed and produce correct output.
        let output = compiled.interpret(source).unwrap();
        assert_eq!(output.sharding(), &sharded);

        let mut observed: Vec<f32> = Vec::with_capacity(values.len());
        for device in client.addressable_devices().unwrap().iter().take(2) {
            let device_id = device.id().unwrap();
            let shard_bytes = output
                .device_shard(device_id)
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            observed.extend(values_from_bytes::<f32>(shard_bytes.as_slice()));
        }
        for (got, &input) in observed.iter().zip(values.iter()) {
            assert!((got - input.sin()).abs() < 1e-5, "got {got}, expected ~{}", input.sin());
        }
    }

    #[test]
    fn test_compile_with_options_rejects_out_shardings_arity_mismatch() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(2)]),
            None,
            Some(Sharding::replicated(mesh.logical_mesh().clone(), 1)),
        )
        .unwrap();
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let xla_options = XlaOptions::new(mesh).with_out_shardings(vec![sharding.clone(), sharding]);
        let options = CompilationOptions::new(xla_options);
        let result: Result<CompiledXlaFunction<'_, ArrayType, ArrayType>, XlaDomainError> =
            compile_with_options(|x| x.sin(), input_type, &engine, options);
        assert!(matches!(result, Err(XlaDomainError::InvalidCompilationOptions { .. })));
    }

    #[test]
    fn test_infer_output_types_returns_output_types_without_compiling() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(7)]),
            None,
            Some(Sharding::replicated(mesh.logical_mesh().clone(), 1)),
        )
        .unwrap();

        let output_type: ArrayType = infer_output_types(|x| x.sin(), input_type.clone()).unwrap();
        assert_eq!(output_type.data_type(), DataType::F32);
        assert_eq!(output_type.shape(), input_type.shape());
        // `infer_output_types` must not have populated the compile cache.
        assert_eq!(engine.cache_size(), 0);
    }

    #[test]
    fn test_jit_value_and_gradient_compiles_and_runs() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new()), None, Some(sharding.clone())).unwrap();

        let primal: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| x.sin(), input_type.clone(), &engine, mesh.clone()).unwrap();
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> = primal.value_and_gradient().unwrap();

        let input_value = 0.75f32;
        let source = Array::from_host_buffer(
            &client,
            input_type,
            mesh.clone(),
            values_to_bytes::<f32>([input_value].as_slice()).as_slice(),
        )
        .unwrap();
        let output = compiled.interpret(source).unwrap();

        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let shard_bytes = output
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        let observed = values_from_bytes::<f32>(shard_bytes.as_slice());
        assert_eq!(observed.len(), 1);
        let expected = input_value.cos();
        assert!(
            (observed[0] - expected).abs() < 1e-5,
            "expected d/dx sin({input_value}) ~= {expected}, got {}",
            observed[0],
        );
    }

    /// Verifies that `with_sharding_constraint` works inside a `compile`-compiled function: the sharding constraint is
    /// preserved through the trace, and the output array carries the constrained sharding on each device.
    #[test]
    fn test_jit_with_sharding_constraint_constrains_output_sharding() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = two_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let shape = Shape::new(vec![Size::Static(4)]);
        let sharded =
            Sharding::new(mesh.logical_mesh().clone(), vec![ryft_core::sharding::ShardingDimension::sharded(["x"])])
                .unwrap();
        let input_type = ArrayType::new(DataType::F32, shape.clone(), None, Some(sharded.clone())).unwrap();
        let target_sharding = sharded.clone();

        // The user invokes `with_sharding_constraint` directly inside the staged closure — it's compiled into the
        // same MLIR program as the rest of the function body.
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile(
            move |x| {
                let constrained = crate::experimental::shard_map::with_sharding_constraint(x, target_sharding.clone())
                    .expect("staged sharding constraint should succeed");
                constrained.sin()
            },
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();
        let output = compiled.interpret(source).unwrap();
        assert_eq!(output.sharding(), &sharded);

        let mut observed: Vec<f32> = Vec::with_capacity(values.len());
        for device in client.addressable_devices().unwrap().iter().take(2) {
            let device_id = device.id().unwrap();
            let shard_bytes = output
                .device_shard(device_id)
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            observed.extend(values_from_bytes::<f32>(shard_bytes.as_slice()));
        }
        for (got, &input) in observed.iter().zip(values.iter()) {
            assert!((got - input.sin()).abs() < 1e-5, "got {got}, expected ~{}", input.sin());
        }
    }

    /// Multiple staged sharding constraints inside one `compile` body compile into a single MLIR program with chained
    /// `sdy.sharding_constraint` ops — exactly one cache entry, exactly one PJRT execute per call. This is the
    /// async-pipelined regime: PJRT runs the whole compiled program in one shot without per-reshard host sync.
    #[test]
    fn test_jit_with_chained_sharding_constraints_compiles_to_one_program() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = two_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let shape = Shape::new(vec![Size::Static(4)]);
        let sharded =
            Sharding::new(mesh.logical_mesh().clone(), vec![ryft_core::sharding::ShardingDimension::sharded(["x"])])
                .unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let input_type = ArrayType::new(DataType::F32, shape.clone(), None, Some(sharded.clone())).unwrap();
        let constraint_a = replicated.clone();
        let constraint_b = sharded.clone();
        let constraint_c = replicated;

        // Three staged sharding constraints compose inside one closure. Each emits a `sdy.sharding_constraint` op into
        // the same MLIR program. After trace+compile, the executable runs all three in one PJRT dispatch.
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile(
            move |x| {
                let a = crate::experimental::shard_map::with_sharding_constraint(x, constraint_a.clone()).unwrap();
                let b =
                    crate::experimental::shard_map::with_sharding_constraint(a.sin(), constraint_b.clone()).unwrap();
                crate::experimental::shard_map::with_sharding_constraint(b.sin(), constraint_c.clone()).unwrap()
            },
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        // One compile means one cache entry for the whole pipeline.
        assert_eq!(engine.cache_size(), 1, "three staged reshards should compile into one program");

        let values = [0.1f32, 0.2, 0.3, 0.4];
        let source =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();
        let output = compiled.interpret(source).unwrap();

        // Final output is replicated (last constraint) — every device sees the full vector.
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let shard_bytes = output
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        let observed = values_from_bytes::<f32>(shard_bytes.as_slice());
        for (got, &input) in observed.iter().zip(values.iter()) {
            let expected = input.sin().sin();
            assert!((got - expected).abs() < 1e-5, "got {got}, expected ~{expected}");
        }
    }

    /// Composing `grad` with a staged sharding constraint: the gradient flows through the sharding constraint via
    /// [`WithShardingConstraintOperation`]'s linear transpose, mirroring JAX's
    /// `jax.grad(jax.compile(... with_sharding_constraint ...))` behavior.
    #[test]
    fn test_jit_with_grad_through_sharding_constraint_runs() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new()), None, Some(sharding.clone())).unwrap();

        // d/dx sin(with_sharding_constraint(x, S)) = cos(x), because the constraint is the identity at the value
        // level — its linear transpose is the identity, so the gradient passes through.
        let target_sharding = sharding.clone();
        let primal: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile(
            move |x| {
                crate::experimental::shard_map::with_sharding_constraint(x, target_sharding.clone()).unwrap().sin()
            },
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> = primal.value_and_gradient().unwrap();

        let input_value = 0.5f32;
        let source = Array::from_host_buffer(
            &client,
            input_type,
            mesh.clone(),
            values_to_bytes::<f32>([input_value].as_slice()).as_slice(),
        )
        .unwrap();
        let output = compiled.interpret(source).unwrap();

        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let shard_bytes = output
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        let observed = values_from_bytes::<f32>(shard_bytes.as_slice());
        assert_eq!(observed.len(), 1);
        let expected = input_value.cos();
        assert!(
            (observed[0] - expected).abs() < 1e-5,
            "expected d/dx sin(with_sharding_constraint(x, S)) ~= cos({input_value}) = {expected}, got {}",
            observed[0],
        );
    }
}
