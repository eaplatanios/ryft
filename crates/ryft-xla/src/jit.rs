//! User-facing `jit` compilation API.
//!
//! [`jit`] is the `ryft` analogue of `jax.jit`: it accepts a function over staged tracers, an
//! abstract description of the function's input types, and returns a [`CompiledFunction`] handle
//! that compiles the program once and then executes it on subsequent calls.
//!
//! The handle reuses [`CompilationContext`]'s in-memory LRU cache (and, when configured, the
//! disk cache), so repeat `jit(f, ..)` invocations at the same call site with the same input
//! type signature reuse the compiled executable without paying the trace + lower + compile cost.
//!
//! # Composition with transforms
//!
//! `ryft`'s functional transforms — `grad`, `jvp`, `vjp`, `vmap` — take primal values and return
//! values (or staged programs), so the JAX-style composition `jit(grad(f))` does not literally
//! apply. The idiomatic Rust pattern is to invoke the transform **inside** the function body
//! passed to `jit`, so the transform is traced as part of the staged program:
//!
//! ```ignore
//! let domain = XlaDomain::token();
//! let compiled = jit(
//!     move |x: ShardMapTracer| domain.grad(move |y| y.sin(), x).unwrap(),
//!     input_type,
//!     &context,
//!     mesh,
//! )?;
//! ```
//!
//! This composes naturally with all of the transforms because tracers are first-class values
//! that flow through them, and `jit`'s tracing of the closure preserves the transform's
//! semantics in the lowered MLIR.

use std::sync::Arc;

use ryft_core::parameters::{Parameterized, ParameterizedFamily};
use ryft_core::sharding::{DeviceMesh, Sharding};
use ryft_core::types::ArrayType;
use ryft_pjrt::LoadedExecutable;
use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions};

use crate::compilation::{CompilationKey, FunctionFingerprint};
use crate::experimental::domains::{XlaDomain, XlaDomainError};
use crate::experimental::shard_map::{ShardMapTensor, ShardMapTracer, TracedXlaProgram, trace};
use crate::{Array, CompilationContext};

/// Optional knobs for [`jit_with_options`]. Mirrors a subset of `jax.jit`'s keyword arguments.
///
/// Construct with struct-literal syntax plus [`Default::default`] for forward-compatibility:
///
/// ```ignore
/// let options = JitOptions {
///     donate_argnums: vec![0],
///     ..Default::default()
/// };
/// ```
#[derive(Default, Clone, Debug)]
pub struct JitOptions {
    /// Flat-input indices whose buffers should be donated to the compiled program. Donated
    /// buffers may be reused by the executor for the output buffers, and are no longer
    /// observable to the caller after the call returns. Mirrors `jax.jit`'s `donate_argnums`.
    /// Defaults to no donation.
    pub donate_argnums: Vec<usize>,

    /// Opaque hash of any state captured by the function's closure that should partition the
    /// compile cache. Mixed into the [`FunctionFingerprint::Composite`] used as the cache key so
    /// that repeat [`jit_with_options`] invocations at the same source location with different
    /// captured state get distinct cache entries. Defaults to `0` (no contribution).
    ///
    /// JAX's `static_argnums` keys the cache on each static argument's value identity. Rust's
    /// closures capture state implicitly and there is no language-level "value identity" we
    /// could read; we let the caller hash the relevant state themselves and pass the digest
    /// here. Any stable hasher works — for example [`std::collections::hash_map::DefaultHasher`]
    /// applied to the captured tuple.
    pub static_args_hash: u64,

    /// Optional override for input shardings. When `Some`, replaces the [`Sharding`] metadata on
    /// each [`ArrayType`] in `input_types` before tracing. Length must equal the flat input
    /// arity or [`jit_with_options`] returns
    /// [`XlaDomainError::InvalidJitOptions`](crate::experimental::domains::XlaDomainError::InvalidJitOptions).
    /// Mirrors `jax.jit`'s `in_shardings`.
    ///
    /// Inputs whose runtime [`Sharding`] doesn't match the override (or, in the no-override
    /// case, the `input_types`' shardings) are silently resharded to match at the
    /// [`CompiledFunction::call`] boundary via [`Array::to`]. Matching inputs skip
    /// the reshard entirely — the implicit-reshard path is the cold path.
    pub in_shardings: Option<Vec<Sharding>>,

    /// Optional override for output shardings. When `Some`, replaces the [`Sharding`] metadata
    /// on each [`ArrayType`] in the traced output type tree before lowering. Length must equal
    /// the flat output arity or [`jit_with_options`] returns
    /// [`XlaDomainError::InvalidJitOptions`](crate::experimental::domains::XlaDomainError::InvalidJitOptions).
    /// Mirrors `jax.jit`'s `out_shardings`.
    ///
    /// The override is plumbed into the func-result `sdy.sharding` attributes that drive
    /// Shardy's SPMD partitioning; the resulting executable produces per-device output buffers
    /// shaped for the overridden sharding. The override does **not** rewrite the traced body to
    /// insert `with_sharding_constraint` ops at the function's tail — for cases where the SPMD
    /// partitioner needs that hint, callers can wrap the relevant outputs inside their closure
    /// via [`with_sharding_constraint`](crate::experimental::shard_map::with_sharding_constraint).
    pub out_shardings: Option<Vec<Sharding>>,
}

/// Just-in-time compiled function handle. Returned by [`jit`] and [`jit_with_options`].
///
/// Holds the cached PJRT executable plus the input / output type metadata needed to marshal a
/// [`Parameterized`] tree of [`Array`]s into the executable and reassemble the outputs back into
/// the user's expected output tree shape.
///
/// `In` and `Out` mirror JAX's PyTree pattern: they describe how nested tuples / structs of
/// `Array` values are flattened into the executable's positional arguments and outputs. For a
/// function `|x: Tracer| -> Tracer`, `In = Out = ArrayType`. For `|(a, b): (Tracer, Tracer)| ->
/// (Tracer, Tracer)`, `In = Out = (ArrayType, ArrayType)`. The
/// [`#[derive(Parameter)]`](ryft_macros::Parameter) macro lets users define their own nested
/// types.
pub struct CompiledFunction<'c, In, Out>
where
    In: Parameterized<ArrayType>,
    Out: Parameterized<ArrayType>,
{
    /// Compiled PJRT executable. Shared with the [`CompilationContext`]'s in-memory cache.
    executable: Arc<LoadedExecutable<'c>>,

    /// PyTree shape of the output. Used by [`Self::call`] to reassemble the executor's flat
    /// output buffer list back into the user's expected output tree.
    output_structure: Out::ParameterStructure,

    /// Flat output [`ArrayType`]s in the same order the executor returns its outputs. Used to
    /// drive `XlaDomain::execute_with_donation`.
    output_types: Vec<ArrayType>,

    /// Flat per-input donation flags derived from [`JitOptions::donate_argnums`] at jit time.
    /// `true` at index `i` marks input `i` as donatable on every [`Self::call`].
    donation_flags: Vec<bool>,

    /// Per-input expected sharding. [`Self::call`] silently reshards any input whose runtime
    /// [`Sharding`] doesn't match its corresponding entry here, mirroring `jax.jit`'s implicit
    /// reshard at the function boundary. Populated from the traced `input_types`, with
    /// [`JitOptions::in_shardings`] overrides folded in.
    expected_input_shardings: Vec<Sharding>,

    /// Domain configured with the target mesh and SPMD compilation options.
    domain: XlaDomain<'c>,

    /// Holds `In` type parameter for type-system tracking.
    _input: std::marker::PhantomData<fn(In)>,
}

impl<'c, In, Out> CompiledFunction<'c, In, Out>
where
    In: Parameterized<ArrayType>,
    Out: Parameterized<ArrayType>,
    In::Family: ParameterizedFamily<Array<'c>>,
    Out::Family: ParameterizedFamily<Array<'c>>,
{
    /// Invokes the compiled program with `inputs`, a [`Parameterized`] tree of [`Array`]s
    /// matching the `In` shape used at [`jit`] time. Returns a `Parameterized` tree of [`Array`]s
    /// in the `Out` shape.
    ///
    /// PJRT's async dispatch model applies: the returned arrays may wrap pending execution
    /// events, and subsequent operations chain on those events transparently. Synchronize
    /// explicitly via `Buffer::copy_to_host(...)?.r#await()` when the host needs to observe
    /// values.
    ///
    /// Inputs whose flat indices appear in [`JitOptions::donate_argnums`] are donated to the
    /// executor and must not be used after this call returns.
    pub fn call(&self, inputs: In::To<Array<'c>>) -> Result<Out::To<Array<'c>>, XlaDomainError>
    where
        In::To<Array<'c>>: Parameterized<Array<'c>>,
        Out::To<Array<'c>>:
            Parameterized<Array<'c>, Family = Out::Family, ParameterStructure = Out::ParameterStructure>,
    {
        let inputs_vec: Vec<Array<'c>> = inputs.into_parameters().collect();
        let inputs_vec = self.reshard_inputs_if_needed(inputs_vec)?;
        let outputs_vec = self.domain.execute_with_donation(
            self.executable.as_ref(),
            inputs_vec,
            self.donation_flags.as_slice(),
            self.output_types.as_slice(),
        )?;
        Out::To::<Array<'c>>::from_parameters(self.output_structure.clone(), outputs_vec)
            .map_err(|error| XlaDomainError::Array(error.into()))
    }

    /// Compares each input's runtime [`Sharding`] against the expected sharding captured at
    /// jit time, and reshards mismatched inputs in place. Mirrors `jax.jit`'s implicit reshard
    /// behavior at the function boundary.
    ///
    /// Reshards go through a fresh [`CompilationContext`] each call, so the reshard executable
    /// is not cached across [`Self::call`] invocations. The implicit-reshard path is the cold
    /// path: well-formed callers pass matching inputs and skip this work entirely.
    fn reshard_inputs_if_needed(&self, inputs: Vec<Array<'c>>) -> Result<Vec<Array<'c>>, XlaDomainError> {
        let needs_reshard = inputs
            .iter()
            .zip(&self.expected_input_shardings)
            .any(|(array, expected)| array.sharding() != expected);
        if !needs_reshard {
            return Ok(inputs);
        }
        let context = CompilationContext::new(self.domain.client());
        let mesh = self.domain.mesh().clone();
        inputs
            .into_iter()
            .zip(&self.expected_input_shardings)
            .map(|(array, expected)| {
                if array.sharding() == expected {
                    Ok(array)
                } else {
                    let target = crate::arrays_v0::DevicePutTarget::placement(mesh.clone(), expected.clone())
                        .map_err(XlaDomainError::Array)?;
                    array.to(&context, target, false).map_err(XlaDomainError::Array)
                }
            })
            .collect()
    }

    /// Returns the flat output [`ArrayType`]s in the order the executor produces them. Useful
    /// when callers want to inspect or reuse the abstract result shape without invoking
    /// [`Self::call`].
    #[inline]
    pub fn output_types(&self) -> &[ArrayType] {
        &self.output_types
    }
}

/// Compiles `function` once and returns a [`CompiledFunction`] that executes it on subsequent
/// calls. Mirrors `jax.jit`.
///
/// On the first call to `jit` at a given source location with a given input signature, the
/// function is traced into a `TracedXlaProgram`, lowered to StableHLO + Shardy MLIR, and compiled
/// via PJRT. The compiled executable is cached in `context` keyed by
/// `(call-site source location, input types, output types, mesh, compilation options)`. Repeat
/// `jit` invocations at the same call site with the same input shapes reuse the cached
/// executable and skip the trace + lower + compile work entirely.
///
/// Equivalent to [`jit_with_options`] called with [`JitOptions::default`].
///
/// # Parameters
///
///   - `function`: The function to compile, expressed as a closure over [`ShardMapTracer`]
///     inputs returning a `Parameterized` tree of tracers.
///   - `input_types`: Abstract input value types (shape, dtype, sharding) for each input slot.
///     A [`Parameterized<ArrayType>`] tree whose shape matches the closure's input tuple shape.
///   - `context`: The [`CompilationContext`] whose cache and PJRT client back this compilation.
///   - `mesh`: Concrete device mesh the compiled program runs against.
#[track_caller]
pub fn jit<'c, F, In, Out>(
    function: F,
    input_types: In,
    context: &CompilationContext<'c>,
    mesh: DeviceMesh,
) -> Result<CompiledFunction<'c, In, Out>, XlaDomainError>
where
    F: FnOnce(In::To<ShardMapTracer>) -> Out::To<ShardMapTracer>,
    In: Parameterized<ArrayType>,
    Out: Parameterized<ArrayType>,
    In::Family:
        ParameterizedFamily<ShardMapTensor> + ParameterizedFamily<ShardMapTracer> + ParameterizedFamily<Array<'c>>,
    Out::Family:
        ParameterizedFamily<ShardMapTensor> + ParameterizedFamily<ShardMapTracer> + ParameterizedFamily<Array<'c>>,
    Out::To<ShardMapTracer>:
        Parameterized<ShardMapTracer, To<ArrayType> = Out, To<ShardMapTensor> = Out::To<ShardMapTensor>>,
{
    jit_with_options(function, input_types, context, mesh, JitOptions::default())
}

/// Same as [`jit`] but accepts a [`JitOptions`] payload for JAX-style configuration: argument
/// donation, captured-state fingerprinting, and explicit input-sharding overrides.
///
/// `options` is consumed; reuse it by `clone`ing if you need to. See [`JitOptions`] for the
/// individual knobs and how they map onto JAX's keyword arguments.
#[track_caller]
pub fn jit_with_options<'c, F, In, Out>(
    function: F,
    input_types: In,
    context: &CompilationContext<'c>,
    mesh: DeviceMesh,
    options: JitOptions,
) -> Result<CompiledFunction<'c, In, Out>, XlaDomainError>
where
    F: FnOnce(In::To<ShardMapTracer>) -> Out::To<ShardMapTracer>,
    In: Parameterized<ArrayType>,
    Out: Parameterized<ArrayType>,
    In::Family:
        ParameterizedFamily<ShardMapTensor> + ParameterizedFamily<ShardMapTracer> + ParameterizedFamily<Array<'c>>,
    Out::Family:
        ParameterizedFamily<ShardMapTensor> + ParameterizedFamily<ShardMapTracer> + ParameterizedFamily<Array<'c>>,
    Out::To<ShardMapTracer>:
        Parameterized<ShardMapTracer, To<ArrayType> = Out, To<ShardMapTensor> = Out::To<ShardMapTensor>>,
{
    // Capture the call site BEFORE doing anything else, so `#[track_caller]` propagates correctly.
    let base_fingerprint = FunctionFingerprint::from_caller();
    let fingerprint = if options.static_args_hash == 0 {
        base_fingerprint
    } else {
        FunctionFingerprint::Composite { base: Box::new(base_fingerprint), extra: options.static_args_hash }
    };

    // Apply the in-shardings override (if any) before tracing — the override changes the input
    // ArrayTypes that the SPMD lowering will see, which in turn changes the executable's
    // expected input layouts and the cache key.
    let input_types = if let Some(ref in_shardings) = options.in_shardings {
        apply_in_shardings_override(input_types, in_shardings)?
    } else {
        input_types
    };

    // Trace the user's function into an MLIR-bound `TracedXlaProgram`. This consumes
    // `input_types` and produces the canonical output type tree.
    let traced: TracedXlaProgram<In, Out> = trace(function, input_types)?;
    let input_types_vec: Vec<ArrayType> = traced.global_input_types().parameters().cloned().collect();
    let mut output_types_vec: Vec<ArrayType> = traced.global_output_types().parameters().cloned().collect();
    let output_structure = traced.global_output_types().parameter_structure();

    // Apply the out-shardings override, if provided, by rewriting each output ArrayType's
    // sharding. The downstream `result_shardings` extraction below propagates this into the
    // func-result `sdy.sharding` attributes and into the `Array`s returned by `CompiledFunction::call`.
    if let Some(ref out_shardings) = options.out_shardings {
        if out_shardings.len() != output_types_vec.len() {
            return Err(XlaDomainError::InvalidJitOptions {
                reason: format!(
                    "out_shardings has {} entries but the function has {} flat output(s)",
                    out_shardings.len(),
                    output_types_vec.len(),
                ),
            });
        }
        for (array_type, sharding) in output_types_vec.iter_mut().zip(out_shardings) {
            *array_type = ArrayType::new(
                array_type.data_type(),
                array_type.shape().clone(),
                array_type.layout().cloned(),
                Some(sharding.clone()),
            )
            .map_err(|error| XlaDomainError::Array(error.into()))?;
        }
    }

    // Validate `donate_argnums` against the function's flat input arity and materialize the flat
    // donation flag vector that `CompiledFunction::call` will pass to the executor.
    let donation_flags = build_donation_flags(input_types_vec.len(), &options.donate_argnums)?;

    // Configure the PJRT execution domain with mesh-derived SPMD options.
    let compilation_options = jit_compilation_options(context.base_options(), mesh.devices().len());
    let domain = XlaDomain::with_compilation_options(context.client(), mesh.clone(), compilation_options);

    // Structural cache key: same shape JAX's `jit` cache uses (function fingerprint + abstract
    // input/output value signatures + mesh + compile options). On cache hit, the closure passed
    // to `get_or_compile` is never invoked, so trace + lower are skipped.
    // Extract per-input and per-output shardings to drive SPMD partitioning. The XLA SPMD
    // partitioner reads `sdy.sharding` attributes on func args and results to slice buffers per
    // device; if we lowered with `None, None` here, the partitioner would emit a program that
    // expects the unsharded global tensor, and inputs sharded at the call boundary would fail
    // PJRT's input-shape check.
    let arg_shardings: Option<Vec<Sharding>> =
        input_types_vec.iter().map(|array_type| array_type.sharding().cloned()).collect::<Option<Vec<_>>>();
    let result_shardings: Option<Vec<Sharding>> =
        output_types_vec.iter().map(|array_type| array_type.sharding().cloned()).collect::<Option<Vec<_>>>();

    let cache_key =
        CompilationKey { fingerprint, input_types: &input_types_vec, output_types: &output_types_vec, mesh: &mesh };
    let executable =
        context.get_or_compile(&cache_key, domain.compilation_options(), || -> Result<String, XlaDomainError> {
            domain
                .lower_with_signature_shardings(&traced, "main", arg_shardings.as_deref(), result_shardings.as_deref())
                .map_err(XlaDomainError::from)
        })?;

    // Capture expected per-input shardings for the implicit reshard path in `CompiledFunction::call`.
    // Inputs that arrive at `call` with a different sharding are silently resharded to match,
    // mirroring `jax.jit`. Inputs whose `ArrayType` has no sharding are tolerated by recording a
    // fully-replicated fallback over the mesh — those inputs are skipped by the equality check
    // below as long as the caller-supplied sharding equals replicated, which is the only sensible
    // interpretation of "no sharding" against a multi-device executable.
    let expected_input_shardings: Vec<Sharding> = input_types_vec
        .iter()
        .map(|array_type| {
            array_type
                .sharding()
                .cloned()
                .unwrap_or_else(|| Sharding::replicated(mesh.logical_mesh().clone(), array_type.shape().rank()))
        })
        .collect();

    Ok(CompiledFunction {
        executable,
        output_structure,
        output_types: output_types_vec,
        donation_flags,
        expected_input_shardings,
        domain,
        _input: std::marker::PhantomData,
    })
}

/// Traces `function` against `input_types` and returns the abstract output type tree, without
/// lowering or compiling. Mirrors `jax.eval_shape`.
///
/// Useful for inspecting the output shape and sharding of a function before paying the
/// trace-and-compile cost — e.g. when sizing buffers, building a higher-level execution graph,
/// or validating that a function's output sharding matches an expected layout.
#[track_caller]
pub fn eval_shape<F, In, Out>(function: F, input_types: In) -> Result<Out, XlaDomainError>
where
    F: FnOnce(In::To<ShardMapTracer>) -> Out::To<ShardMapTracer>,
    In: Parameterized<ArrayType>,
    Out: Parameterized<ArrayType>,
    In::Family: ParameterizedFamily<ShardMapTensor> + ParameterizedFamily<ShardMapTracer>,
    Out::Family: ParameterizedFamily<ShardMapTensor> + ParameterizedFamily<ShardMapTracer>,
    Out::To<ShardMapTracer>:
        Parameterized<ShardMapTracer, To<ArrayType> = Out, To<ShardMapTensor> = Out::To<ShardMapTensor>>,
{
    let traced: TracedXlaProgram<In, Out> = trace(function, input_types)?;
    let structure = traced.global_output_types().parameter_structure();
    let flat: Vec<ArrayType> = traced.global_output_types().parameters().cloned().collect();
    Out::from_parameters(structure, flat).map_err(|error| XlaDomainError::Array(error.into()))
}

/// Replaces the [`Sharding`] metadata on every [`ArrayType`] leaf of `input_types` with the
/// corresponding entry of `in_shardings`. Errors when arities disagree or when a substituted
/// sharding has the wrong rank for its array type.
fn apply_in_shardings_override<In>(input_types: In, in_shardings: &[Sharding]) -> Result<In, XlaDomainError>
where
    In: Parameterized<ArrayType>,
{
    let structure = input_types.parameter_structure();
    let flat: Vec<ArrayType> = input_types.into_parameters().collect();
    if flat.len() != in_shardings.len() {
        return Err(XlaDomainError::InvalidJitOptions {
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

/// Expands `donate_argnums` into a flat `Vec<bool>` of length `input_arity`. Errors when any
/// index is out of range, or when an index appears more than once.
fn build_donation_flags(input_arity: usize, donate_argnums: &[usize]) -> Result<Vec<bool>, XlaDomainError> {
    let mut flags = vec![false; input_arity];
    for &index in donate_argnums {
        if index >= input_arity {
            return Err(XlaDomainError::InvalidJitOptions {
                reason: format!(
                    "donate_argnums contains index {index} but the function has only {input_arity} flat input(s)",
                ),
            });
        }
        if flags[index] {
            return Err(XlaDomainError::InvalidJitOptions {
                reason: format!("donate_argnums contains duplicate index {index}"),
            });
        }
        flags[index] = true;
    }
    Ok(flags)
}

/// Overlays the SPMD partitioning fields required by [`jit`]-compiled programs onto a base
/// [`CompilationOptions`] template.
fn jit_compilation_options(base: &CompilationOptions, partition_count: usize) -> CompilationOptions {
    let mut options = base.clone();
    let exec_options = options.executable_build_options.get_or_insert_with(ExecutableCompilationOptions::default);
    if exec_options.device_ordinal == 0 {
        // `0` is the protobuf default but PJRT expects `-1` to mean "use the default device".
        exec_options.device_ordinal = -1;
    }
    exec_options.replica_count = 1;
    exec_options.partition_count = partition_count as i64;
    exec_options.use_spmd_partitioning = true;
    exec_options.use_shardy_partitioner = true;
    options
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    use ryft_core::operations::trigonometric::Sin;
    use ryft_core::sharding::{Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, Sharding};
    use ryft_core::types::data_types::DataType;
    use ryft_core::types::{ArrayType, Shape, Size};
    use ryft_pjrt::{ClientOptions, CpuClientOptions, load_cpu_plugin};

    use crate::experimental::domains::{XlaDomain, XlaDomainError};
    use crate::tests::{values_from_bytes, values_to_bytes};
    use crate::{Array, CompilationContext, CompiledFunction, FromPjrt, JitOptions, eval_shape, jit, jit_with_options};

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

    #[test]
    fn test_jit_unary_function_runs_end_to_end() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let context = CompilationContext::new(&client);

        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(4)]),
            None,
            Some(Sharding::replicated(mesh.logical_mesh().clone(), 1)),
        )
        .unwrap();
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> =
            jit(|x| x.sin(), input_type.clone(), &context, mesh.clone()).unwrap();

        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();
        let output = compiled.call(source).unwrap();

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
    fn test_jit_binary_function_with_tuple_input() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let context = CompilationContext::new(&client);

        let shape = Shape::new(vec![Size::Static(3)]);
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let input_type = ArrayType::new(DataType::F32, shape.clone(), None, Some(sharding.clone())).unwrap();
        let compiled: CompiledFunction<'_, (ArrayType, ArrayType), ArrayType> =
            jit(|(a, b)| a + b, (input_type.clone(), input_type.clone()), &context, mesh.clone()).unwrap();

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
        let output = compiled.call((a, b)).unwrap();

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
        let context = CompilationContext::new(&client);
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(4)]),
            None,
            Some(Sharding::replicated(mesh.logical_mesh().clone(), 1)),
        )
        .unwrap();

        assert_eq!(context.cache_size(), 0);
        // Two `jit` invocations on the same source line (inside the loop body) share a call-site
        // fingerprint, so the second invocation hits the cache instead of compiling again.
        for _ in 0..2 {
            let _: CompiledFunction<'_, ArrayType, ArrayType> =
                jit(|x| x.sin(), input_type.clone(), &context, mesh.clone()).unwrap();
        }
        assert_eq!(context.cache_size(), 1, "repeat jit at the same call site should hit the cache");
    }

    #[test]
    fn test_jit_distinct_call_sites_use_distinct_cache_entries() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let context = CompilationContext::new(&client);
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(4)]),
            None,
            Some(Sharding::replicated(mesh.logical_mesh().clone(), 1)),
        )
        .unwrap();

        assert_eq!(context.cache_size(), 0);
        // Two `jit` invocations at distinct source lines populate two cache entries even when
        // the closure and inputs are identical, mirroring the way JAX's compile cache keys on
        // function identity (which differs per Python `id()` even for source-equivalent
        // lambdas).
        let _: CompiledFunction<'_, ArrayType, ArrayType> =
            jit(|x| x.sin(), input_type.clone(), &context, mesh.clone()).unwrap();
        let _: CompiledFunction<'_, ArrayType, ArrayType> = jit(|x| x.sin(), input_type, &context, mesh).unwrap();
        assert_eq!(context.cache_size(), 2);
    }

    #[test]
    fn test_jit_with_options_donates_argument() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let context = CompilationContext::new(&client);
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(3)]),
            None,
            Some(Sharding::replicated(mesh.logical_mesh().clone(), 1)),
        )
        .unwrap();
        let options = JitOptions { donate_argnums: vec![0], ..Default::default() };
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> =
            jit_with_options(|x| x.sin(), input_type.clone(), &context, mesh.clone(), options).unwrap();

        let values = [0.0f32, 0.5, 1.0];
        let source =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();
        let output = compiled.call(source).unwrap();

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
    fn test_jit_with_options_rejects_out_of_range_donate_argnum() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let context = CompilationContext::new(&client);
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(2)]),
            None,
            Some(Sharding::replicated(mesh.logical_mesh().clone(), 1)),
        )
        .unwrap();
        let options = JitOptions { donate_argnums: vec![5], ..Default::default() };
        let result: Result<CompiledFunction<'_, ArrayType, ArrayType>, XlaDomainError> =
            jit_with_options(|x| x.sin(), input_type, &context, mesh, options);
        assert!(matches!(result, Err(XlaDomainError::InvalidJitOptions { .. })));
    }

    #[test]
    fn test_jit_with_options_static_args_hash_partitions_cache() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let context = CompilationContext::new(&client);
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(4)]),
            None,
            Some(Sharding::replicated(mesh.logical_mesh().clone(), 1)),
        )
        .unwrap();

        // Identical call site, identical closures, identical input types — but different
        // `static_args_hash` values, so the cache must place each compile under its own entry.
        let hash_for = |seed: &str| {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            hasher.finish()
        };
        for seed in ["a", "b"] {
            let options = JitOptions { static_args_hash: hash_for(seed), ..Default::default() };
            let _: CompiledFunction<'_, ArrayType, ArrayType> =
                jit_with_options(|x| x.sin(), input_type.clone(), &context, mesh.clone(), options).unwrap();
        }
        assert_eq!(context.cache_size(), 2);
    }

    #[test]
    fn test_jit_with_options_in_shardings_override_replaces_input_sharding() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = two_device_mesh(&client);
        let context = CompilationContext::new(&client);

        // input_type carries the abstract shape & dtype but a "wrong" sharding (replicated). The
        // `in_shardings` override replaces it with a 2-way shard along "x" before tracing, so
        // the compiled program shards the input across the 2-device mesh.
        let shape = Shape::new(vec![Size::Static(4)]);
        let abstract_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let abstract_input_type = ArrayType::new(DataType::F32, shape.clone(), None, Some(abstract_sharding)).unwrap();
        let sharded =
            Sharding::new(mesh.logical_mesh().clone(), vec![ryft_core::sharding::ShardingDimension::sharded(["x"])])
                .unwrap();
        let options = JitOptions { in_shardings: Some(vec![sharded.clone()]), ..Default::default() };
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> =
            jit_with_options(|x| x.sin(), abstract_input_type, &context, mesh.clone(), options).unwrap();

        // Build the input array under the overridden sharding so it matches the executable's
        // expected layout.
        let input_type = ArrayType::new(DataType::F32, shape, None, Some(sharded)).unwrap();
        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();
        let output = compiled.call(source).unwrap();

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
    fn test_jit_with_options_rejects_in_shardings_arity_mismatch() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let context = CompilationContext::new(&client);
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(2)]),
            None,
            Some(Sharding::replicated(mesh.logical_mesh().clone(), 1)),
        )
        .unwrap();
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        // Two shardings for one flat input — should fail.
        let options = JitOptions { in_shardings: Some(vec![sharding.clone(), sharding]), ..Default::default() };
        let result: Result<CompiledFunction<'_, ArrayType, ArrayType>, XlaDomainError> =
            jit_with_options(|x| x.sin(), input_type, &context, mesh, options);
        assert!(matches!(result, Err(XlaDomainError::InvalidJitOptions { .. })));
    }

    #[test]
    fn test_jit_with_options_out_shardings_override_propagates_to_output_array() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = two_device_mesh(&client);
        let context = CompilationContext::new(&client);

        let shape = Shape::new(vec![Size::Static(4)]);
        let sharded =
            Sharding::new(mesh.logical_mesh().clone(), vec![ryft_core::sharding::ShardingDimension::sharded(["x"])])
                .unwrap();
        let input_type = ArrayType::new(DataType::F32, shape.clone(), None, Some(sharded.clone())).unwrap();
        // Override the output sharding to the same 2-way shard along "x" so the partitioner
        // emits a fully-sharded output and `Array`'s sharding metadata matches.
        let options = JitOptions { out_shardings: Some(vec![sharded.clone()]), ..Default::default() };
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> =
            jit_with_options(|x| x.sin(), input_type.clone(), &context, mesh.clone(), options).unwrap();

        // The returned Array should carry the overridden sharding.
        assert_eq!(compiled.output_types()[0].sharding(), Some(&sharded));

        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();
        let output = compiled.call(source).unwrap();
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
        let context = CompilationContext::new(&client);

        // The executable expects a 2-way shard along "x", but the caller will pass a fully
        // replicated array. `CompiledFunction::call` should silently reshard before executing.
        let shape = Shape::new(vec![Size::Static(4)]);
        let sharded =
            Sharding::new(mesh.logical_mesh().clone(), vec![ryft_core::sharding::ShardingDimension::sharded(["x"])])
                .unwrap();
        let input_type = ArrayType::new(DataType::F32, shape.clone(), None, Some(sharded.clone())).unwrap();
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> =
            jit(|x| x.sin(), input_type.clone(), &context, mesh.clone()).unwrap();

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
        let output = compiled.call(source).unwrap();
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
    fn test_jit_with_options_rejects_out_shardings_arity_mismatch() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let context = CompilationContext::new(&client);
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(2)]),
            None,
            Some(Sharding::replicated(mesh.logical_mesh().clone(), 1)),
        )
        .unwrap();
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let options = JitOptions { out_shardings: Some(vec![sharding.clone(), sharding]), ..Default::default() };
        let result: Result<CompiledFunction<'_, ArrayType, ArrayType>, XlaDomainError> =
            jit_with_options(|x| x.sin(), input_type, &context, mesh, options);
        assert!(matches!(result, Err(XlaDomainError::InvalidJitOptions { .. })));
    }

    #[test]
    fn test_eval_shape_returns_output_types_without_compiling() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let context = CompilationContext::new(&client);
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(7)]),
            None,
            Some(Sharding::replicated(mesh.logical_mesh().clone(), 1)),
        )
        .unwrap();

        let output_type: ArrayType = eval_shape(|x| x.sin(), input_type.clone()).unwrap();
        assert_eq!(output_type.data_type(), DataType::F32);
        assert_eq!(output_type.shape(), input_type.shape());
        // `eval_shape` must not have populated the compile cache.
        assert_eq!(context.cache_size(), 0);
    }

    #[test]
    fn test_jit_with_grad_inside_closure_compiles_and_runs() {
        use ryft_core::tracing_v2::DifferentiableDomain;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let context = CompilationContext::new(&client);
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new()), None, Some(sharding.clone())).unwrap();

        // `jit` composes with `grad` by invoking the transform *inside* the staged closure. The
        // tracing system records `grad`'s lowering into the same MLIR module that `jit`
        // compiles, so the resulting executable computes `d/dx sin(x) = cos(x)` directly.
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> = jit(
            |x: crate::experimental::shard_map::ShardMapTracer| {
                XlaDomain::token().grad(|y: crate::experimental::shard_map::ShardMapTracer| y.sin(), x).unwrap()
            },
            input_type.clone(),
            &context,
            mesh.clone(),
        )
        .unwrap();

        let input_value = 0.75f32;
        let source = Array::from_host_buffer(
            &client,
            input_type,
            mesh.clone(),
            values_to_bytes::<f32>([input_value].as_slice()).as_slice(),
        )
        .unwrap();
        let output = compiled.call(source).unwrap();

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

    /// Verifies that the staged `to` operation works inside a `jit`-compiled function: the
    /// sharding constraint is preserved through the trace, and the output array carries the
    /// constrained sharding on each device.
    #[test]
    fn test_jit_with_staged_to_constrains_output_sharding() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = two_device_mesh(&client);
        let context = CompilationContext::new(&client);

        let shape = Shape::new(vec![Size::Static(4)]);
        let sharded =
            Sharding::new(mesh.logical_mesh().clone(), vec![ryft_core::sharding::ShardingDimension::sharded(["x"])])
                .unwrap();
        let input_type = ArrayType::new(DataType::F32, shape.clone(), None, Some(sharded.clone())).unwrap();
        let target_sharding = sharded.clone();

        // The user invokes `to` directly inside the staged closure — it's compiled into the same
        // MLIR program as the rest of the function body.
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> = jit(
            move |x: crate::experimental::shard_map::ShardMapTracer| {
                let constrained =
                    crate::experimental::shard_map::to(x, target_sharding.clone()).expect("staged to should succeed");
                constrained.sin()
            },
            input_type.clone(),
            &context,
            mesh.clone(),
        )
        .unwrap();

        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();
        let output = compiled.call(source).unwrap();
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

    /// Multiple staged `to` calls inside one `jit` body compile into a single MLIR program with
    /// chained `sdy.sharding_constraint` ops — exactly one cache entry, exactly one PJRT execute
    /// per call. This is the async-pipelined regime: PJRT runs the whole compiled program in
    /// one shot without per-reshard host sync.
    #[test]
    fn test_jit_with_chained_staged_to_calls_compiles_to_one_program() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = two_device_mesh(&client);
        let context = CompilationContext::new(&client);

        let shape = Shape::new(vec![Size::Static(4)]);
        let sharded =
            Sharding::new(mesh.logical_mesh().clone(), vec![ryft_core::sharding::ShardingDimension::sharded(["x"])])
                .unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let input_type = ArrayType::new(DataType::F32, shape.clone(), None, Some(sharded.clone())).unwrap();
        let constraint_a = replicated.clone();
        let constraint_b = sharded.clone();
        let constraint_c = replicated;

        // Three staged `to` calls compose inside one closure. Each emits a
        // `sdy.sharding_constraint` op into the same MLIR program. After trace+compile, the
        // executable runs all three in one PJRT dispatch.
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> = jit(
            move |x: crate::experimental::shard_map::ShardMapTracer| {
                let a = crate::experimental::shard_map::to(x, constraint_a.clone()).unwrap();
                let b = crate::experimental::shard_map::to(a.sin(), constraint_b.clone()).unwrap();
                crate::experimental::shard_map::to(b.sin(), constraint_c.clone()).unwrap()
            },
            input_type.clone(),
            &context,
            mesh.clone(),
        )
        .unwrap();

        // One compile means one cache entry for the whole pipeline.
        assert_eq!(context.cache_size(), 1, "three staged reshards should compile into one program");

        let values = [0.1f32, 0.2, 0.3, 0.4];
        let source =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
                .unwrap();
        let output = compiled.call(source).unwrap();

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

    /// Composing `grad` with the staged `to` operation: the gradient flows through the sharding
    /// constraint via [`WithShardingConstraintOperation`]'s linear transpose, mirroring JAX's
    /// `jax.grad(jax.jit(... with_sharding_constraint ...))` behavior.
    #[test]
    fn test_jit_with_grad_through_staged_to_runs() {
        use ryft_core::tracing_v2::DifferentiableDomain;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let context = CompilationContext::new(&client);
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new()), None, Some(sharding.clone())).unwrap();

        // d/dx sin(to(x, S)) = cos(x), because `to`/`with_sharding_constraint` is the identity at
        // the value level — its linear transpose is the identity, so the gradient passes through.
        let target_sharding = sharding.clone();
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> = jit(
            move |x: crate::experimental::shard_map::ShardMapTracer| {
                let inner_sharding = target_sharding.clone();
                XlaDomain::token()
                    .grad(
                        move |y: crate::experimental::shard_map::ShardMapTracer| {
                            crate::experimental::shard_map::to(y, inner_sharding.clone()).unwrap().sin()
                        },
                        x,
                    )
                    .unwrap()
            },
            input_type.clone(),
            &context,
            mesh.clone(),
        )
        .unwrap();

        let input_value = 0.5f32;
        let source = Array::from_host_buffer(
            &client,
            input_type,
            mesh.clone(),
            values_to_bytes::<f32>([input_value].as_slice()).as_slice(),
        )
        .unwrap();
        let output = compiled.call(source).unwrap();

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
            "expected d/dx sin(to(x, S)) ~= cos({input_value}) = {expected}, got {}",
            observed[0],
        );
    }
}
