//! User-facing XLA compile-and-execute API.
//!
//! [`compile_and_execute`] is the `ryft` analogue of `jax.jit`: it traces a closure over staged
//! tracers into an XLA program, compiles it via PJRT, and returns a runtime handle that
//! executes the compiled program against [`Array`] inputs. The trace happens against the static
//! tracing-only token ([`XlaDomain::token`]) — that way users can call domain methods like
//! `.grad(...)` / `.vmap(...)` inside the closure without threading the engine's lifetime
//! through the closure body — and the resulting [`Program`](ryft_core::tracing::Program) is
//! then compiled and executed via the user-supplied [`XlaDomain`]'s internal cache.
//!
//! New backend-agnostic code that doesn't need this tracing-token convenience should prefer the
//! core pipeline at [`ryft_core::compilation::compile_and_execute_with_options`].

use std::marker::PhantomData;
use std::sync::Arc;

use ryft_core::compilation::{CompilationDomain, CompilationOptions, FunctionFingerprint};
use ryft_core::parameters::{Parameterized, ParameterizedFamily};
use ryft_core::sharding::{DeviceMesh, Sharding};
use ryft_core::tracing::domains::{Tracer, TracingDomain};
use ryft_core::tracing::programs::Program;
use ryft_core::types::{ArrayType, Typed};

use crate::Array;
use crate::experimental::domains::{XlaCompiledProgram, XlaDomain, XlaDomainError, XlaOptions};
use crate::experimental::ops::XlaOperation;
use crate::experimental::shard_map::XlaValue;

/// Just-in-time compiled function handle. Returned by [`compile_and_execute`] and
/// [`compile_and_execute_with_options`].
///
/// Holds the cached PJRT-backed [`XlaCompiledProgram`] plus the input / output type metadata
/// needed to marshal a [`Parameterized`] tree of [`Array`]s into the executable and reassemble
/// the outputs back into the user's expected output tree shape.
///
/// Also retains the **user closure** that was traced. Applying an outer transform (`grad` /
/// `jvp` / `vjp` / `vmap`) wraps the retained closure in a transform-aware closure and yields
/// a new [`CompiledXlaFunction`]; staging this function inside another [`compile_and_execute`]
/// body re-executes the retained closure against the outer trace's tracers via
/// [`Self::call_traced`]. Each call retraces (tracing is cheap); compilation hits the cache.
pub struct CompiledXlaFunction<'c, In, Out>
where
    In: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<XlaValue<'static>> + ParameterizedFamily<Tracer<'static, XlaDomain<'static>>>,
        >,
    Out: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<XlaValue<'static>> + ParameterizedFamily<Tracer<'static, XlaDomain<'static>>>,
        >,
{
    /// Compiled XLA program. Carries the loaded PJRT executable plus per-call state baked at
    /// compile time (output types, donation flags, expected input shardings, mesh).
    program: XlaCompiledProgram<'c>,

    /// Retained user closure, traced against the static [`XlaDomain::token`]. Stored as a
    /// refcounted trait object so clones share the same callable and transforms can wrap it
    /// without re-running the original closure's captures. The closure lifetime tracks the
    /// engine's `'c` so a [`CompiledXlaFunction`] can capture other [`CompiledXlaFunction`]s
    /// (whose engines also borrow from the client).
    function:
        Arc<dyn Fn(In::To<Tracer<'static, XlaDomain<'static>>>) -> Out::To<Tracer<'static, XlaDomain<'static>>> + 'c>,

    /// PyTree shape of the output. Used by [`Self::call`] to reassemble the executor's flat
    /// output buffer list back into the user's expected output tree.
    output_structure: Out::ParameterStructure,

    /// Flat output [`ArrayType`]s in executor-output order.
    output_types: Vec<ArrayType>,

    /// XLA backend used to execute the compiled program. Cloned from the context's engine so
    /// the compiled function isn't tied to the context's borrow scope.
    engine: XlaDomain<'c>,

    /// Holds the `In` type parameter for type-system tracking.
    _input: PhantomData<fn(In)>,
}

/// Backward-compatible alias for the renamed type.
pub type CompiledFunction<'c, In, Out> = CompiledXlaFunction<'c, In, Out>;

impl<'c, In, Out> Clone for CompiledXlaFunction<'c, In, Out>
where
    In: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<XlaValue<'static>> + ParameterizedFamily<Tracer<'static, XlaDomain<'static>>>,
        >,
    Out: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<XlaValue<'static>> + ParameterizedFamily<Tracer<'static, XlaDomain<'static>>>,
        >,
    Out::ParameterStructure: Clone,
{
    fn clone(&self) -> Self {
        Self {
            program: self.program.clone(),
            function: Arc::clone(&self.function),
            output_structure: self.output_structure.clone(),
            output_types: self.output_types.clone(),
            engine: self.engine.clone(),
            _input: PhantomData,
        }
    }
}

impl<'c, In, Out> CompiledXlaFunction<'c, In, Out>
where
    In: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<XlaValue<'static>> + ParameterizedFamily<Tracer<'static, XlaDomain<'static>>>,
        >,
    Out: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<XlaValue<'static>> + ParameterizedFamily<Tracer<'static, XlaDomain<'static>>>,
        >,
{
    /// Returns the flat output [`ArrayType`]s in the order the executor produces them.
    #[inline]
    pub fn output_types(&self) -> &[ArrayType] {
        &self.output_types
    }

    /// Invokes the compiled program with `inputs`, a [`Parameterized`] tree of [`Array`]s
    /// matching the `In` shape used at [`compile_and_execute`] time. Returns a `Parameterized`
    /// tree of [`Array`]s in the `Out` shape.
    pub fn call(&self, inputs: In::To<Array<'c>>) -> Result<Out::To<Array<'c>>, XlaDomainError>
    where
        In: Parameterized<ArrayType, Family: ParameterizedFamily<Array<'c>>>,
        Out: Parameterized<ArrayType, Family: ParameterizedFamily<Array<'c>>>,
        In::To<Array<'c>>: Parameterized<Array<'c>>,
        Out::To<Array<'c>>:
            Parameterized<Array<'c>, Family = Out::Family, ParameterStructure = Out::ParameterStructure>,
    {
        let inputs_vec: Vec<Array<'c>> = inputs.into_parameters().collect();
        let xla_inputs: Vec<XlaValue<'c>> = inputs_vec
            .into_iter()
            .map(|array| {
                let array_type = ArrayType::new(
                    array.data_type(),
                    ryft_core::Shape::new(
                        array.shape().as_slice().iter().copied().map(ryft_core::Size::Static).collect(),
                    ),
                    None,
                    Some(array.sharding().clone()),
                )
                .expect("source array's metadata should always be a valid ArrayType");
                XlaValue::concrete(array_type, array)
            })
            .collect();
        let outputs = CompilationDomain::execute(&self.engine, &self.program, xla_inputs)?;
        let arrays_vec: Vec<Array<'c>> = outputs
            .into_iter()
            .map(|value| {
                value.into_data().ok_or(XlaDomainError::InvalidCompilationOptions {
                    reason: "compiled program produced an abstract output (no runtime data)".to_string(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Out::To::<Array<'c>>::from_parameters(self.output_structure.clone(), arrays_vec)
            .map_err(|error| XlaDomainError::Array(error.into()))
    }

    /// Stages this compiled function into an outer trace by re-executing the retained user
    /// closure against `inputs`. Use this to nest one [`CompiledXlaFunction`] inside another
    /// [`compile_and_execute`] body, or to apply an outer transform (`grad`, `jvp`, `vjp`,
    /// `vmap`) by wrapping the call in a transform-aware closure.
    ///
    /// Mirrors how JAX inlines `jit(f)` into outer trace contexts: each call retraces `f`
    /// against the current trace, so transforms compose naturally without needing a separate
    /// program-walking pass.
    #[inline]
    pub fn call_traced(
        &self,
        inputs: In::To<Tracer<'static, XlaDomain<'static>>>,
    ) -> Out::To<Tracer<'static, XlaDomain<'static>>> {
        (self.function)(inputs)
    }
}

/// Compiles `function` once and returns a [`CompiledXlaFunction`] that executes it on subsequent
/// calls. Mirrors `jax.jit`.
///
/// Equivalent to [`compile_and_execute_with_options`] called with [`CompilationOptions::new`]
/// wrapping [`XlaOptions::new(mesh)`](XlaOptions::new).
///
/// The function is traced against [`XlaDomain::token`] (the static tracing-only domain) so
/// callers can use methods like `.grad` / `.vmap` on the token inside the closure without
/// threading an engine lifetime through the closure body. The resulting program is then
/// compiled and executed against `engine`, sharing its
/// [`CompilationContext`](ryft_core::compilation::CompilationContext) cache across repeat
/// invocations at the same source line.
#[track_caller]
pub fn compile_and_execute<'c, F, In, Out>(
    function: F,
    input_types: In,
    engine: &XlaDomain<'c>,
    mesh: DeviceMesh,
) -> Result<CompiledXlaFunction<'c, In, Out>, XlaDomainError>
where
    F: Fn(In::To<Tracer<'static, XlaDomain<'static>>>) -> Out::To<Tracer<'static, XlaDomain<'static>>> + 'c,
    In: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<XlaValue<'static>> + ParameterizedFamily<Tracer<'static, XlaDomain<'static>>>,
        >,
    In::ParameterStructure: std::hash::Hash,
    Out: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<XlaValue<'static>> + ParameterizedFamily<Tracer<'static, XlaDomain<'static>>>,
        >,
    Out::To<Tracer<'static, XlaDomain<'static>>>: Parameterized<
            Tracer<'static, XlaDomain<'static>>,
            To<ArrayType> = Out,
            To<XlaValue<'static>> = Out::To<XlaValue<'static>>,
        >,
{
    compile_and_execute_with_options::<F, In, Out>(
        function,
        input_types,
        engine,
        CompilationOptions::new(XlaOptions::new(mesh)),
    )
}

/// Same as [`compile_and_execute`] but accepts a full [`CompilationOptions`] payload for
/// JAX-style configuration: captured-state fingerprinting plus the XLA-specific [`XlaOptions`]
/// (mesh, sharding overrides, per-input buffer donation flags).
#[track_caller]
pub fn compile_and_execute_with_options<'c, F, In, Out>(
    function: F,
    input_types: In,
    engine: &XlaDomain<'c>,
    options: CompilationOptions<XlaDomain<'c>>,
) -> Result<CompiledXlaFunction<'c, In, Out>, XlaDomainError>
where
    F: Fn(In::To<Tracer<'static, XlaDomain<'static>>>) -> Out::To<Tracer<'static, XlaDomain<'static>>> + 'c,
    In: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<XlaValue<'static>> + ParameterizedFamily<Tracer<'static, XlaDomain<'static>>>,
        >,
    In::ParameterStructure: std::hash::Hash,
    Out: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<XlaValue<'static>> + ParameterizedFamily<Tracer<'static, XlaDomain<'static>>>,
        >,
    Out::To<Tracer<'static, XlaDomain<'static>>>: Parameterized<
            Tracer<'static, XlaDomain<'static>>,
            To<ArrayType> = Out,
            To<XlaValue<'static>> = Out::To<XlaValue<'static>>,
        >,
{
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
    let input_types = if let Some(ref in_shardings) = xla_options.in_shardings.clone() {
        apply_in_shardings_override(input_types, in_shardings)?
    } else {
        input_types
    };

    // Trace via the static tracing-only token. This is what allows closures like
    // `|x| XlaDomain::token().grad(..., x)` to work without threading a non-static lifetime
    // through the closure body. The trace borrows `function` immutably so it can be retained
    // in the resulting [`CompiledXlaFunction`] for transform composition / inner staging.
    let token: &'static XlaDomain<'static> = XlaDomain::token();
    let (output_types_tree, program_static) = token
        .trace::<_, In, Out::To<Tracer<'static, XlaDomain<'static>>>>(|tracers| Ok((&function)(tracers)), input_types)
        .map_err(XlaDomainError::from)?;
    let output_structure = output_types_tree.parameter_structure();
    let mut output_types_vec: Vec<ArrayType> = output_types_tree.parameters().cloned().collect();

    // Apply the out-shardings override, if provided, by rewriting each output ArrayType's
    // sharding metadata.
    if let Some(ref out_shardings) = xla_options.out_shardings {
        if out_shardings.len() != output_types_vec.len() {
            return Err(XlaDomainError::InvalidCompilationOptions {
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

    // Build the input type list from the traced program's atoms (these reflect any
    // in-shardings override that was applied above).
    let input_types_vec: Vec<ArrayType> = program_static
        .input_ids()
        .iter()
        .map(|atom_id| program_static.atoms()[atom_id.index()].r#type().into_owned())
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

    // Cache key derived from the engine. This is what makes repeat `compile_and_execute`
    // invocations at the same source location with the same inputs share a cache entry.
    let cache_key = engine.compilation_key(&function_fingerprint, &input_types_vec, &xla_options);

    // Cache lookup / on-miss compile. The static program is re-cast at the engine's lifetime
    // via the unsafe lifetime "unextension" helper — sound because traced values have
    // `data: None`, so the `'engine` lifetime in `XlaValue<'engine>` / `XlaOperation<'engine>`
    // is purely phantom for tracing-time programs.
    let cache = engine.cache().expect("XlaDomain always exposes a compile cache");
    let compiled: XlaCompiledProgram<'c> =
        cache.get_or_compile(engine, cache_key, || -> Result<XlaCompiledProgram<'c>, XlaDomainError> {
            let program_engine = unsafe { unextend_program_lifetime(&program_static) };
            engine.compile(program_engine, &xla_options)
        })?;

    Ok(CompiledXlaFunction {
        program: compiled,
        function: Arc::new(function),
        output_structure,
        output_types: output_types_vec,
        engine: engine.clone(),
        _input: PhantomData,
    })
}

/// Traces `function` against `input_types` and returns the abstract output type tree, without
/// lowering or compiling. Mirrors `jax.eval_shape`.
#[track_caller]
pub fn eval_shape<F, In, Out>(function: F, input_types: In) -> Result<Out, XlaDomainError>
where
    F: FnOnce(In::To<Tracer<'static, XlaDomain<'static>>>) -> Out::To<Tracer<'static, XlaDomain<'static>>>,
    In: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<XlaValue<'static>> + ParameterizedFamily<Tracer<'static, XlaDomain<'static>>>,
        >,
    Out: Parameterized<
            ArrayType,
            Family: ParameterizedFamily<XlaValue<'static>> + ParameterizedFamily<Tracer<'static, XlaDomain<'static>>>,
        >,
    Out::To<Tracer<'static, XlaDomain<'static>>>: Parameterized<
            Tracer<'static, XlaDomain<'static>>,
            To<ArrayType> = Out,
            To<XlaValue<'static>> = Out::To<XlaValue<'static>>,
        >,
{
    let token: &'static XlaDomain<'static> = XlaDomain::token();
    let (output_types_tree, _program) = token
        .trace::<_, In, Out::To<Tracer<'static, XlaDomain<'static>>>>(|tracers| Ok(function(tracers)), input_types)?;
    Ok(output_types_tree)
}

/// Replaces the [`Sharding`] metadata on every [`ArrayType`] leaf of `input_types`.
fn apply_in_shardings_override<In>(input_types: In, in_shardings: &[Sharding]) -> Result<In, XlaDomainError>
where
    In: Parameterized<ArrayType>,
{
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

/// Reinterprets a `'static`-lifetimed [`Program`] reference at a narrower `'engine` lifetime.
/// Sound only when the program carries purely-abstract [`XlaValue`]s (i.e. `data: None`) — which
/// is always the case for programs produced by [`TracingDomain::trace`], since the trace path
/// records type metadata and atoms without ever materializing concrete runtime arrays.
///
/// # Safety
///
/// The caller must guarantee that no atom in `program` carries concrete runtime data tied to a
/// shorter lifetime than `'engine`. For `TracingDomain::trace`-produced programs this is
/// trivially true: the trace path never assigns `data: Some(_)` to any `XlaValue`.
///
/// `'static: 'engine` always holds, and `XlaValue<'o>` / `XlaOperation<'o>` have identical
/// in-memory layouts at every lifetime (the lifetime parameter is only consumed by the
/// `data: Option<Array<'o>>` field, which is `None` for traced atoms).
unsafe fn unextend_program_lifetime<'a, 'c, Input, Output>(
    program: &'a Program<ArrayType, XlaValue<'static>, XlaOperation<'static>, Input, Output>,
) -> &'a Program<ArrayType, XlaValue<'c>, XlaOperation<'c>, Vec<XlaValue<'c>>, Vec<XlaValue<'c>>>
where
    Input: Parameterized<XlaValue<'static>>,
    Output: Parameterized<XlaValue<'static>>,
{
    // SAFETY: `XlaValue<'o>` and `XlaOperation<'o>` have identical in-memory layouts at every
    // lifetime — the lifetime parameter is consumed only by the `data: Option<Array<'o>>` field,
    // which is `None` for tracing-produced atoms. The `Input`/`Output` parameter-tree shapes
    // are not read by the lowering pipeline beyond their flattened atom list, so erasing them
    // to `Vec<XlaValue<'engine>>` is sound.
    unsafe { &*(program as *const _ as *const _) }
}
#[cfg(test)]
mod tests {
    use ryft_core::operations::trigonometric::Sin;
    use ryft_core::sharding::{Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, Sharding};
    use ryft_core::types::data_types::DataType;
    use ryft_core::types::{ArrayType, Shape, Size};
    use ryft_pjrt::{ClientOptions, CpuClientOptions, load_cpu_plugin};

    use ryft_core::compilation::CompilationOptions;

    use crate::experimental::domains::{XlaDomain, XlaDomainError, XlaOptions};
    use crate::tests::{values_from_bytes, values_to_bytes};
    use crate::{Array, CompiledFunction, FromPjrt, compile_and_execute, compile_and_execute_with_options, eval_shape};

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
        let engine = XlaDomain::new(&client);

        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(4)]),
            None,
            Some(Sharding::replicated(mesh.logical_mesh().clone(), 1)),
        )
        .unwrap();
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> =
            compile_and_execute(|x| x.sin(), input_type.clone(), &engine, mesh.clone()).unwrap();

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

    /// Inner-composition smoke test: a compiled function can be staged into another
    /// `compile_and_execute` closure as a sub-routine, producing the same result as if the
    /// whole computation were a single closure. Mirrors JAX's
    /// `jit(lambda x: jit(f)(x).cos())` pattern.
    ///
    /// This test exercises [`CompiledXlaFunction::call_traced`] — calling a compiled function
    /// inside an outer trace re-executes the retained closure against the outer tracers, which
    /// is how inner staging and outer transform composition both compose in the JAX-style
    /// pipeline.
    #[test]
    fn test_compiled_function_staged_inside_compile_and_execute() {
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
        let inner: CompiledFunction<'_, ArrayType, ArrayType> =
            compile_and_execute(|x| x.sin(), input_type.clone(), &engine, mesh.clone()).unwrap();

        // Outer: compile `g = |x| cos(inner(x))` by re-executing inner's retained closure into
        // the outer trace and applying `cos` to its output.
        let outer: CompiledFunction<'_, ArrayType, ArrayType> =
            compile_and_execute(move |x| inner.call_traced(x).cos(), input_type.clone(), &engine, mesh.clone())
                .unwrap();

        // Execute and compare against the inlined reference.
        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source = Array::from_host_buffer(
            &client,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();
        let output = outer.call(source).unwrap();

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

    /// Outer-transform smoke test: applying `grad` to a `CompiledFunction` (by re-tracing its
    /// retained closure under a `grad` trace) produces a new compiled function that computes
    /// `d/dx f(x)`. This mirrors JAX's `grad(jit(f))` idiom.
    ///
    /// The mechanism: the inner closure is re-executed inside a `grad`-mode trace via
    /// [`CompiledXlaFunction::call_traced`]. The outer trace context automatically routes each
    /// op through its `DifferentiableOperation::jvp` rule because that's the active trace mode.
    #[test]
    fn test_grad_of_compiled_function_round_trips() {
        use ryft_core::tracing_v2::DifferentiableDomain;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new()), None, Some(sharding)).unwrap();

        // Compile `f = |x| x.sin()`.
        let inner: CompiledFunction<'_, ArrayType, ArrayType> =
            compile_and_execute(|x| x.sin(), input_type.clone(), &engine, mesh.clone()).unwrap();

        // Compile `grad(f)` by re-tracing inner's retained closure inside a `grad` trace.
        let grad_compiled: CompiledFunction<'_, ArrayType, ArrayType> = compile_and_execute(
            move |x| {
                let inner = inner.clone();
                XlaDomain::token().grad(move |y| inner.call_traced(y), x).unwrap()
            },
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        // d/dx sin(x) = cos(x). Verify at a few points.
        for &point in &[0.0f32, 0.25, 0.5, 1.0] {
            let source = Array::from_host_buffer(
                &client,
                input_type.clone(),
                mesh.clone(),
                values_to_bytes::<f32>([point].as_slice()).as_slice(),
            )
            .unwrap();
            let output = grad_compiled.call(source).unwrap();
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

    #[test]
    fn test_jit_binary_function_with_tuple_input() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);

        let shape = Shape::new(vec![Size::Static(3)]);
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let input_type = ArrayType::new(DataType::F32, shape.clone(), None, Some(sharding.clone())).unwrap();
        let compiled: CompiledFunction<'_, (ArrayType, ArrayType), ArrayType> =
            compile_and_execute(|(a, b)| a + b, (input_type.clone(), input_type.clone()), &engine, mesh.clone())
                .unwrap();

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
        let engine = XlaDomain::new(&client);
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(4)]),
            None,
            Some(Sharding::replicated(mesh.logical_mesh().clone(), 1)),
        )
        .unwrap();

        assert_eq!(engine.cache_size(), 0);
        // Two `compile_and_execute` invocations on the same source line (inside the loop body) share a call-site
        // fingerprint, so the second invocation hits the cache instead of compiling again.
        for _ in 0..2 {
            let _: CompiledFunction<'_, ArrayType, ArrayType> =
                compile_and_execute(|x| x.sin(), input_type.clone(), &engine, mesh.clone()).unwrap();
        }
        assert_eq!(engine.cache_size(), 1, "repeat compile_and_execute at the same call site should hit the cache");
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
        // Two `compile_and_execute` invocations at distinct source lines populate two cache entries even when
        // the closure and inputs are identical, mirroring the way JAX's compile cache keys on
        // function identity (which differs per Python `id()` even for source-equivalent
        // lambdas).
        let _: CompiledFunction<'_, ArrayType, ArrayType> =
            compile_and_execute(|x| x.sin(), input_type.clone(), &engine, mesh.clone()).unwrap();
        let _: CompiledFunction<'_, ArrayType, ArrayType> =
            compile_and_execute(|x| x.sin(), input_type, &engine, mesh).unwrap();
        assert_eq!(engine.cache_size(), 2);
    }

    #[test]
    fn test_compile_and_execute_with_options_donates_argument() {
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
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> =
            compile_and_execute_with_options(|x| x.sin(), input_type.clone(), &engine, options).unwrap();

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
    fn test_compile_and_execute_with_options_rejects_donation_arity_mismatch() {
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
        let result: Result<CompiledFunction<'_, ArrayType, ArrayType>, XlaDomainError> =
            compile_and_execute_with_options(|x| x.sin(), input_type, &engine, options);
        assert!(matches!(result, Err(XlaDomainError::InvalidCompilationOptions { .. })));
    }

    /// Two compiles at the same source line with the SAME array leaf but DIFFERENT
    /// non-`Parameter` field on the input tree (`batch_size: usize`) must each populate their
    /// own cache entry. The cache key folds in a hash of `Input::ParameterStructure`, so any
    /// treedef-only difference (hyperparameters, mode flags, ...) partitions automatically.
    #[test]
    fn test_compile_and_execute_partitions_cache_by_input_tree_structure() {
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
            let _: CompiledFunction<'_, HyperparamInput<ArrayType>, ArrayType> =
                compile_and_execute(|input| input.array.sin(), input, &engine, mesh.clone()).unwrap();
        }
        assert_eq!(engine.cache_size(), 2);
    }

    #[test]
    fn test_compile_and_execute_with_options_in_shardings_override_replaces_input_sharding() {
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
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> =
            compile_and_execute_with_options(|x| x.sin(), abstract_input_type, &engine, options).unwrap();

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
    fn test_compile_and_execute_with_options_rejects_in_shardings_arity_mismatch() {
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
        let result: Result<CompiledFunction<'_, ArrayType, ArrayType>, XlaDomainError> =
            compile_and_execute_with_options(|x| x.sin(), input_type, &engine, options);
        assert!(matches!(result, Err(XlaDomainError::InvalidCompilationOptions { .. })));
    }

    #[test]
    fn test_compile_and_execute_with_options_out_shardings_override_propagates_to_output_array() {
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
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> =
            compile_and_execute_with_options(|x| x.sin(), input_type.clone(), &engine, options).unwrap();

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
        let engine = XlaDomain::new(&client);

        // The executable expects a 2-way shard along "x", but the caller will pass a fully
        // replicated array. `CompiledFunction::call` should silently reshard before executing.
        let shape = Shape::new(vec![Size::Static(4)]);
        let sharded =
            Sharding::new(mesh.logical_mesh().clone(), vec![ryft_core::sharding::ShardingDimension::sharded(["x"])])
                .unwrap();
        let input_type = ArrayType::new(DataType::F32, shape.clone(), None, Some(sharded.clone())).unwrap();
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> =
            compile_and_execute(|x| x.sin(), input_type.clone(), &engine, mesh.clone()).unwrap();

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
    fn test_compile_and_execute_with_options_rejects_out_shardings_arity_mismatch() {
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
        let result: Result<CompiledFunction<'_, ArrayType, ArrayType>, XlaDomainError> =
            compile_and_execute_with_options(|x| x.sin(), input_type, &engine, options);
        assert!(matches!(result, Err(XlaDomainError::InvalidCompilationOptions { .. })));
    }

    #[test]
    fn test_eval_shape_returns_output_types_without_compiling() {
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

        let output_type: ArrayType = eval_shape(|x| x.sin(), input_type.clone()).unwrap();
        assert_eq!(output_type.data_type(), DataType::F32);
        assert_eq!(output_type.shape(), input_type.shape());
        // `eval_shape` must not have populated the compile cache.
        assert_eq!(engine.cache_size(), 0);
    }

    #[test]
    fn test_jit_with_grad_inside_closure_compiles_and_runs() {
        use ryft_core::tracing_v2::DifferentiableDomain;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new()), None, Some(sharding.clone())).unwrap();

        // `compile_and_execute` composes with `grad` by invoking the transform *inside* the staged closure. The
        // tracing system records `grad`'s lowering into the same MLIR module that `compile_and_execute`
        // compiles, so the resulting executable computes `d/dx sin(x) = cos(x)` directly.
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> = compile_and_execute(
            |x: crate::experimental::shard_map::ShardMapTracer| {
                XlaDomain::token().grad(|y: crate::experimental::shard_map::ShardMapTracer| y.sin(), x).unwrap()
            },
            input_type.clone(),
            &engine,
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

    /// Verifies that the staged `to` operation works inside a `compile_and_execute`-compiled function: the
    /// sharding constraint is preserved through the trace, and the output array carries the
    /// constrained sharding on each device.
    #[test]
    fn test_jit_with_staged_to_constrains_output_sharding() {
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

        // The user invokes `to` directly inside the staged closure — it's compiled into the same
        // MLIR program as the rest of the function body.
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> = compile_and_execute(
            move |x: crate::experimental::shard_map::ShardMapTracer| {
                let constrained =
                    crate::experimental::shard_map::to(x, target_sharding.clone()).expect("staged to should succeed");
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

    /// Multiple staged `to` calls inside one `compile_and_execute` body compile into a single MLIR program with
    /// chained `sdy.sharding_constraint` ops — exactly one cache entry, exactly one PJRT execute
    /// per call. This is the async-pipelined regime: PJRT runs the whole compiled program in
    /// one shot without per-reshard host sync.
    #[test]
    fn test_jit_with_chained_staged_to_calls_compiles_to_one_program() {
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

        // Three staged `to` calls compose inside one closure. Each emits a
        // `sdy.sharding_constraint` op into the same MLIR program. After trace+compile, the
        // executable runs all three in one PJRT dispatch.
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> = compile_and_execute(
            move |x: crate::experimental::shard_map::ShardMapTracer| {
                let a = crate::experimental::shard_map::to(x, constraint_a.clone()).unwrap();
                let b = crate::experimental::shard_map::to(a.sin(), constraint_b.clone()).unwrap();
                crate::experimental::shard_map::to(b.sin(), constraint_c.clone()).unwrap()
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
    /// `jax.grad(jax.compile_and_execute(... with_sharding_constraint ...))` behavior.
    #[test]
    fn test_jit_with_grad_through_staged_to_runs() {
        use ryft_core::tracing_v2::DifferentiableDomain;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = single_device_mesh(&client);
        let engine = XlaDomain::new(&client);
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new()), None, Some(sharding.clone())).unwrap();

        // d/dx sin(to(x, S)) = cos(x), because `to`/`with_sharding_constraint` is the identity at
        // the value level — its linear transpose is the identity, so the gradient passes through.
        let target_sharding = sharding.clone();
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> = compile_and_execute(
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
            &engine,
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
