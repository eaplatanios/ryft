//! Eager operation dispatch and session-local specialization.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::sync::Arc;

use ryft_core::macros::check_count;
use ryft_core::{
    ArrayIrType, ArrayIrValue, ArrayType, AttachedRegions, COMPOSITE_OPERATION_NAME, CastType, Compare, Concretizable,
    Constant as CoreConstant, ConstantOperation, Context, DataType, Device, DeviceId, DeviceMesh, DimensionFromScalar,
    DimensionOperation, DimensionSize, DimensionType, DimensionValue, EagerContext, EffectClass, EmptyRegionDriver,
    InterpretableOperation, LogicalMesh, MeshAxis, MeshAxisType, ONE_OPERATION_NAME, Operation as CoreOperation,
    OperationProvider, Placeholder, ProgramError, Provenance, ProvenanceScope, RegionDriver, RegionKey, RegionRef,
    Shape, SpecializationCacheEntry, Type, TypeError, Typed, TypedProgram, ZERO_OPERATION_NAME, Zero, ZeroOperation,
};
use ryft_pjrt::Client;
use ryft_pjrt::protos::CompilationOptions;

use crate::domains::Domain;
use crate::domains::effects::ensure_effect_dispatch_allowed;
use crate::domains::options::Options;
use crate::domains::programs::CompiledProgram;
use crate::domains::target::Target;
use crate::experimental::ops::{Constant, FlatProgram, Operation, ProgramBuilder};
use crate::operations::composites::NamedCompositeImplementation;
use crate::{Array, ArrayError, Dimension, Error, FromPjrt, ShardDescriptor, ShardLayout, ToPjrt, Value};

impl<'c> Context for Domain<'c> {
    /// Immediate dimension extents materialize directly; Boolean literals upload a replicated scalar. A
    /// [`Constant::Captured`] payload is a symbolic index into a compiled function's capture table carrying only a
    /// type and no data, so there is nothing to materialize without the surrounding capture table and lifting it is
    /// always rejected.
    fn lift(&self, constant: Constant) -> Result<Value<'c>, ProgramError> {
        match constant {
            Constant::Captured(constant) => Err(TypeError::invalid(format!(
                "xla captured constant {constant} requires a captured program capture table"
            ))
            .into()),
            Constant::Dimension(value) => Ok(Value::Dimension(Dimension::new(value, self.clone()))),
            Constant::Boolean(value) => Ok(Value::Array(self.eager_scalar(DataType::Boolean, &[u8::from(value)])?)),
        }
    }

    /// Eagerly executes `operation` on concrete input [`Array`]s, mirroring JAX's op-by-op dispatch: a repeated
    /// application (including one whose attached regions are structurally equal to those of an earlier application) is
    /// served from the session's eager dispatch cache, and otherwise the operation is traced into a
    /// single-instruction program over the inputs' physical [`ArrayType`]s (shardings included), compiled through this
    /// domain's compile cache, and executed on this domain's PJRT client via the crate-private `eager_bind` path. The
    /// nullary additive/multiplicative identities keep a fast path that materializes the constant directly through the
    /// runtime client without compiling a program.
    fn bind<'r, P, R: Into<AttachedRegions<'r, Self::Constant, Self::Operation>>>(
        &self,
        operation: P,
        regions: R,
        inputs: &[Self::Value],
    ) -> Result<Vec<Self::Value>, ProgramError>
    where
        P: Into<Self::Operation>,
        Self::Constant: 'r,
        Self::Operation: 'r,
    {
        ensure_effect_dispatch_allowed()
            .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
        let operation = operation.into();
        let regions = regions.into();
        operation.validate_region_interfaces(&regions.region_interfaces())?;
        let name = operation.name();
        if inputs.is_empty() && (name == ZERO_OPERATION_NAME || name == ONE_OPERATION_NAME) {
            let array_type = eager_identity_output_type(&operation)?;
            validate_identity_synthesis(name, &array_type)?;
            // The direct constant-materialization fast path needs a concrete device mesh; mesh-less domains fall
            // through to the compiled eager path below, which derives a default execution mesh instead.
            if self.mesh.is_some() {
                let kind = if name == ZERO_OPERATION_NAME { ConstantKind::Zero } else { ConstantKind::One };
                let value = self.constant(&array_type, kind).map_err(|error| TypeError::invalid(error.to_string()))?;
                return Ok(vec![Value::Array(value)]);
            }
        }
        self.eager_bind(operation, regions, inputs)
    }

    /// Every [`Domain`] is client-backed, so it executes every bound operation for real and its concrete
    /// [`Array`]s support host readback through [`Concretizable<bool>`](ryft_core::Concretizable) and
    /// [`WhilePredicate`](ryft_core::operations::control_flow::WhilePredicate). Strategies that fold data-dependent
    /// work through host-visible values (the eager data-dependent `while` rules and concretizable-`while` unrolling)
    /// therefore always apply.
    fn is_eager(&self) -> bool {
        true
    }

    // `Domain` is a terminal execution domain: it dispatches operations to compiled per-operation programs
    // instead of recording instructions, so there is nothing to attach provenance to and scopes are a deliberate
    // no-op. Programs traced *over* this domain attach provenance through their own tracing contexts.
    #[inline]
    fn provenance(&self) -> Provenance {
        Provenance::unknown()
    }

    #[inline]
    fn invoke_with_provenance_origin<R, F: FnOnce() -> R>(&self, _origin: Provenance, function: F) -> R {
        function()
    }

    #[inline]
    fn invoke_with_provenance_scope<R, F: FnOnce() -> R>(&self, _scope: ProvenanceScope, function: F) -> R {
        function()
    }
}

// Context capability that materializes additive-identity [`Array`]s. Transform machinery over this domain (e.g.,
// the batching rules of nullary constant operations and the accumulator seeding of recursive higher-order rules)
// synthesizes constants through the active context's type-driven [`Zero`], [`One`](ryft_core::One),
// [`Fill`](ryft_core::Fill), and [`Iota`](ryft_core::Iota) leaves.
// The binds below take the constant-materialization fast path on domains constructed with a concrete mesh and the
// compiled eager dispatch path (over a derived default mesh) otherwise. Dynamic array zeros are intentionally not
// available through this type-only capability because they require explicit first-class extent operands.
impl<'c> Zero<Value<'c>> for Domain<'c> {
    fn zero(&self, r#type: &ArrayIrType) -> Result<Value<'c>, ProgramError> {
        let mut outputs = self.bind(Operation::provide(ZeroOperation::new(r#type.clone()), &[])?, Vec::new(), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

// Materialization delegates to [`Context::lift`] and therefore shares its semantics: an immediate
// dimension extent materializes directly and a Boolean literal uploads a replicated scalar, while a
// [`Constant::Captured`] payload carries only a type and no data and is always rejected outside a surrounding
// capture table. The implementation exists because interpretation- and batching-capable operation families require a
// [`CoreConstant`](ryft_core::Constant) leaf on their contexts; programs whose constants were compiled into capture tables never take the
// captured path.
impl<'c> CoreConstant<Value<'c>, Constant> for Domain<'c> {
    fn constant(&self, value: Constant) -> Result<Value<'c>, ProgramError> {
        self.lift(value)
    }
}

/// Returns the single [`ArrayType`] produced by a nullary additive/multiplicative identity operation
/// ([`ZERO_OPERATION_NAME`] / [`ONE_OPERATION_NAME`]). The identity fast path in [`Context::bind`] materializes these
/// constants directly through the runtime client instead of compiling a program.
fn eager_identity_output_type<O: CoreOperation<Type = ArrayIrType>>(operation: &O) -> Result<ArrayType, ProgramError> {
    let mut output_types = operation.infer_output_types(&[], &[])?;
    if output_types.len() != 1 {
        return Err(TypeError::invalid(format!(
            "xla identity operation `{}` must produce exactly one output but produced {}",
            operation.name(),
            output_types.len(),
        ))
        .into());
    }
    let output_type = output_types.pop().expect("output count checked above");
    <&ArrayType>::try_from(&output_type).cloned().map_err(Into::into)
}

fn validate_identity_synthesis(identity: &'static str, array_type: &ArrayType) -> Result<(), ProgramError> {
    match array_type.data_type() {
        DataType::Token | DataType::Zero if identity == ONE_OPERATION_NAME => Err(TypeError::invalid(format!(
            "xla domain cannot synthesize {identity} value for element type {}",
            array_type.data_type()
        ))
        .into()),
        DataType::Token => Err(TypeError::invalid(format!(
            "xla domain cannot synthesize {identity} value for element type {}",
            array_type.data_type()
        ))
        .into()),
        _ => Ok(()),
    }
}

impl<'c> Domain<'c> {
    /// Executes `operation` eagerly on host-side reference handles if it is a reference operation, and returns
    /// [`None`] otherwise. Reference operations act on shared handles rather than on device computations, so they
    /// execute through the generic [`ArrayIrValue`] reference implementations in `ryft-core` (which own handle
    /// validation and locking), and new references are bound to this domain.
    fn eager_reference_bind(
        &self,
        operation: &Operation,
        inputs: &[Value<'c>],
    ) -> Option<Result<Vec<Value<'c>>, ProgramError>> {
        let context = EagerContext::<ArrayIrValue<Array<'c>>>::new();
        let inputs = || inputs.iter().cloned().map(Value::into_array_ir_value).collect::<Vec<_>>();
        let outputs = match operation {
            Operation::ReferenceNew(operation) => operation.interpret(&context, &EmptyRegionDriver, &inputs()),
            Operation::ReferenceRead(operation) => operation.interpret(&context, &EmptyRegionDriver, &inputs()),
            Operation::ReferenceWrite(operation) => operation.interpret(&context, &EmptyRegionDriver, &inputs()),
            Operation::ReferenceSwap(operation) => operation.interpret(&context, &EmptyRegionDriver, &inputs()),
            Operation::ReferenceAddUpdate(operation) => operation.interpret(&context, &EmptyRegionDriver, &inputs()),
            Operation::ReferenceAtomicAddUpdate(operation) => {
                operation.interpret(&context, &EmptyRegionDriver, &inputs())
            }
            Operation::ReferenceFreeze(operation) => operation.interpret(&context, &EmptyRegionDriver, &inputs()),
            _ => return None,
        };
        Some(outputs.and_then(|outputs| {
            outputs
                .into_iter()
                .map(|output| Value::from_array_ir_value(output, self).map_err(ProgramError::from))
                .collect()
        }))
    }

    /// Uploads a replicated scalar with the provided host `bytes` and `data_type`, placed through [`Self::eager_mesh`].
    fn eager_scalar(&self, data_type: DataType, bytes: &[u8]) -> Result<Array<'c>, ProgramError> {
        let output_type = ArrayType::scalar(data_type);
        let mesh = self.eager_mesh(&[], std::slice::from_ref(&output_type))?;
        let output_type = output_type
            .replicated(&mesh)
            .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
        Array::from_host_buffer(self, output_type, mesh, bytes)
            .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })
    }

    /// Eagerly executes one staged operation on concrete input [`Array`]s — the JAX-style op-by-op dispatch path
    /// behind [`Context::bind`].
    ///
    /// An application is first looked up in the session's eager dispatch cache, keyed by the operation, the structure
    /// of its attached regions, its inputs, the lowering mesh, and the compilation options (refer to
    /// [`EagerDispatchKey`]). A hit validates the input placement and executes the cached program directly, skipping
    /// tracing, lowering, and compilation-key construction. On a miss, the operation is traced into a
    /// single-instruction flat program over the inputs' physical [`ArrayType`]s (shardings included), compiled through
    /// this domain's compile cache, and executed on this domain's PJRT client. The compile cache key contains the
    /// complete lowered computation, its effective compilation options, and the derived mesh, so repeated eager binds
    /// of the same lowering reuse one compiled executable without conflating distinct computations. Higher-order
    /// operations (`condition` / `while` / `scan` / `jit_call` / `shard_map`) receive their nested programs as attached
    /// regions and flow through this same path, and the compiler handles their control flow, so no host interpreter
    /// loops are needed.
    ///
    /// Work that involves no device computation never reaches the compile cache. Reference operations act on host-side
    /// handles (refer to [`Self::eager_reference_bind`]), while dimension operations, dimension casts and comparisons,
    /// and conversions between dimensions and scalars are decided on the host. Only their array results are uploaded
    /// (refer to [`Self::eager_scalar`]). Reference operands of any other operation, and reference state carried by
    /// attached regions, are rejected because XLA eager execution lowers each operation.
    fn eager_bind(
        &self,
        operation: Operation,
        regions: AttachedRegions<'_, Constant, Operation>,
        inputs: &[Value<'c>],
    ) -> Result<Vec<Value<'c>>, ProgramError> {
        if let Some(outputs) = self.eager_reference_bind(&operation, inputs) {
            return outputs;
        }
        if inputs.iter().any(|input| matches!(input, Value::Reference(_))) {
            return Err(ProgramError::UnsupportedOperation {
                message: "XLA eager execution lowers each operation and cannot bind a reference operand; discharge \
                          references before lowering"
                    .to_string(),
            });
        }
        // The intrinsic operation effects alone would miss region-carried state, while effect scans alone would miss
        // a pure reference-typed pass-through or capture. The borrowed closure scan covers dormant rule regions too
        // without cloning their program graphs. Although those rules do not execute in this eager bind, ordinary XLA
        // rejects unresolved references artifact-wide, so accepting one here would only defer failure to compilation
        // with a less targeted diagnostic.
        if operation.effects().classes().contains(EffectClass::OrderedState)
            || regions.regions().any(|region| {
                region.contains_effect_in_closure(EffectClass::OrderedState)
                    || region.contains_atom_type_in_closure(Type::is_reference)
            })
        {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "`{}` carries reference state that XLA eager execution cannot lower; discharge references before \
                     lowering",
                    operation.name(),
                ),
            });
        }
        if let Operation::Dimension(operation) = &operation {
            if regions.regions().count() != 0 {
                return Err(TypeError::invalid("dimension operations do not accept attached regions").into());
            }
            let inputs = inputs
                .iter()
                .map(|input| {
                    <Value<'c> as ryft_core::ValueProjection<DimensionType>>::projected(input)
                        .map(|dimension| dimension.value().clone())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let outputs = EagerContext::<DimensionValue, DimensionOperation<DimensionValue>>::new().bind(
                operation.clone(),
                Vec::new(),
                inputs.as_slice(),
            )?;
            return Ok(outputs
                .into_iter()
                .map(|output| Value::Dimension(Dimension::new(output, self.clone())))
                .collect());
        }
        if let Operation::DimensionSize(operation) = &operation {
            if regions.regions().count() != 0 {
                return Err(TypeError::invalid("dimension_size does not accept attached regions").into());
            }
            check_count!("input", inputs, 1, ProgramError);
            operation.infer_output_types(&[inputs[0].r#type().into_owned()], &[])?;
            let array = <Value<'c> as ryft_core::ValueProjection<ArrayType>>::projected(&inputs[0])?;
            let extent = array.dimension_size(operation.axis())?;
            // Arrays derived from this dimension are placed on the mesh of the array that it was read from.
            let value = DimensionValue::new(operation.output_type().clone(), extent)?;
            return Ok(vec![Value::Dimension(Dimension::new(value, self.with_mesh(array.mesh().clone())))]);
        }
        if let Operation::DimensionFromScalar(operation) = &operation {
            if regions.regions().count() != 0 {
                return Err(TypeError::invalid("dimension_from_scalar does not accept attached regions").into());
            }
            check_count!("input", inputs, 1, ProgramError);
            operation.infer_output_types(&[inputs[0].r#type().into_owned()], &[])?;
            let array = <Value<'c> as ryft_core::ValueProjection<ArrayType>>::projected(&inputs[0])?;
            // Arrays derived from this dimension are placed on the mesh of the array that it was decoded from.
            let value = array.to_dimension(operation.output_type().to_variable())?;
            return Ok(vec![Value::Dimension(Dimension::new(value, self.with_mesh(array.mesh().clone())))]);
        }

        if matches!(&operation, Operation::DimensionToScalar(_)) {
            if regions.regions().count() != 0 {
                return Err(TypeError::invalid("dimension_to_scalar does not accept attached regions").into());
            }
            check_count!("input", inputs, 1, ProgramError);
            let dimension = <Value<'c> as ryft_core::ValueProjection<DimensionType>>::projected(&inputs[0])?;
            let extent = i64::try_from(dimension.value().extent()).unwrap();
            return Ok(vec![Value::Array(self.eager_scalar(DataType::I64, extent.to_ne_bytes().as_slice())?)]);
        }
        if let Operation::Compare(operation) = &operation
            && let [Value::Dimension(left), Value::Dimension(right)] = inputs
        {
            if regions.regions().count() != 0 {
                return Err(TypeError::invalid("compare does not accept attached regions").into());
            }
            // Both extents are concrete, so the predicate is decided on the host by the comparison of first-class
            // dimensions (which proves it from their types and extents), instead of compiling a program per extent
            // pair. Only the Boolean result is materialized in this domain.
            let output =
                Compare::<ArrayIrValue<ryft_core::Array>>::compare(left.value(), right.value(), operation.direction())?;
            let output = Concretizable::<bool>::concretize(&output)?;
            return Ok(vec![Value::Array(self.eager_scalar(DataType::Boolean, &[u8::from(output)])?)]);
        }

        // A composite call whose concrete inputs refine the types that its primal region was traced over (e.g., a named
        // call staged in a bounded-dynamic program) replays its primal region, exactly as the interpretation rule of
        // composites does, because that region describes the call at its traced types rather than at the exact types
        // of its inputs. Calls at their traced types compile as one program, so backend implementations of named
        // composites (e.g., fused kernels) still apply to them. A named call that explicitly requests a backend
        // implementation reports that it cannot use it instead of silently replaying its body.
        if operation.name() == COMPOSITE_OPERATION_NAME
            && let Some(primal) = regions.regions().next()
            && primal
                .interface()
                .input_types()
                .iter()
                .zip(inputs)
                .any(|(declared, input)| *declared != *input.r#type())
        {
            if let Some(key) = crate::LowerableOperation::composite_key(&operation) {
                let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
                let target_platform = self.target().ok().map(Target::platform_name);
                let implementation = NamedCompositeImplementation::select(
                    key,
                    input_types.as_slice(),
                    primal.interface().output_types(),
                    target_platform,
                );
                if let Some(error) = implementation.and_then(|implementation| implementation.body_substitution_error())
                {
                    return Err(ProgramError::UnsupportedOperation { message: error.to_string() });
                }
            }
            return primal.interpret_in_context(self, inputs.to_vec());
        }

        let array_inputs = inputs
            .iter()
            .filter_map(|input| match input {
                Value::Array(array) => Some(array.clone()),
                Value::Dimension(_) | Value::Reference(_) => None,
            })
            .collect::<Vec<_>>();

        if let Operation::Cast(operation) = &operation
            && matches!(operation.output_type(), ArrayIrType::Dimension(_))
        {
            // Context::bind rejects attached regions. Validate the original declaration before observing every
            // array witness producer: the host dimension has no buffer that could retain those dependencies. Eager
            // values have exact types, so interpreted inference proves the relationship or rejects it; it never
            // leaves a check to a backend.
            let values = inputs.iter().cloned().map(Value::into_array_ir_value).collect::<Vec<_>>();
            let input_types = values.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
            let (_, interpreted) = ArrayIrType::infer_cast_output_type(operation, &input_types)?;
            if interpreted.is_none() {
                return Err(ProgramError::Concretization {
                    message: format!(
                        "`{}` cannot check symbolic type relationships during interpretation; \
                         bind the operation through its context",
                        operation.name(),
                    ),
                });
            }
            self.validate_eager_placement(&array_inputs)?;
            for input in &array_inputs {
                input
                    .block_until_ready()
                    .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
            }
            return Ok(vec![Value::from_array_ir_value(values[0].clone(), self)?]);
        }

        // Applications go through the session's eager dispatch cache, which maps the operation, its attached regions,
        // and its inputs directly to a compiled program, so a repeated application skips tracing, lowering, and
        // compilation-key construction and only revalidates its input placement. Hits are looked up through a
        // borrowed query, and the owned key is only built on a miss.
        let region_keys = regions.regions().map(RegionRef::key).collect::<Vec<_>>();
        let query = EagerDispatchQuery {
            operation: &operation,
            regions: region_keys.as_slice(),
            inputs,
            mesh: self.mesh.as_ref().or_else(|| array_inputs.first().map(Array::mesh)),
            compilation_options: &self.compilation_options,
        };
        let compiled = match self.session.eager_dispatch_cache.get(&query as &dyn EagerDispatchKeyView) {
            Some(compiled) => {
                self.validate_eager_placement(array_inputs.as_slice())?;
                compiled
            }
            None => {
                // The query borrows `region_keys`, so the key takes them only after its last use of the query.
                let key = EagerDispatchKey {
                    operation: operation.clone(),
                    inputs: (0..inputs.len()).map(|index| query.input(index).into_owned()).collect(),
                    mesh: query.mesh.cloned(),
                    compilation_options: Arc::clone(&self.compilation_options),
                    regions: region_keys.into(),
                };
                let entry = self
                    .session
                    .eager_dispatch_cache
                    .try_entry(key)
                    .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
                match entry {
                    SpecializationCacheEntry::Occupied(compiled) => {
                        self.validate_eager_placement(array_inputs.as_slice())?;
                        compiled
                    }
                    SpecializationCacheEntry::Vacant(producer) => producer.insert(self.compile_eager_operation(
                        operation,
                        regions,
                        inputs,
                        array_inputs.as_slice(),
                    )?),
                }
            }
        };

        // Execution produces outputs that belong to this domain, so chained eager operations and transforms over them
        // keep executing in the same session and effect scope.
        let outputs = self
            .execute_program(&compiled, array_inputs)
            .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
        Ok(outputs.into_iter().map(Value::Array).collect())
    }

    /// Traces one eager operation application into a single-instruction program over the inputs' physical types
    /// (shardings included), validates the placement of its array inputs, lowers the program, and compiles it through
    /// this domain's compilation cache (so that equal lowerings share one compiled program).
    fn compile_eager_operation(
        &self,
        operation: Operation,
        regions: AttachedRegions<'_, Constant, Operation>,
        inputs: &[Value<'c>],
        array_inputs: &[Array<'c>],
    ) -> Result<Arc<CompiledProgram<'c>>, ProgramError> {
        // Trace the single-instruction program, attaching the provided region bodies to that instruction.
        let region_input_types = vec![None; regions.region_count()];
        let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        let region_ids = regions.import_into(&builder, &region_input_types)?;
        let output_atoms = {
            let mut builder = builder.borrow_mut();
            // Eager dimension values have static types, which define no identities, so equal dimension inputs share one
            // constant instruction.
            let mut dimension_atoms = HashMap::new();
            let input_atoms = inputs
                .iter()
                .map(|input| match input {
                    Value::Array(array) => Ok(builder.add_input(ArrayIrType::Array(array.r#type().into_owned()))),
                    Value::Dimension(dimension) => {
                        let dimension = dimension.value();
                        if let Some(&atom) = dimension_atoms.get(dimension) {
                            return Ok(atom);
                        }
                        let operation = DimensionOperation::Constant(ConstantOperation::new(dimension.clone()));
                        let atom =
                            builder.add_instruction(Operation::Dimension(operation), Vec::new(), Vec::new(), None)?[0];
                        dimension_atoms.insert(dimension, atom);
                        Ok(atom)
                    }
                    Value::Reference(_) => {
                        unreachable!("reference inputs are rejected at the `eager_bind` entry guard")
                    }
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            builder.add_instruction(operation, region_ids, input_atoms, None)?.to_vec()
        };
        self.validate_eager_placement(array_inputs)?;
        let output_count = output_atoms.len();
        let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
        let program: FlatProgram =
            builder.build(output_atoms, vec![Placeholder; array_inputs.len()], vec![Placeholder; output_count])?;

        // Derive the mesh after tracing so that input-free operations can fall back to their inferred output
        // shardings, and then compile through the domain's compilation cache.
        let output_types = program
            .output_types()
            .iter()
            .map(<&ArrayType>::try_from)
            .map(|result| result.cloned())
            .collect::<Result<Vec<_>, _>>()?;
        let mesh = self.eager_mesh(array_inputs, output_types.as_slice())?;
        let options = Options::new(mesh);
        let lowered = self
            .lower_program(&program, 0, &options)
            .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
        self.compilation_context()
            .compile(self, &lowered)
            .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })
    }

    /// Validates that every input lives on this domain's PJRT client and that all inputs share one device placement,
    /// mirroring JAX's "received incompatible devices for jitted computation" error for the eager path. Every
    /// [`Array`] belongs to a client-backed domain whose client owns its addressable shard buffers (validated at
    /// construction), so client identity is exact and also rejects same-device-id arrays owned by a *different* client.
    fn validate_eager_placement(&self, inputs: &[Array<'c>]) -> Result<(), ProgramError> {
        if inputs.is_empty() {
            return Ok(());
        }
        let invalid_argument = |message: String| ProgramError::InvalidArgument { message };
        for (index, input) in inputs.iter().enumerate() {
            if !std::ptr::eq(input.client(), self.client()) {
                return Err(invalid_argument(format!(
                    "received incompatible devices for eager xla execution: input #{index} is owned by a different \
                     PJRT client than this domain's client",
                )));
            }
        }
        // Arrays produced by the same program (or cloned from one array) share their mesh allocation, so pointer
        // equality settles the common case without comparing devices.
        let first_mesh = inputs[0].mesh();
        let device_ids = |mesh: &DeviceMesh| mesh.devices().iter().map(Device::id).collect::<Vec<_>>();
        for (index, input) in inputs.iter().enumerate().skip(1) {
            let mesh = input.mesh();
            if !std::ptr::eq(mesh, first_mesh)
                && !mesh.devices().iter().map(Device::id).eq(first_mesh.devices().iter().map(Device::id))
            {
                return Err(invalid_argument(format!(
                    "received incompatible devices for eager xla execution: input #{index} is placed on devices {:?} \
                     but input #0 is placed on devices {:?}",
                    device_ids(mesh),
                    device_ids(first_mesh),
                )));
            }
        }
        Ok(())
    }

    /// Returns the concrete [`DeviceMesh`] eager execution compiles against: the domain's own mesh when one is
    /// attached, otherwise the mesh implied by the inputs' shard placement, otherwise the logical mesh of the first
    /// declared output [`Sharding`](ryft_core::Sharding) assembled over the client's addressable devices (for
    /// input-free operations with sharded outputs, such as the nullary sharded constants that transform machinery
    /// synthesizes through [`Zero`] / [`One`] / [`Fill`] / [`Iota`] on mesh-less recovered domains), otherwise a
    /// single-device mesh over the client's first addressable device (for input-free operations over unsharded data).
    fn eager_mesh(&self, inputs: &[Array<'c>], output_types: &[ArrayType]) -> Result<DeviceMesh, ProgramError> {
        if let Some(mesh) = self.mesh.as_ref() {
            return Ok(mesh.clone());
        }
        if let Some(input) = inputs.first() {
            return Ok(input.mesh().clone());
        }
        let invalid_argument = |message: String| ProgramError::InvalidArgument { message };
        let devices = self.client().addressable_devices().map_err(|error| invalid_argument(error.to_string()))?;
        if let Some(logical_mesh) =
            output_types.iter().find_map(|r#type| r#type.sharding().map(|sharding| sharding.mesh().clone()))
        {
            // An input-free operation with sharded outputs has no concrete placement to inherit, so materialize its
            // declared logical mesh over the client's addressable devices in enumeration order.
            let device_count = logical_mesh.device_count();
            if devices.len() < device_count {
                return Err(invalid_argument(format!(
                    "eager xla execution of an input-free operation sharded over mesh {logical_mesh:?} requires \
                     {device_count} addressable device(s) but the client only has {}",
                    devices.len(),
                )));
            }
            let devices = devices
                .iter()
                .take(device_count)
                .map(|device| Device::from_pjrt(device).map_err(|error| invalid_argument(error.to_string())))
                .collect::<Result<Vec<_>, _>>()?;
            return DeviceMesh::new(logical_mesh, devices).map_err(|error| invalid_argument(error.to_string()));
        }
        let device = devices
            .first()
            .ok_or_else(|| invalid_argument("eager xla execution requires at least one addressable device".into()))?;
        let device = Device::from_pjrt(device).map_err(|error| invalid_argument(error.to_string()))?;
        let axis = MeshAxis::new("x", 1, MeshAxisType::Auto).map_err(|error| invalid_argument(error.to_string()))?;
        let logical_mesh = LogicalMesh::new(vec![axis]).map_err(|error| invalid_argument(error.to_string()))?;
        DeviceMesh::new(logical_mesh, vec![device]).map_err(|error| invalid_argument(error.to_string()))
    }

    /// Materializes a concrete [`Array`] whose addressable shards are filled with a constant.
    fn constant(&self, array_type: &ArrayType, kind: ConstantKind) -> Result<Array<'c>, Error> {
        let client = self.client();
        let mesh = self.mesh.as_ref().ok_or_else(|| Error::from(ArrayError::MissingMesh))?;
        static_dimensions_or_panic(array_type);
        let effective_type = match array_type.sharding() {
            Some(_) => array_type.clone(),
            None => array_type.replicated(mesh)?,
        };
        if array_type.data_type().is_zero() {
            return Ok(Array::from_zero_space(self, effective_type, mesh.clone())?);
        }
        let addressable_ids = addressable_device_ids(client, mesh)?;
        let element_size_in_bytes = array_type.data_type().to_pjrt().element_size_in_bytes()?;

        let mut addressable_buffers = Vec::with_capacity(addressable_ids.len());
        for shard in shards_for_type(&effective_type, mesh)? {
            let shard_device = shard.device();
            let shard_device_id = shard_device.id();
            if !addressable_ids.contains(&shard_device_id) {
                continue;
            }
            let shard_shape = shard.shape();
            let element_count = Shape::from(&shard_shape).element_count()?.expect("shard shapes are static");
            let bytes = constant_bytes(array_type.data_type(), kind, element_count, element_size_in_bytes);
            let dimensions = shard_shape.as_slice().iter().map(|&dimension| dimension as u64).collect::<Vec<_>>();
            let device = client
                .addressable_devices()?
                .into_iter()
                .find(|device| device.id().map(|id| id == shard_device_id).unwrap_or(false))
                .ok_or_else(|| {
                    Error::from(ArrayError::NonAddressableDevice {
                        device_id: shard_device_id,
                        process_index: shard_device.process_index(),
                    })
                })?;
            let buffer = client.buffer(
                bytes.as_slice(),
                Array::physical_buffer_type(array_type.data_type()),
                dimensions.as_slice(),
                None,
                device,
                None,
            )?;
            addressable_buffers.push(buffer);
        }

        // The constant belongs to this domain so that chained eager operations and transforms over it recover a
        // context that keeps the same session, effect scope, and compilation options.
        Ok(Array::from_canonical_addressable_buffers(self, effective_type, mesh.clone(), addressable_buffers)?)
    }
}

// ---------------------------------------------------------------------------
// Constant materialization
// ---------------------------------------------------------------------------

/// Kind of constant value materialized by [`Domain::constant`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ConstantKind {
    /// Additive identity.
    Zero,

    /// Multiplicative identity.
    One,
}

/// Returns the static dimensions encoded by `array_type`, panicking if any dimension is dynamic.
///
/// Tests use this helper when constructing static-only values and treat dynamic shapes as programmer error.
fn static_dimensions_or_panic(array_type: &ArrayType) -> Vec<usize> {
    array_type
        .static_shape()
        .unwrap_or_else(|| panic!("Domain requires static ArrayType shapes, but got {}", array_type.shape()))
        .dimensions()
        .to_vec()
}

/// Returns a dense row-major host buffer encoding `element_count` copies of `kind` for
/// `data_type`.
///
/// Booleans are encoded as one byte per element (`0` / `1`). Integers and floating-point numbers
/// are encoded in native-endian byte order matching
/// [`ryft_pjrt::Client::buffer`](ryft_pjrt::Client::buffer)'s expectations. Complex numbers are
/// encoded as a `(real, imaginary)` pair of native-endian floats.
fn constant_bytes(data_type: DataType, kind: ConstantKind, element_count: usize, element_size: usize) -> Vec<u8> {
    match kind {
        ConstantKind::Zero => vec![0u8; element_count * element_size],
        ConstantKind::One => {
            let pattern = one_pattern_bytes(data_type);
            debug_assert_eq!(pattern.len(), element_size);
            let mut bytes = Vec::with_capacity(element_count * element_size);
            for _ in 0..element_count {
                bytes.extend_from_slice(&pattern);
            }
            bytes
        }
    }
}

/// Returns the native-endian byte pattern for a single `1`-valued element of `data_type`.
fn one_pattern_bytes(data_type: DataType) -> Vec<u8> {
    match data_type {
        DataType::Boolean => vec![1u8],
        DataType::I8 => 1i8.to_ne_bytes().to_vec(),
        DataType::U8 => 1u8.to_ne_bytes().to_vec(),
        DataType::I16 => 1i16.to_ne_bytes().to_vec(),
        DataType::U16 => 1u16.to_ne_bytes().to_vec(),
        DataType::I32 => 1i32.to_ne_bytes().to_vec(),
        DataType::U32 => 1u32.to_ne_bytes().to_vec(),
        DataType::I64 => 1i64.to_ne_bytes().to_vec(),
        DataType::U64 => 1u64.to_ne_bytes().to_vec(),
        DataType::BF16 => half::bf16::ONE.to_bits().to_ne_bytes().to_vec(),
        DataType::F16 => half::f16::ONE.to_bits().to_ne_bytes().to_vec(),
        DataType::F32 => 1.0f32.to_ne_bytes().to_vec(),
        DataType::F64 => 1.0f64.to_ne_bytes().to_vec(),
        DataType::C64 => {
            let mut bytes = Vec::with_capacity(8);
            bytes.extend_from_slice(&1.0f32.to_ne_bytes());
            bytes.extend_from_slice(&0.0f32.to_ne_bytes());
            bytes
        }
        DataType::C128 => {
            let mut bytes = Vec::with_capacity(16);
            bytes.extend_from_slice(&1.0f64.to_ne_bytes());
            bytes.extend_from_slice(&0.0f64.to_ne_bytes());
            bytes
        }
        // Low-precision (4-, 6-, and 8-bit) floating-point types do not have a canonical Rust representation;
        // encoding `1.0` as a raw byte pattern would depend on the exact variant.
        DataType::F8E3M4
        | DataType::F8E4M3
        | DataType::F8E4M3FN
        | DataType::F8E4M3FNUZ
        | DataType::F8E4M3B11FNUZ
        | DataType::F8E5M2
        | DataType::F8E5M2FNUZ
        | DataType::F8E8M0FNU
        | DataType::Token
        | DataType::Zero
        | DataType::I1
        | DataType::I2
        | DataType::I4
        | DataType::U1
        | DataType::U2
        | DataType::U4
        | DataType::F4E2M1FN
        | DataType::F6E2M3FN
        | DataType::F6E3M2FN => {
            panic!("Domain::one does not support element type {data_type}")
        }
    }
}

/// Returns the addressable device IDs for `client`, filtered to devices that are both addressable by the client and
/// present in the mesh.
pub(super) fn addressable_device_ids(client: &Client<'_>, mesh: &DeviceMesh) -> Result<Vec<DeviceId>, Error> {
    let mut addressable = Vec::new();
    for device in client.addressable_devices()? {
        let device_id = device.id()?;
        if mesh.devices().iter().any(|device| device.id() == device_id) {
            addressable.push(device_id);
        }
    }
    Ok(addressable)
}

/// Returns the shard descriptors implied by `array_type` and `mesh`.
pub(super) fn shards_for_type(array_type: &ArrayType, mesh: &DeviceMesh) -> Result<Vec<ShardDescriptor>, Error> {
    let sharding = array_type.sharding().ok_or_else(|| Error::from(ArrayError::MissingSharding))?;
    let global_shape = array_type
        .static_shape()
        .ok_or_else(|| Error::from(ArrayError::DynamicShape { shape: array_type.shape().clone() }))?;
    let (descriptors, _) = ShardLayout::new(&global_shape, mesh, sharding)?.into_parts();
    Ok(descriptors)
}

/// Default capacity of the eager dispatch cache of a [`Session`](crate::Session).
pub(super) const EAGER_DISPATCH_CACHE_CAPACITY: usize = 4096;

/// Key of the eager dispatch cache of a [`Session`](crate::Session), which maps one eager operation application
/// directly to its compiled program so that a repeated application skips tracing, lowering, and compilation-key
/// construction, like JAX's per-primitive `(primitive, params)` cache in front of its dispatch path.
///
/// The key must capture everything that the eager lowering reads besides the session itself (whose client, and thus
/// [`Target`], is fixed): the operation including all of its attributes, the contents of every attached region
/// (for operations such as `condition`, `while`, `scan`, `jit_call`, or `shard_map`), the physical type of every array
/// input (shardings included), the extent of every dimension input (because eager binding bakes dimension inputs into
/// the traced program as constants), the device mesh that the lowering compiles against, and the domain's compilation
/// options. Attached regions are keyed by their contents (refer to [`RegionKey`]), so applications whose closures are
/// traced afresh on every call still hit the cache when they trace to the same regions. Instruction provenance is not
/// part of a [`RegionKey`], so applications that differ only in provenance share one compiled program, whose debug
/// locations are those of the first application.
///
/// Keys are compared and hashed through [`EagerDispatchKeyView`], which they share with the borrowed
/// [`EagerDispatchQuery`] that eager dispatch looks hits up with.
#[derive(Clone)]
pub(super) struct EagerDispatchKey {
    /// Applied operation, compared and hashed with all of its attributes.
    operation: Operation,

    /// Keys of the regions attached to the application, in operation-defined order.
    regions: Arc<[RegionKey<Constant, Operation>]>,

    /// Inputs of the application, in order.
    inputs: Arc<[EagerDispatchInput]>,

    /// Mesh that eager lowering compiles against when it is not derived from the operation alone: the domain mesh if
    /// it has one, and otherwise the mesh of the first array input. It is [`None`] only for an input-free operation of
    /// a domain without a mesh, whose mesh is then determined by the operation and the session's client.
    mesh: Option<DeviceMesh>,

    /// Compilation options of the dispatching domain, compared and hashed by the address of their shared allocation.
    /// Domains created through [`Session::domain`](crate::Session::domain) share one default allocation, so they
    /// share eager dispatch cache entries, while each [`Domain::with_compilation_options`] call creates a distinct
    /// identity. Options that are equal but separately allocated therefore only miss each other's entries.
    compilation_options: Arc<CompilationOptions>,
}

/// One input of an [`EagerDispatchKey`].
#[derive(Clone)]
enum EagerDispatchInput {
    /// Array input, keyed by its physical [`ArrayType`] (including its sharding).
    Array(ArrayType),

    /// Dimension input, keyed by its extent, which is how [`DimensionValue`]s compare.
    Dimension(DimensionValue),
}

/// Borrowed form of one [`EagerDispatchInput`], which owned keys and borrowed queries compare and hash alike.
#[derive(PartialEq, Eq, Hash)]
enum EagerDispatchInputRef<'a> {
    /// Array input, keyed by its physical [`ArrayType`] (including its sharding).
    Array(Cow<'a, ArrayType>),

    /// Dimension input, keyed by its extent, which is how [`DimensionValue`]s compare.
    Dimension(&'a DimensionValue),
}

impl EagerDispatchInputRef<'_> {
    /// Returns the owned [`EagerDispatchInput`] that this reference borrows from or describes.
    fn into_owned(self) -> EagerDispatchInput {
        match self {
            Self::Array(r#type) => EagerDispatchInput::Array(r#type.into_owned()),
            Self::Dimension(dimension) => EagerDispatchInput::Dimension(dimension.clone()),
        }
    }
}

/// Common view of an owned [`EagerDispatchKey`] and a borrowed [`EagerDispatchQuery`]. Both forms are compared
/// and hashed through `dyn EagerDispatchKeyView`, so that equal keys and queries hash alike and an owned key can be
/// [borrowed](std::borrow::Borrow) as a view for lookups in the eager dispatch cache.
trait EagerDispatchKeyView {
    /// Returns the applied operation.
    fn operation(&self) -> &Operation;

    /// Returns the keys of the regions attached to the application, in operation-defined order.
    fn regions(&self) -> &[RegionKey<Constant, Operation>];

    /// Returns the number of inputs of the application.
    fn input_count(&self) -> usize;

    /// Returns the input at `index`, which must be smaller than [`Self::input_count`].
    fn input(&self, index: usize) -> EagerDispatchInputRef<'_>;

    /// Returns the mesh that the eager lowering compiles against, if it is not derived from the operation alone.
    fn mesh(&self) -> Option<&DeviceMesh>;

    /// Returns the compilation options of the dispatching domain, which are compared by identity.
    fn compilation_options(&self) -> &Arc<CompilationOptions>;
}

impl EagerDispatchKeyView for EagerDispatchKey {
    fn operation(&self) -> &Operation {
        &self.operation
    }

    fn regions(&self) -> &[RegionKey<Constant, Operation>] {
        &self.regions
    }

    fn input_count(&self) -> usize {
        self.inputs.len()
    }

    fn input(&self, index: usize) -> EagerDispatchInputRef<'_> {
        match &self.inputs[index] {
            EagerDispatchInput::Array(r#type) => EagerDispatchInputRef::Array(Cow::Borrowed(r#type)),
            EagerDispatchInput::Dimension(dimension) => EagerDispatchInputRef::Dimension(dimension),
        }
    }

    fn mesh(&self) -> Option<&DeviceMesh> {
        self.mesh.as_ref()
    }

    fn compilation_options(&self) -> &Arc<CompilationOptions> {
        &self.compilation_options
    }
}

impl PartialEq for dyn EagerDispatchKeyView + '_ {
    fn eq(&self, other: &Self) -> bool {
        self.operation() == other.operation()
            && self.regions() == other.regions()
            && self.input_count() == other.input_count()
            && (0..self.input_count()).all(|index| self.input(index) == other.input(index))
            && self.mesh() == other.mesh()
            && Arc::ptr_eq(self.compilation_options(), other.compilation_options())
    }
}

impl Eq for dyn EagerDispatchKeyView + '_ {}

impl Hash for dyn EagerDispatchKeyView + '_ {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.operation().hash(state);
        self.regions().hash(state);
        self.input_count().hash(state);
        (0..self.input_count()).for_each(|index| self.input(index).hash(state));
        self.mesh().hash(state);
        Arc::as_ptr(self.compilation_options()).hash(state);
    }
}

impl PartialEq for EagerDispatchKey {
    fn eq(&self, other: &Self) -> bool {
        (self as &dyn EagerDispatchKeyView) == (other as &dyn EagerDispatchKeyView)
    }
}

impl Eq for EagerDispatchKey {}

impl Hash for EagerDispatchKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self as &dyn EagerDispatchKeyView).hash(state);
    }
}

impl<'a> std::borrow::Borrow<dyn EagerDispatchKeyView + 'a> for EagerDispatchKey {
    fn borrow(&self) -> &(dyn EagerDispatchKeyView + 'a) {
        self
    }
}

/// Borrowed [`EagerDispatchKey`] over the arguments of one eager application, used to look up cache hits without
/// building an owned key.
struct EagerDispatchQuery<'a, 'c> {
    /// Applied operation.
    operation: &'a Operation,

    /// Keys of the regions attached to the application, in operation-defined order.
    regions: &'a [RegionKey<Constant, Operation>],

    /// Inputs of the application, which contain no references.
    inputs: &'a [Value<'c>],

    /// Mesh that the eager lowering compiles against (refer to [`EagerDispatchKey`] for more information).
    mesh: Option<&'a DeviceMesh>,

    /// Compilation options of the dispatching domain.
    compilation_options: &'a Arc<CompilationOptions>,
}

impl EagerDispatchKeyView for EagerDispatchQuery<'_, '_> {
    fn operation(&self) -> &Operation {
        self.operation
    }

    fn regions(&self) -> &[RegionKey<Constant, Operation>] {
        self.regions
    }

    fn input_count(&self) -> usize {
        self.inputs.len()
    }

    fn input(&self, index: usize) -> EagerDispatchInputRef<'_> {
        match &self.inputs[index] {
            Value::Array(array) => EagerDispatchInputRef::Array(array.r#type()),
            Value::Dimension(dimension) => EagerDispatchInputRef::Dimension(dimension.value()),
            Value::Reference(_) => unreachable!("reference inputs are rejected at the `eager_bind` entry guard"),
        }
    }

    fn mesh(&self) -> Option<&DeviceMesh> {
        self.mesh
    }

    fn compilation_options(&self) -> &Arc<CompilationOptions> {
        self.compilation_options
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, AndOperation, Array as CoreArray, ArrayOperation, ArrayReference, Atan2Operation,
        AttachedRegions, CaptureReference, Cast, CastOperation, CompareOperation, ComparisonDirection,
        CompilationCacheDomain, CompositeJvpRule, CompositeOperation, ConditionOperation, ConstantOperation,
        Dimension as CoreDimension, DimensionAddOperation, DimensionBounds, DimensionDivOperation,
        DimensionSizeOperation, DimensionToScalarOperation, DimensionVariable, DivOperation, DynamicBroadcastOperation,
        DynamicReshapeOperation, DynamicSliceOperation, DynamicUpdateSliceOperation, Fill, Memory, MulOperation,
        NegOperation, OneOperation, ReferenceNewOperation, ReferenceReadOperation, ReferenceType, ScanOperation,
        SelectOperation, ShardMapBoundary, ShardMapOperation, Sharding, ShardingDimension, StaticShape, SubOperation,
        Typed, WhileOperation, ZeroOperation,
    };
    use ryft_pjrt::{ClientOptions, CpuClientOptions, ExecutionFence, load_cpu_plugin};

    use crate::Reference;
    use crate::domains::sessions::Session;
    use crate::domains::tests::{
        array_domain, boolean_scalar, boolean_vector, domain_mesh, f32_scalar, f32_vector, f64_vector, program_array,
        read_booleans, read_f32s, read_f64s, read_i64s, read_u64s, replicated_scalar_type, replicated_vector_type,
    };
    use crate::operations::jit_call::JitCallOperation;
    use crate::tests::{execution_client, hash_of, values_from_bytes, values_to_bytes};

    use super::*;

    #[test]
    fn test_domain_zero_constructs_a_bufferless_logical_zero_space_array() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh);
        let array_type = ArrayType::new(DataType::Zero, Shape::new(vec![CoreDimension::Static(3)]));

        let array = domain.constant(&array_type, ConstantKind::Zero).unwrap();

        assert_eq!(array.data_type(), DataType::Zero);
        assert!(array.addressable_shards().next().unwrap().buffer().is_none());
    }

    #[test]
    fn test_domain_zero_defaults_missing_sharding_to_replicated() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 2);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());

        let array_type =
            ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(3), CoreDimension::Static(2)]));
        let array = domain.constant(&array_type, ConstantKind::Zero).unwrap();

        assert_eq!(array.shape(), StaticShape::new(vec![3, 2]));
        assert_eq!(array.shards().len(), 2);
        assert_eq!(array.addressable_shards().count(), 2);
        for shard in array.addressable_shards() {
            let buffer = shard.buffer().unwrap();
            let host_bytes = buffer.copy_to_host(None).unwrap().r#await().unwrap();
            let values = values_from_bytes::<f32>(host_bytes.as_slice());
            assert_eq!(values, vec![0.0; 6]);
        }
    }

    #[test]
    fn test_domain_one_fills_sharded_array_with_ones() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 2);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(sharding)
            .unwrap();
        let domain = Session::new(&client).domain().with_mesh(mesh);

        let array = domain.constant(&array_type, ConstantKind::One).unwrap();

        assert_eq!(array.shape(), StaticShape::new(vec![4]));
        assert_eq!(array.shards().len(), 2);
        assert_eq!(array.addressable_shards().count(), 2);
        for shard in array.addressable_shards() {
            assert_eq!(shard.shape(), StaticShape::new(vec![2]));
            let buffer = shard.buffer().unwrap();
            let host_bytes = buffer.copy_to_host(None).unwrap().r#await().unwrap();
            let values = values_from_bytes::<f32>(host_bytes.as_slice());
            assert_eq!(values, vec![1.0, 1.0]);
        }
    }

    #[test]
    fn test_domain_identity_synthesis_rejects_unsupported_constant_type() {
        use ryft_core::OneOperation;
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let array_type = ArrayType::scalar(DataType::Token);

        assert!(matches!(
            domain.bind(OneOperation::new(array_type.clone()), Vec::new(), &[]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "data type `token` cannot represent one"
        ));
    }

    #[test]
    fn test_domain_identity_fast_path_rejects_attached_regions() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh);
        let array_type = ArrayType::scalar(DataType::F32);
        let operations = [
            Operation::provide(ZeroOperation::new(ArrayIrType::Array(array_type.clone())), &[]).unwrap(),
            OneOperation::new(array_type).into(),
        ];

        for operation in operations {
            let name = operation.name();
            let attached_region = ProgramBuilder::new()
                .build::<Vec<Constant>, Vec<Constant>>(Vec::new(), Vec::new(), Vec::new())
                .unwrap();
            assert_eq!(
                domain.bind(operation, vec![attached_region], &[]),
                Err(ProgramError::MalformedProgram(format!(
                    "operation `{name}` declares no region slots but 1 regions were attached",
                ))),
            );
        }
    }

    #[test]
    fn test_eager_dimension_dispatch_runs_on_the_host_without_compilation() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh);
        let left = DimensionValue::constant(2).unwrap();
        let right = DimensionValue::constant(3).unwrap();
        let add = DimensionAddOperation::new(left.r#type().as_ref(), right.r#type().as_ref()).unwrap();

        let output = domain
            .bind(
                Operation::Dimension(DimensionOperation::Add(add)),
                Vec::new(),
                &[
                    Value::Dimension(Dimension::new(left, domain.clone())),
                    Value::Dimension(Dimension::new(right, domain.clone())),
                ],
            )
            .unwrap()
            .remove(0);
        let Value::Dimension(output) = output else {
            panic!("dimension arithmetic must produce a dimension value");
        };
        assert_eq!(output.value().extent(), 5);
        assert_eq!(domain.program_count(), 0);

        let scalar = domain
            .bind(DimensionToScalarOperation, Vec::new(), &[Value::Dimension(output.clone())])
            .unwrap()
            .remove(0);
        let Value::Array(scalar) = scalar else {
            panic!("dimension_to_scalar must produce an array");
        };
        assert_eq!(read_i64s(&client, &scalar), vec![5]);
        assert_eq!(domain.program_count(), 0);

        let from_scalar = domain
            .bind(
                ryft_core::DimensionFromScalarOperation::new(output.r#type().into_owned()),
                Vec::new(),
                &[Value::Array(scalar)],
            )
            .unwrap()
            .remove(0);
        let Value::Dimension(from_scalar) = from_scalar else {
            panic!("dimension_from_scalar must produce a dimension value");
        };
        assert_eq!(from_scalar.value().extent(), 5);
        assert_eq!(domain.program_count(), 0);
    }

    #[test]
    fn test_eager_mixed_shape_operations_specialize_dimensions_and_share_the_array_kernel_cache() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let input = f32_vector(&domain, &mesh, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let extent = domain
            .bind(
                DimensionSizeOperation::new(input.r#type().as_ref(), 0).unwrap(),
                Vec::new(),
                &[Value::Array(input.clone())],
            )
            .unwrap()
            .remove(0);
        let Value::Dimension(extent) = extent else {
            panic!("dimension_size must produce a dimension value");
        };
        assert_eq!(extent.value().extent(), 6);
        assert_eq!(domain.program_count(), 0);

        let two = DimensionValue::constant(2).unwrap();
        let division = DimensionDivOperation::new(extent.r#type().as_ref(), two.r#type().as_ref()).unwrap();
        let three = domain
            .bind(
                Operation::Dimension(DimensionOperation::Div(division)),
                Vec::new(),
                &[Value::Dimension(extent), Value::Dimension(Dimension::new(two.clone(), domain.clone()))],
            )
            .unwrap()
            .remove(0);
        let Value::Dimension(three) = three else {
            panic!("dimension division must produce a dimension value");
        };
        assert_eq!(three.value().extent(), 3);
        assert_eq!(domain.program_count(), 0);

        let reshape_inputs = [
            Value::Array(input),
            Value::Dimension(Dimension::new(two.clone(), domain.clone())),
            Value::Dimension(three.clone()),
        ];
        let reshaped = domain.bind(DynamicReshapeOperation::new(), Vec::new(), &reshape_inputs).unwrap().remove(0);
        assert_eq!(program_array(&reshaped).shape().as_slice(), &[2, 3]);
        assert_eq!(domain.program_count(), 1);

        let four = DimensionValue::constant(4).unwrap();
        let broadcast_inputs = [
            reshaped,
            Value::Dimension(Dimension::new(four, domain.clone())),
            Value::Dimension(Dimension::new(two, domain.clone())),
            Value::Dimension(three),
        ];
        let broadcast = domain
            .bind(DynamicBroadcastOperation::new(vec![1, 2]), Vec::new(), &broadcast_inputs)
            .unwrap()
            .remove(0);
        assert_eq!(program_array(&broadcast).shape().as_slice(), &[4, 2, 3]);
        assert_eq!(read_f32s(&client, program_array(&broadcast)), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].repeat(4));
        assert_eq!(domain.program_count(), 2);

        let repeated = domain
            .bind(DynamicBroadcastOperation::new(vec![1, 2]), Vec::new(), &broadcast_inputs)
            .unwrap()
            .remove(0);
        assert_eq!(program_array(&repeated).shape().as_slice(), &[4, 2, 3]);
        assert_eq!(domain.program_count(), 2);
    }

    #[test]
    fn test_eager_bind_executes_binary_operation() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = array_domain(&client);

        let left = f32_vector(domain.parent(), &mesh, &[1.0, 2.0, 3.0, 4.0]);
        let right = f32_vector(domain.parent(), &mesh, &[10.0, 20.0, 30.0, 40.0]);
        let outputs = domain.bind(AddOperation::new(), Vec::new(), &[left, right]).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(read_f32s(&client, &outputs[0]), vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_eager_bind_executes_dimension_cast() {
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let selected = domain.fork_effect_scope();
        let value = Value::Dimension(Dimension::new(DimensionValue::constant(2).unwrap(), domain.clone()));
        let output_type = ArrayIrType::Dimension(DimensionType::from(CoreDimension::Static(2)));
        for checked in [true, false] {
            let output = if checked {
                value.cast(output_type.clone(), &[])
            } else {
                value.cast_unchecked(output_type.clone(), &[])
            }
            .unwrap();
            assert_eq!(output, value);
            assert!(Arc::ptr_eq(output.backend_domain().session(), domain.session()));
            assert!(Arc::ptr_eq(&output.backend_domain().effect_scope, &domain.effect_scope));

            let operation = if checked {
                CastOperation::checked(output_type.clone(), vec![output_type.clone()]).unwrap()
            } else {
                CastOperation::unchecked(output_type.clone(), vec![output_type.clone()]).unwrap()
            };
            let outputs = selected.bind(operation, Vec::new(), &[value.clone()]).unwrap();
            assert_eq!(outputs, vec![value.clone()]);
            assert!(Arc::ptr_eq(&outputs[0].backend_domain().effect_scope, &selected.effect_scope));
        }
        assert_eq!(domain.program_count(), 0);
    }

    #[test]
    fn test_eager_bind_dimension_cast_validates_declared_relationships() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let value = Value::Dimension(Dimension::new(DimensionValue::constant(2).unwrap(), domain.clone()));
        let output_type = ArrayIrType::Dimension(DimensionType::from(CoreDimension::Static(2)));
        let extent = DimensionVariable::new("extent", DimensionBounds::new(0, Some(5)).unwrap());
        let declared_types = vec![
            DimensionType::from(extent.clone()).into(),
            ArrayType::new(DataType::F32, Shape::new(vec![extent.into()])).into(),
        ];
        let valid = Value::Array(f32_vector(&domain, &mesh, &[1.0, 2.0]));
        let invalid = Value::Array(f32_vector(&domain, &mesh, &[1.0, 2.0, 3.0]));
        for constructor in [CastOperation::checked, CastOperation::unchecked] {
            let operation = constructor(output_type.clone(), declared_types.clone()).unwrap();
            assert_eq!(
                domain.bind(operation.clone(), Vec::new(), &[value.clone(), valid.clone()]),
                Ok(vec![value.clone()]),
            );
            assert_eq!(
                domain.bind(operation.clone(), Vec::new(), &[value.clone(), invalid.clone()]),
                Err(ProgramError::Type(TypeError::custom(ryft_core::DimensionError::InputDimensionMismatch {
                    dimension: "extent".to_owned(),
                    expected: 2,
                    actual: 3,
                }))),
            );
            assert_eq!(
                domain.bind(operation, Vec::new(), &[value.clone()]),
                Err(TypeError::invalid("expected 2 inputs but got 1").into()),
            );
            let mismatch = ArrayIrType::Dimension(DimensionType::from(CoreDimension::Static(3)));
            assert_eq!(
                domain.bind(constructor(mismatch, vec![output_type.clone()]).unwrap(), Vec::new(), &[value.clone()]),
                Err(TypeError::invalid("`cast` cannot convert type `dimension<2>` to unrelated type `dimension<3>`")
                    .into()),
            );
        }
        let region = ProgramBuilder::new()
            .build::<Vec<Constant>, Vec<Constant>>(Vec::new(), Vec::new(), Vec::new())
            .unwrap();
        assert_eq!(
            domain.bind(
                CastOperation::checked(output_type.clone(), vec![output_type]).unwrap(),
                vec![region],
                &[value],
            ),
            Err(ProgramError::MalformedProgram(
                "operation `cast` declares no region slots but 1 regions were attached".to_owned(),
            )),
        );
        assert_eq!(domain.program_count(), 0);
    }

    #[test]
    fn test_eager_bind_dimension_cast_preserves_witness_errors() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let value = Value::Dimension(Dimension::new(DimensionValue::constant(2).unwrap(), domain.clone()));
        let output_type = ArrayIrType::Dimension(DimensionType::from(CoreDimension::Static(2)));
        for data_type in [DataType::F32, DataType::Zero] {
            let (event, promise) = client.event(()).unwrap();
            promise.set(Some(ryft_pjrt::Error::invalid_argument("cast witness failed"))).unwrap();
            let bytes = if data_type.is_zero() { Vec::new() } else { values_to_bytes::<f32>(&[1.0, 2.0]) };
            let witness = Value::Array(
                Array::from_host_buffer(&domain, ArrayType::new_static(data_type, [2]), mesh.clone(), bytes)
                    .unwrap()
                    .with_execution_fence(ExecutionFence::new(vec![event])),
            );
            for checked in [true, false] {
                let wrapped = ArrayIrValue::<Array<'_>>::Dimension(DimensionValue::constant(2).unwrap());
                let Value::Array(array) = &witness else { unreachable!() };
                let witnesses = [ArrayIrValue::Array(array.clone())];
                let result = if checked {
                    wrapped.cast(output_type.clone(), &witnesses)
                } else {
                    wrapped.cast_unchecked(output_type.clone(), &witnesses)
                };
                assert_eq!(
                    result,
                    Err(ProgramError::Concretization {
                        message: "host dimension `cast` cannot retain an array witness producer; bind the cast \
                                  through an array IR context"
                            .to_owned(),
                    })
                );
                let result = if checked {
                    value.cast(output_type.clone(), &[witness.clone()])
                } else {
                    value.cast_unchecked(output_type.clone(), &[witness.clone()])
                };
                assert_eq!(result, Err(ProgramError::InvalidArgument { message: "cast witness failed".to_owned() }),);
            }
        }
        assert_eq!(domain.program_count(), 0);
    }

    #[test]
    fn test_eager_bind_rejects_unresolved_references_before_special_cases_or_tracing() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();

        // Top-level reference operations execute eagerly on their host-side handles (their full lifecycle is covered
        // by the `values` tests), so they validate their inputs like the core reference implementations do.
        assert!(domain.bind(Operation::ReferenceNew(ReferenceNewOperation::new()), Vec::new(), &[]).is_err());

        let reference =
            Value::Reference(Reference::new(ArrayReference::new(f32_vector(&domain, &mesh, &[1.0])), domain.clone()));
        assert_eq!(
            domain.bind(AddOperation::new(), Vec::new(), &[reference]),
            Err(ProgramError::UnsupportedOperation {
                message: "XLA eager execution lowers each operation and cannot bind a reference operand; discharge \
                          references before lowering"
                    .to_string(),
            }),
        );

        // Region-carried state must be rejected too: control-flow operations report pure intrinsic effects, so the
        // guard has to union the driver's region-program effects to see a reference operation inside a loop body.
        let condition = {
            let mut builder = ProgramBuilder::new();
            let predicate = builder
                .add_instruction(ZeroOperation::new(ArrayType::scalar(DataType::Boolean)), Vec::new(), Vec::new(), None)
                .unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![predicate], Vec::new(), vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::new();
            let state = builder
                .add_instruction(ZeroOperation::new(ArrayType::scalar(DataType::F32)), Vec::new(), Vec::new(), None)
                .unwrap()[0];
            let reference =
                builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap();
            builder.build::<Vec<Constant>, Vec<Constant>>(Vec::new(), Vec::new(), Vec::new()).unwrap()
        };
        assert_eq!(
            domain.bind(Operation::While(WhileOperation::new()), vec![condition, body], &[]),
            Err(ProgramError::UnsupportedOperation {
                message: "`while` carries reference state that XLA eager execution cannot lower; discharge \
                          references before lowering"
                    .to_string(),
            }),
        );

        // Reference state confined to a dormant custom-derivative rule region is rejected too: ordinary XLA rejects
        // unresolved reference artifacts program-wide, so eager binding matches that policy instead of deferring the
        // failure to compilation with a less targeted diagnostic.
        let primal = {
            let mut builder = ProgramBuilder::new();
            let state = builder.add_input(ArrayType::scalar(DataType::F32).into());
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![state], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };
        let rule = {
            let mut builder = ProgramBuilder::new();
            let state = builder.add_input(ArrayType::scalar(DataType::F32).into());
            let state_tangent = builder.add_input(ArrayType::scalar(DataType::F32).into());
            let reference =
                builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
            let output =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(
                    vec![output, state_tangent],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        assert_eq!(
            domain.bind(
                Operation::Composite(CompositeOperation::from_rule_regions(CompositeJvpRule::Explicit, false)),
                vec![primal.clone(), rule],
                &[],
            ),
            Err(ProgramError::UnsupportedOperation {
                message: "`composite` carries reference state that XLA eager execution cannot lower; \
                          discharge references before lowering"
                    .to_string(),
            }),
        );

        // A pure reference-typed capture in the dormant rule has no ordered-state effect, so only the complete-arena
        // type scan can keep eager and compiled XLA rejection policy aligned.
        let pure_reference_rule = {
            let mut builder = ProgramBuilder::new();
            let state = builder.add_input(ArrayType::scalar(DataType::F32).into());
            let state_tangent = builder.add_input(ArrayType::scalar(DataType::F32).into());
            builder.add_constant(Constant::Captured(CaptureReference::new(
                0,
                ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32))),
            )));
            builder
                .build::<Vec<Constant>, Vec<Constant>>(
                    vec![state, state_tangent],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        assert_eq!(
            domain.bind(
                Operation::Composite(CompositeOperation::from_rule_regions(CompositeJvpRule::Explicit, false)),
                vec![primal, pure_reference_rule],
                &[],
            ),
            Err(ProgramError::UnsupportedOperation {
                message: "`composite` carries reference state that XLA eager execution cannot lower; \
                          discharge references before lowering"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_eager_bind_executes_promoted_and_broadcast_elementwise_operations() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = array_domain(&client);

        let scalar = f32_scalar(domain.parent(), &mesh, 2.0);
        let vector = f64_vector(domain.parent(), &mesh, &[1.0, 2.0, 4.0, 8.0]);

        let divide = domain.bind(DivOperation::new(), Vec::new(), &[scalar.clone(), vector.clone()]).unwrap();
        assert_eq!(read_f64s(&client, &divide[0]), vec![2.0, 1.0, 0.5, 0.25]);

        let atan2 = domain.bind(Atan2Operation::new(), Vec::new(), &[vector.clone(), scalar.clone()]).unwrap();
        for (actual, expected) in read_f64s(&client, &atan2[0]).into_iter().zip([
            1.0f64.atan2(2.0),
            2.0f64.atan2(2.0),
            4.0f64.atan2(2.0),
            8.0f64.atan2(2.0),
        ]) {
            assert!((actual - expected).abs() < 1e-12, "got {actual}, expected {expected}");
        }

        let compare = domain
            .bind(CompareOperation::new(ComparisonDirection::LessThan), Vec::new(), &[scalar.clone(), vector.clone()])
            .unwrap();
        assert_eq!(read_booleans(&client, &compare[0]), vec![false, false, true, true]);

        let select = domain
            .bind(SelectOperation::new(), Vec::new(), &[boolean_scalar(domain.parent(), &mesh, true), scalar, vector])
            .unwrap();
        assert_eq!(read_f64s(&client, &select[0]), vec![2.0, 2.0, 2.0, 2.0]);

        let boolean_vector = boolean_vector(domain.parent(), &mesh, &[true, false, true, false]);
        let and = domain
            .bind(AndOperation::new(), Vec::new(), &[boolean_scalar(domain.parent(), &mesh, true), boolean_vector])
            .unwrap();
        assert_eq!(read_booleans(&client, &and[0]), vec![true, false, true, false]);
    }

    #[test]
    fn test_eager_bind_executes_unary_operation() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = array_domain(&client);

        let input = f32_vector(domain.parent(), &mesh, &[1.0, -2.0, 3.5, 0.0]);
        let outputs = domain.bind(NegOperation::new(), Vec::new(), &[input]).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(read_f32s(&client, &outputs[0]), vec![-1.0, 2.0, -3.5, 0.0]);
    }

    #[test]
    fn test_eager_fill_materializes_scalar_literal_then_broadcasts_over_a_default_mesh() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let domain = array_domain(&client);

        for memory in [Memory::Device, Memory::Host { pinned: true }, Memory::Host { pinned: false }] {
            let r#type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(3)])).with_memory(memory);
            let output = domain.fill(&r#type, 2.5f64).unwrap();

            assert_eq!(output.r#type().memory(), memory);
            assert_eq!(output.shape(), StaticShape::new(vec![3]));
            assert_eq!(read_f32s(&client, &output), vec![2.5, 2.5, 2.5]);
        }

        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(2)]));
        let value = num_complex::Complex::new(1.5f64, -2.0f64);
        let output = domain.fill(&r#type, value).unwrap();
        assert_eq!(read_f32s(&client, &output), vec![1.5, 1.5]);

        // A complex fill value lowers as two real part splats composed through `stablehlo.complex`, and a `c64`
        // buffer's bytes are the interleaved `f32` real and imaginary parts of its elements.
        let r#type = ArrayType::new(DataType::C64, Shape::new(vec![CoreDimension::Static(2)]));
        let value = num_complex::Complex::new(1.5f32, -2.0f32);
        let output = domain.fill(&r#type, value).unwrap();

        assert_eq!(output.shape(), StaticShape::new(vec![2]));
        assert_eq!(read_f32s(&client, &output), vec![1.5, -2.0, 1.5, -2.0]);

        // Integer literals remain exact through conversion and scalar constant lowering, including values that
        // cannot be represented exactly by the floating-point intermediary used for floating-point constants.
        let r#type = ArrayType::new(DataType::U64, Shape::new(vec![CoreDimension::Static(2)]));
        let value = u64::MAX - 1;
        let output = domain.fill(&r#type, value).unwrap();
        assert_eq!(read_u64s(&client, &output), vec![value, value]);
    }

    #[test]
    fn test_eager_bind_reuses_cached_executable_for_repeated_operations() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = array_domain(&client);
        assert_eq!(domain.parent().program_count(), 0);

        let left = f32_vector(domain.parent(), &mesh, &[1.0, 2.0]);
        let right = f32_vector(domain.parent(), &mesh, &[3.0, 4.0]);
        let first = domain.bind(AddOperation::new(), Vec::new(), &[left.clone(), right.clone()]).unwrap();
        assert_eq!(domain.parent().program_count(), 1);

        let second = domain.bind(AddOperation::new(), Vec::new(), &[left, right]).unwrap();
        assert_eq!(domain.parent().program_count(), 1, "a repeated eager operation must be a compile-cache hit");
        assert_eq!(read_f32s(&client, &first[0]), read_f32s(&client, &second[0]));

        // A different input signature compiles (and caches) a distinct executable.
        let wider_left = f32_vector(domain.parent(), &mesh, &[1.0, 2.0, 3.0]);
        let wider_right = f32_vector(domain.parent(), &mesh, &[4.0, 5.0, 6.0]);
        domain.bind(AddOperation::new(), Vec::new(), &[wider_left, wider_right]).unwrap();
        assert_eq!(domain.parent().program_count(), 2);
    }

    #[test]
    fn test_independent_arrays_share_compile_cache() {
        use ryft_core::Value as CoreValue;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();

        // Arrays constructed independently of each other in one session dispatch through their own recovered domains,
        // which share that session's compilation context so that the same eager operation compiles exactly once.
        let first = f32_vector(&domain, &mesh, &[1.0, 2.0]);
        let second = f32_vector(&domain, &mesh, &[3.0, 4.0]);
        let first_sum = ryft_core::Add::add(&first, &first).unwrap();
        let second_sum = ryft_core::Add::add(&second, &second).unwrap();
        let first_domain = first_sum.domain();
        let second_domain = second_sum.domain();
        assert!(std::ptr::eq(
            first_domain.parent().compilation_context(),
            second_domain.parent().compilation_context(),
        ));
        assert_eq!(first_domain.parent().compilation_context().statistics().compilations, 1);
        assert_eq!(read_f32s(&client, &first_sum), vec![2.0, 4.0]);
        assert_eq!(read_f32s(&client, &second_sum), vec![6.0, 8.0]);
    }

    #[test]
    fn test_eager_dispatch_cache_skips_tracing_and_lowering() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let left = f32_vector(&domain, &mesh, &[1.0, 2.0]);
        let right = f32_vector(&domain, &mesh, &[3.0, 4.0]);
        let inputs = [Value::Array(left), Value::Array(right)];

        let first = domain.bind(Operation::Array(AddOperation::new().into()), Vec::new(), &inputs).unwrap();
        let second = domain.bind(Operation::Array(AddOperation::new().into()), Vec::new(), &inputs).unwrap();
        assert_eq!(read_f32s(&client, program_array(&first[0])), vec![4.0, 6.0]);
        assert_eq!(read_f32s(&client, program_array(&second[0])), vec![4.0, 6.0]);
        let statistics = domain.eager_dispatch_statistics();
        assert_eq!((statistics.misses, statistics.hits, statistics.productions), (1, 1, 1));

        // The hit never reached the compilation cache, so it skipped tracing, lowering, and key construction.
        let compilation_statistics = domain.compilation_context().statistics();
        assert_eq!((compilation_statistics.misses, compilation_statistics.memory_hits), (1, 0));
        assert_eq!(compilation_statistics.compilations, 1);
    }

    #[test]
    fn test_eager_dispatch_cache_keys_every_lowering_input() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 2);
        let domain = Session::new(&client).domain();
        fn negate<'c>(domain: &Domain<'c>, input: Array<'c>) {
            let inputs = [Value::Array(input)];
            domain.bind(Operation::Array(NegOperation::new().into()), Vec::new(), &inputs).unwrap();
        }
        let misses = |domain: &Domain<'_>| domain.eager_dispatch_statistics().misses;

        let vector = f32_vector(&domain, &mesh, &[1.0, 2.0, 3.0, 4.0]);
        negate(&domain, vector.clone());
        negate(&domain, vector.clone());
        assert_eq!(misses(&domain), 1);

        // A different shape, element type, or sharding is a different key.
        negate(&domain, f32_vector(&domain, &mesh, &[1.0, 2.0]));
        assert_eq!(misses(&domain), 2);
        let f64_type = ArrayType::new(DataType::F64, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        negate(&domain, Array::from_host_buffer(&domain, f64_type, mesh.clone(), values_to_bytes(&[1f64; 4])).unwrap());
        assert_eq!(misses(&domain), 3);
        let sharded_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let sharded_bytes = values_to_bytes(&[1f32; 4]);
        negate(&domain, Array::from_host_buffer(&domain, sharded_type, mesh.clone(), sharded_bytes).unwrap());
        assert_eq!(misses(&domain), 4);

        // A domain with different compilation options does not reuse the entries of the default domain.
        let mut compilation_options = CompilationOptions::default();
        compilation_options.profile_version = 1;
        let sibling = domain.with_compilation_options(compilation_options);
        negate(&sibling, vector.clone().associate(&sibling).unwrap());
        assert_eq!(misses(&domain), 5);

        // Dimension inputs are keyed by their extents, because eager binding bakes them into the traced program as
        // constants of exact types. Dynamic reshapes execute on a single device, so this part uses a single-device
        // session.
        let single_device_client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let single_device_mesh = domain_mesh(&single_device_client, "x", 1);
        let single_device_domain = Session::new(&single_device_client).domain();
        let single_device_vector = f32_vector(&single_device_domain, &single_device_mesh, &[1.0, 2.0, 3.0, 4.0]);
        let reshape = |rows: &DimensionValue, columns: &DimensionValue| {
            let inputs = [
                Value::Array(single_device_vector.clone()),
                Value::Dimension(Dimension::new(rows.clone(), single_device_domain.clone())),
                Value::Dimension(Dimension::new(columns.clone(), single_device_domain.clone())),
            ];
            single_device_domain.bind(DynamicReshapeOperation::new(), Vec::new(), &inputs).unwrap().remove(0)
        };
        let rows = DimensionValue::constant(2).unwrap();
        let columns = DimensionValue::constant(2).unwrap();
        assert_eq!(program_array(&reshape(&rows, &columns)).shape().as_slice(), &[2, 2]);
        assert_eq!(program_array(&reshape(&rows, &columns)).shape().as_slice(), &[2, 2]);
        assert_eq!(misses(&single_device_domain), 1);
        let one = DimensionValue::constant(1).unwrap();
        let four = DimensionValue::constant(4).unwrap();
        assert_eq!(program_array(&reshape(&one, &four)).shape().as_slice(), &[1, 4]);
        assert_eq!(misses(&single_device_domain), 2);
        // A separately created dimension value with the same extent, and hence a distinct identity, is the same key.
        let other_rows = DimensionValue::constant(2).unwrap();
        assert_eq!(program_array(&reshape(&other_rows, &columns)).shape().as_slice(), &[2, 2]);
        assert_eq!(misses(&single_device_domain), 2);
    }

    #[test]
    fn test_eager_bind_shares_one_constant_for_repeated_dimension_inputs() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let vector = f32_vector(&domain, &mesh, &[1.0, 2.0, 3.0, 4.0]);
        let two = DimensionValue::constant(2).unwrap();
        let inputs = [
            Value::Array(vector),
            Value::Dimension(Dimension::new(two.clone(), domain.clone())),
            Value::Dimension(Dimension::new(two, domain.clone())),
        ];
        let output = domain.bind(DynamicReshapeOperation::new(), Vec::new(), &inputs).unwrap().remove(0);
        let output = program_array(&output);
        assert_eq!(output.shape().as_slice(), &[2, 2]);
        assert_eq!(read_f32s(&client, output), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_eager_bind_defines_separate_constants_for_distinct_dimension_identities() {
        // Dimension values compare by extent, but each distinct identity still needs its own constant, because a
        // constant defines its identity and the operation's type refers to both identities.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
        let bounds = DimensionBounds::new(1, Some(5)).unwrap();
        let rows = DimensionVariable::new("rows", bounds);
        let columns = DimensionVariable::new("columns", bounds);
        let operation = OneOperation::new(ArrayType::new(
            DataType::F32,
            Shape::new(vec![CoreDimension::Dynamic(rows.clone()), CoreDimension::Dynamic(columns.clone())]),
        ));
        let inputs = [
            Value::Dimension(Dimension::new(
                DimensionValue::new(DimensionType::from(rows), 2).unwrap(),
                domain.clone(),
            )),
            Value::Dimension(Dimension::new(
                DimensionValue::new(DimensionType::from(columns), 2).unwrap(),
                domain.clone(),
            )),
        ];
        let output = domain.bind(operation, Vec::new(), &inputs).unwrap().remove(0);
        let output = program_array(&output);
        assert_eq!(output.shape().as_slice(), &[2, 2]);
        assert_eq!(read_f32s(&client, output), vec![1.0; 4]);
    }

    #[test]
    fn test_eager_dispatch_cache_keys_attached_regions_structurally() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let vector_type = replicated_vector_type(&mesh, 2);
        let branch = |operation: Operation| {
            let mut builder = ProgramBuilder::new();
            let input = builder.add_input(vector_type.clone().into());
            let output = builder.add_instruction(operation, Vec::new(), vec![input, input], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };
        let inputs =
            [Value::Array(boolean_scalar(&domain, &mesh, true)), Value::Array(f32_vector(&domain, &mesh, &[1.0, 2.0]))];

        // Branches that are rebuilt for every application but are structurally equal share one cache entry.
        for _ in 0..2 {
            let branches = [
                branch(Operation::Array(AddOperation::new().into())),
                branch(Operation::Array(MulOperation::new().into())),
            ];
            let outputs = domain.bind(Operation::Condition(ConditionOperation::new()), branches, &inputs).unwrap();
            assert_eq!(read_f32s(&client, program_array(&outputs[0])), vec![2.0, 4.0]);
        }
        let statistics = domain.eager_dispatch_statistics();
        assert_eq!((statistics.misses, statistics.hits), (1, 1));

        // A structurally different branch is a different key.
        let branches = [
            branch(Operation::Array(SubOperation::new().into())),
            branch(Operation::Array(MulOperation::new().into())),
        ];
        let outputs = domain.bind(Operation::Condition(ConditionOperation::new()), branches, &inputs).unwrap();
        assert_eq!(read_f32s(&client, program_array(&outputs[0])), vec![0.0, 0.0]);
        assert_eq!(domain.eager_dispatch_statistics().misses, 2);
        assert_eq!(domain.compilation_context().statistics().compilations, 2);
    }

    #[test]
    fn test_eager_dispatch_query_compares_and_hashes_like_its_key() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let operation = Operation::Array(AddOperation::new().into());
        let inputs = [
            Value::Array(f32_vector(&domain, &mesh, &[1.0, 2.0])),
            Value::Dimension(Dimension::new(DimensionValue::constant(2).unwrap(), domain.clone())),
        ];
        let query = EagerDispatchQuery {
            operation: &operation,
            regions: &[],
            inputs: &inputs,
            mesh: Some(&mesh),
            compilation_options: &domain.compilation_options,
        };
        let key = EagerDispatchKey {
            operation: operation.clone(),
            inputs: (0..inputs.len()).map(|index| query.input(index).into_owned()).collect(),
            mesh: Some(mesh.clone()),
            compilation_options: Arc::clone(&domain.compilation_options),
            regions: Arc::new([]),
        };
        let query_hash = hash_of(&query as &dyn EagerDispatchKeyView);

        // Owned keys borrow as views that equal, and hash like, the queries that they were built from.
        let borrowed_key: &dyn EagerDispatchKeyView = std::borrow::Borrow::borrow(&key);
        assert!(borrowed_key == &query as &dyn EagerDispatchKeyView);
        assert_eq!(hash_of(borrowed_key), query_hash);
        assert_eq!(hash_of(&key), query_hash);

        // Compilation options are compared by identity, and every other component by value (dimensions by extent).
        let other_options = Arc::new(domain.compilation_options.as_ref().clone());
        let other_options_query = EagerDispatchQuery { compilation_options: &other_options, ..query };
        assert!(borrowed_key != &other_options_query as &dyn EagerDispatchKeyView);
        let other_inputs =
            [inputs[0].clone(), Value::Dimension(Dimension::new(DimensionValue::constant(3).unwrap(), domain.clone()))];
        let other_inputs_query = EagerDispatchQuery { inputs: &other_inputs, ..query };
        assert!(borrowed_key != &other_inputs_query as &dyn EagerDispatchKeyView);
        let meshless_query = EagerDispatchQuery { mesh: None, ..query };
        assert!(borrowed_key != &meshless_query as &dyn EagerDispatchKeyView);
    }

    #[test]
    fn test_eager_dispatch_cache_does_not_retain_failures() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let inputs = [
            Value::Array(f32_vector(&domain, &mesh, &[1.0, 2.0])),
            Value::Array(f32_vector(&domain, &mesh, &[1.0, 2.0, 3.0])),
        ];
        for _ in 0..2 {
            assert!(domain.bind(Operation::Array(AddOperation::new().into()), Vec::new(), &inputs).is_err());
        }
        let statistics = domain.eager_dispatch_statistics();
        assert_eq!((statistics.misses, statistics.hits), (2, 0));
        assert_eq!((statistics.productions, statistics.abandoned_productions), (0, 2));
    }

    #[test]
    fn test_eager_dispatch_cache_hits_still_validate_placement() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let other_client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let other_domain = Session::new(&other_client).domain();
        let local = f32_vector(&domain, &mesh, &[1.0, 2.0]);
        let foreign = f32_vector(&other_domain, &mesh, &[3.0, 4.0]);
        let operation = Operation::Array(AddOperation::new().into());
        let local_inputs = [Value::Array(local.clone()), Value::Array(local.clone())];
        domain.bind(operation.clone(), Vec::new(), &local_inputs).unwrap();

        // The second input's type and the first input's mesh match the cached entry, so this lookup hits the cache,
        // and placement validation must still reject the foreign input.
        let mixed_inputs = [Value::Array(local), Value::Array(foreign)];
        let error = domain.bind(operation, Vec::new(), &mixed_inputs).unwrap_err();
        assert!(error.to_string().contains("different PJRT client"), "{error}");
        assert_eq!(domain.eager_dispatch_statistics().hits, 1);
    }

    #[test]
    fn test_clear_cache_clears_the_eager_dispatch_cache() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let inputs = [Value::Array(f32_vector(&domain, &mesh, &[1.0, 2.0]))];
        domain.bind(Operation::Array(NegOperation::new().into()), Vec::new(), &inputs).unwrap();
        assert_eq!((domain.session().eager_dispatch_cache.len(), domain.program_count()), (1, 1));
        domain.clear_cache();
        assert_eq!((domain.session().eager_dispatch_cache.len(), domain.program_count()), (0, 0));
        domain.bind(Operation::Array(NegOperation::new().into()), Vec::new(), &inputs).unwrap();
        assert_eq!(domain.eager_dispatch_statistics().misses, 2);
    }

    /// Measures eager dispatch overhead for `.tasks/plan_eager_dispatch_front_cache.md`. Compares repeated eager `add`
    /// calls against a warm jitted function over the same input, and then times each step of the eager dispatch path
    /// (tracing, lowering, compilation-key construction, compile-cache lookup, and execution) in isolation.
    #[test]
    #[ignore = "phase 0 baseline measurement"]
    fn test_baseline_eager_dispatch_overhead() {
        use std::time::Instant;

        use crate::jit::{CompileTracer, JittedFunction, jitted};

        const REPETITIONS: usize = 10_000;
        let measure = |label: &str, function: &mut dyn FnMut()| {
            for _ in 0..100 {
                function();
            }
            let start = Instant::now();
            for _ in 0..REPETITIONS {
                function();
            }
            let mean = start.elapsed().as_secs_f64() * 1e6 / REPETITIONS as f64;
            println!("phase 0 baseline: {label}: {mean:.2} us/op");
        };

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let input = f32_vector(&domain, &mesh, &[1.0; 8]);

        measure("eager add (end to end)", &mut || {
            ryft_core::Add::add(&input, &input).unwrap();
        });
        let function: JittedFunction<'_, _, (), ArrayType, ArrayType> =
            jitted(|_, value: CompileTracer<'_>| Ok(ryft_core::Add::add(&value, &value)?), &domain, mesh.clone());
        measure("jitted add (warm dispatch)", &mut || {
            function.call((), input.clone()).unwrap();
        });
        let vector_type = ArrayIrType::Array(input.r#type().into_owned());
        let branch = |operation: Operation| {
            let mut builder = ProgramBuilder::new();
            let value = builder.add_input(vector_type.clone());
            let output = builder.add_instruction(operation, Vec::new(), vec![value, value], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };
        let condition_inputs = [Value::Array(boolean_scalar(&domain, &mesh, true)), Value::Array(input.clone())];
        measure("eager condition with freshly built branches (end to end)", &mut || {
            let branches = [
                branch(Operation::Array(AddOperation::new().into())),
                branch(Operation::Array(MulOperation::new().into())),
            ];
            domain.bind(Operation::Condition(ConditionOperation::new()), branches, &condition_inputs).unwrap();
        });

        let array_type = ArrayIrType::Array(input.r#type().into_owned());
        let trace = || -> FlatProgram {
            let mut builder = ProgramBuilder::new();
            let left = builder.add_input(array_type.clone());
            let right = builder.add_input(array_type.clone());
            let operation = Operation::Array(AddOperation::new().into());
            let outputs = builder.add_instruction(operation, Vec::new(), vec![left, right], None).unwrap().to_vec();
            builder.build(outputs, vec![Placeholder; 2], vec![Placeholder; 1]).unwrap()
        };
        let program = trace();
        let options = Options::new(mesh.clone());
        let lowered = domain.lower_program(&program, 0, &options).unwrap();
        let compiled = domain.compilation_context().compile(&domain, &lowered).unwrap();
        measure("stage: trace one-instruction program", &mut || {
            trace();
        });
        measure("stage: lower to StableHLO", &mut || {
            domain.lower_program(&program, 0, &options).unwrap();
        });
        measure("stage: build compilation key", &mut || {
            domain.compilation_key(&lowered).unwrap();
        });
        measure("stage: cached compilation (hit)", &mut || {
            domain.compilation_context().compile(&domain, &lowered).unwrap();
        });
        measure("stage: execute", &mut || {
            domain.execute_program(&compiled, vec![input.clone(), input.clone()]).unwrap();
        });
    }

    #[test]
    fn test_eager_bind_rejects_inputs_placed_on_a_foreign_device() {
        let plugin = load_cpu_plugin().unwrap();
        let domain_client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let foreign_client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let foreign_mesh = domain_mesh(&foreign_client, "x", 2);
        let domain = array_domain(&domain_client);
        let foreign_domain = Session::new(&foreign_client).domain();

        // An input that belongs to a session over a different client is rejected by client identity.
        let input = f32_vector(&foreign_domain, &foreign_mesh, &[1.0, 2.0]);
        assert!(matches!(
            domain.bind(NegOperation::new(), Vec::new(), &[input.clone()]),
            Err(ProgramError::InvalidArgument { message })
                if message == "received incompatible devices for eager xla execution: input #0 is owned by a \
                    different PJRT client than this domain's client",
        ));
        // Dimension-result casts execute on the host but still require array witnesses from the selected client.
        let dimension_type = ArrayIrType::Dimension(DimensionType::from(CoreDimension::Static(2)));
        let dimension = Value::Dimension(Dimension::new(DimensionValue::constant(2).unwrap(), domain.parent().clone()));
        assert_eq!(
            domain.parent().bind(
                CastOperation::checked(
                    dimension_type.clone(),
                    vec![dimension_type, input.r#type().into_owned().into()],
                )
                .unwrap(),
                Vec::new(),
                &[dimension, Value::Array(input)],
            ),
            Err(ProgramError::InvalidArgument {
                message: "received incompatible devices for eager xla execution: input #0 is owned by a different \
                          PJRT client than this domain's client"
                    .to_owned(),
            }),
        );
    }

    #[test]
    fn test_eager_bind_executes_condition_with_concrete_predicate() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let vector_type = replicated_vector_type(&mesh, 4);

        let doubled = {
            let mut builder = ProgramBuilder::new();
            let input = builder.add_input(vector_type.clone().into());
            let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };
        let squared = {
            let mut builder = ProgramBuilder::new();
            let input = builder.add_input(vector_type.clone().into());
            let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };
        let operation = Operation::Condition(ConditionOperation::new());

        let input = f32_vector(&domain, &mesh, &[1.0, 2.0, 3.0, 4.0]);
        let true_outputs = domain
            .bind(
                operation.clone(),
                [doubled.clone(), squared.clone()],
                &[Value::Array(boolean_scalar(&domain, &mesh, true)), Value::Array(input.clone())],
            )
            .unwrap();
        assert_eq!(read_f32s(&client, program_array(&true_outputs[0])), vec![2.0, 4.0, 6.0, 8.0]);

        let false_outputs = domain
            .bind(
                operation,
                [doubled, squared],
                &[Value::Array(boolean_scalar(&domain, &mesh, false)), Value::Array(input)],
            )
            .unwrap();
        assert_eq!(read_f32s(&client, program_array(&false_outputs[0])), vec![1.0, 4.0, 9.0, 16.0]);
        assert_eq!(domain.program_count(), 1, "both predicate values must share one compiled executable");
    }

    #[test]
    fn test_eager_bind_executes_condition_over_branches_refined_by_its_inputs() {
        // Eager compilation attaches the branches as provided. Their bounded `f32[rows]` input type is refined by the
        // static `f32[4]` input, so the eagerly executed output takes the refined static type rather than the bounded
        // branch output type.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let rows = DimensionVariable::new("rows", DimensionBounds::positive(Some(8)).unwrap());
        let branch_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Dynamic(rows)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let branch = |squares: bool| {
            let mut builder = ProgramBuilder::new();
            let input = builder.add_input(branch_type.clone().into());
            let output = if squares {
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0]
            } else {
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0]
            };
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };

        let outputs = domain
            .bind(
                Operation::Condition(ConditionOperation::new()),
                [branch(false), branch(true)],
                &[
                    Value::Array(boolean_scalar(&domain, &mesh, false)),
                    Value::Array(f32_vector(&domain, &mesh, &[1.0, 2.0, 3.0, 4.0])),
                ],
            )
            .unwrap();
        assert_eq!(program_array(&outputs[0]).shape(), StaticShape::new(vec![4]));
        assert_eq!(read_f32s(&client, program_array(&outputs[0])), vec![1.0, 4.0, 9.0, 16.0]);
    }

    #[test]
    fn test_eager_bind_condition_branches_consume_forwarded_dimension_authority() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let extent = DimensionValue::constant(3).unwrap();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);

        let branch = |negate: bool| {
            let mut builder = ProgramBuilder::new();
            let extent = builder.add_input(extent.r#type().into_owned().into());
            let scalar = builder.add_input(scalar_type.clone().into());
            let scalar = if negate {
                builder.add_instruction(NegOperation::new(), Vec::new(), vec![scalar], None).unwrap()[0]
            } else {
                scalar
            };
            let output = builder
                .add_instruction(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), vec![scalar, extent], None)
                .unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
                .unwrap()
        };
        let operation = Operation::Condition(ConditionOperation::new());
        let scalar = f32_scalar(&domain, &mesh, 2.0);

        let true_outputs = domain
            .bind(
                operation.clone(),
                [branch(false), branch(true)],
                &[
                    Value::Array(boolean_scalar(&domain, &mesh, true)),
                    Value::Dimension(Dimension::new(extent.clone(), domain.clone())),
                    Value::Array(scalar.clone()),
                ],
            )
            .unwrap();
        assert_eq!(program_array(&true_outputs[0]).shape(), StaticShape::new(vec![3]));
        assert_eq!(read_f32s(&client, program_array(&true_outputs[0])), vec![2.0; 3]);

        let false_outputs = domain
            .bind(
                operation,
                [branch(false), branch(true)],
                &[
                    Value::Array(boolean_scalar(&domain, &mesh, false)),
                    Value::Dimension(Dimension::new(extent, domain.clone())),
                    Value::Array(scalar),
                ],
            )
            .unwrap();
        assert_eq!(read_f32s(&client, program_array(&false_outputs[0])), vec![-2.0; 3]);
        assert_eq!(domain.program_count(), 1, "both branch selections must share one compiled executable");
    }

    #[test]
    fn test_eager_bind_executes_bounded_while() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);

        // Loop `state = state + 1` while `state < 3`, starting from `0`.
        let condition = {
            let mut builder = ProgramBuilder::new();
            let state = builder.add_input(scalar_type.clone().into());
            let literal = CoreArray::from_elements(scalar_type.clone(), &[3.0f32]).unwrap();
            let limit = builder.add_instruction(ConstantOperation::new(literal), Vec::new(), vec![], None).unwrap()[0];
            let predicate = builder
                .add_instruction(
                    Operation::Array(ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::LessThan))),
                    Vec::new(),
                    vec![state, limit],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![predicate], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::new();
            let state = builder.add_input(scalar_type.clone().into());
            let one =
                builder.add_instruction(OneOperation::new(scalar_type.clone()), Vec::new(), vec![], None).unwrap()[0];
            let next = builder.add_instruction(AddOperation::new(), Vec::new(), vec![state, one], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![next], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };
        let operation = Operation::While(WhileOperation::new());

        let outputs = domain
            .bind(operation, vec![condition, body], &[Value::Array(f32_scalar(&domain, &mesh, 0.0))])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(read_f32s(&client, program_array(&outputs[0])), vec![3.0]);
    }

    #[test]
    fn test_eager_bind_executes_elementwise_operation_on_sharded_inputs() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 2);
        let domain = array_domain(&client);

        // A vector sharded over a 2-device mesh executes eagerly through per-operation SPMD compilation: each device
        // adds its own 2-element shard.
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(sharding.clone())
            .unwrap();
        let input = Array::from_host_buffer(
            domain.parent(),
            input_type,
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let outputs = domain.bind(AddOperation::new(), Vec::new(), &[input.clone(), input]).unwrap();

        assert_eq!(outputs.len(), 1);
        let output = &outputs[0];
        // The input sharding propagates to the output type, so the result stays sharded over the same mesh axis.
        assert_eq!(output.sharding(), &sharding);
        assert_eq!(output.shape(), StaticShape::new(vec![4]));
        assert_eq!(output.shards().len(), 2);
        let shard_values = output
            .addressable_shards()
            .map(|shard| {
                let bytes = shard.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap();
                values_from_bytes::<f32>(bytes.as_slice())
            })
            .collect::<Vec<_>>();
        assert_eq!(shard_values, vec![vec![2.0, 4.0], vec![6.0, 8.0]]);
    }

    #[test]
    fn test_eager_bind_clamps_unsigned_dynamic_indices_without_narrowing() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let input = Value::Array(f32_vector(&domain, &mesh, &[1.0, 2.0, 3.0]));
        let index = Value::Array(
            Array::from_host_buffer(&domain, ArrayType::scalar(DataType::U64), mesh.clone(), u64::MAX.to_ne_bytes())
                .unwrap(),
        );

        // An unsigned index larger than i64::MAX selects the last valid slice, rather than wrapping negative.
        let sliced = domain
            .bind(
                Operation::Array(ArrayOperation::DynamicSlice(DynamicSliceOperation::new(vec![1]))),
                Vec::new(),
                &[input.clone(), index.clone()],
            )
            .unwrap();
        assert_eq!(read_f32s(&client, program_array(&sliced[0])), vec![3.0]);
        let updated = domain
            .bind(
                Operation::Array(ArrayOperation::DynamicUpdateSlice(DynamicUpdateSliceOperation::new())),
                Vec::new(),
                &[input, Value::Array(f32_vector(&domain, &mesh, &[9.0])), index],
            )
            .unwrap();
        assert_eq!(read_f32s(&client, program_array(&updated[0])), vec![1.0, 2.0, 9.0]);
    }

    #[test]
    fn test_eager_bind_executes_scan_with_per_step_outputs() {
        use ryft_core::ScanOperation;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);

        // Carry-only scan body `(index, carry) -> (carry + 1, carry + 1)`: the first output is the next carry and the
        // second is the per-step stacked output, so scanning 4 steps from `0` yields the cumulative sums `[1, 2, 3,
        // 4]`.
        let body = {
            let mut builder = ProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let carry = builder.add_input(scalar_type.clone().into());
            let one =
                builder.add_instruction(OneOperation::new(scalar_type.clone()), Vec::new(), vec![], None).unwrap()[0];
            let next = builder.add_instruction(AddOperation::new(), Vec::new(), vec![carry, one], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![next, next], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let scan = ScanOperation::new(1, 4);

        let outputs = domain
            .bind(Operation::Scan(scan), [body], &[Value::Array(f32_scalar(&domain, &mesh, 0.0))])
            .unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(read_f32s(&client, program_array(&outputs[0])), vec![4.0]);
        assert_eq!(program_array(&outputs[1]).shape(), StaticShape::new(vec![4]));
        assert_eq!(read_f32s(&client, program_array(&outputs[1])), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_eager_bind_executes_scan_with_explicit_slice_index() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let scalar_type = ArrayType::scalar(DataType::I64);
        let body = {
            let mut builder = ProgramBuilder::new();
            let index = builder.add_input(scalar_type.clone().into());
            builder.add_input(scalar_type.clone().into());
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![index, index], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };

        // The carry records the last visited index; stacked results retain slice order even when visits reverse.
        // Check the same contract for looped, partially unrolled (with and without remaining iterations after the
        // loop), and fully unrolled lowering.
        for (reverse, unroll, last_index) in [
            (false, 1, 3),
            (true, 1, 0),
            (false, 2, 3),
            (true, 2, 0),
            (false, 3, 3),
            (true, 3, 0),
            (false, 4, 3),
            (true, 4, 0),
            (false, 8, 3),
            (true, 8, 0),
            (false, usize::MAX, 3),
            (true, usize::MAX, 0),
        ] {
            let initial =
                Array::from_host_buffer(&domain, scalar_type.clone(), mesh.clone(), &(-1_i64).to_ne_bytes()).unwrap();
            let scan = ScanOperation::new(1, 4).with_reverse(reverse).with_unroll(unroll).unwrap();
            let outputs = domain.bind(Operation::Scan(scan), [body.clone()], &[Value::Array(initial)]).unwrap();
            assert_eq!(read_i64s(&client, program_array(&outputs[0])), vec![last_index]);
            assert_eq!(read_i64s(&client, program_array(&outputs[1])), vec![0, 1, 2, 3]);
        }
    }

    #[test]
    fn test_eager_bind_executes_scan_with_dynamic_scalar_ssa_length() {
        use ryft_core::ScanOperation;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let length_type = DimensionType::new("length", DimensionBounds::new(0, Some(9)).unwrap());
        let length = DimensionValue::new(length_type.clone(), 3).unwrap();

        let body = {
            let mut builder = ProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let carry = builder.add_input(scalar_type.clone().into());
            let one =
                builder.add_instruction(OneOperation::new(scalar_type.clone()), Vec::new(), vec![], None).unwrap()[0];
            let next = builder.add_instruction(AddOperation::new(), Vec::new(), vec![carry, one], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![next], vec![Placeholder, Placeholder], vec![Placeholder])
                .unwrap()
        };
        // An unroll factor that does not divide the runtime length runs the remaining iterations in a second loop, and
        // factors beyond the length's upper bound are clamped to it.
        for unroll in [1, 2, 4, usize::MAX] {
            let scan = ScanOperation::new(1, length_type.simplified_dimension()).with_unroll(unroll).unwrap();
            let outputs = domain
                .bind(
                    Operation::Scan(scan),
                    [body.clone()],
                    &[
                        Value::Array(f32_scalar(&domain, &mesh, 0.0)),
                        Value::Dimension(Dimension::new(length.clone(), domain.clone())),
                    ],
                )
                .unwrap();
            assert_eq!(outputs.len(), 1);
            assert_eq!(read_f32s(&client, program_array(&outputs[0])), vec![3.0]);
        }
    }

    #[test]
    fn test_eager_bind_executes_scan_with_stacked_inputs() {
        use ryft_core::ScanOperation;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let scalar_type = ArrayType::scalar(DataType::F32);

        // Cumulative-sum scan body `(index, carry, x) -> (carry + x, carry + x)` over the stacked input `[1, 2, 3, 4]`
        // starting from carry `0`: the final carry is the total `10` and the stacked per-step outputs are the
        // running sums `[1, 3, 6, 10]`. The body's metadata-free declared types are refined by the concrete input
        // types, which carry normalized shardings, so the scan binds eagerly despite the metadata mismatch.
        let body = {
            let mut builder = ProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let carry = builder.add_input(scalar_type.clone().into());
            let x = builder.add_input(scalar_type.clone().into());
            let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![carry, x], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![sum, sum], vec![Placeholder; 3], vec![Placeholder; 2])
                .unwrap()
        };
        let scan = ScanOperation::new(1, 4);

        let carry = f32_scalar(&domain, &mesh, 0.0);
        let xs = f32_vector(&domain, &mesh, &[1.0, 2.0, 3.0, 4.0]);
        let outputs = domain.bind(Operation::Scan(scan), vec![body], &[Value::Array(carry), Value::Array(xs)]).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(read_f32s(&client, program_array(&outputs[0])), vec![10.0]);
        assert_eq!(program_array(&outputs[1]).shape(), StaticShape::new(vec![4]));
        assert_eq!(read_f32s(&client, program_array(&outputs[1])), vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_eager_bind_executes_scan_with_sharded_stacked_inputs() {
        use ryft_core::ScanOperation;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 2);
        let domain = Session::new(&client).domain();
        let scalar_type = ArrayType::scalar(DataType::F32);

        // The same cumulative-sum scan as above, but with the stacked input sharded over the scanned (leading) axis
        // of a 2-device mesh: per-operation SPMD compilation handles the cross-shard slicing, and the inferred scan
        // output types leave shardings unspecified, so the outputs come back replicated over the mesh.
        let body = {
            let mut builder = ProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let carry = builder.add_input(scalar_type.clone().into());
            let x = builder.add_input(scalar_type.clone().into());
            let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![carry, x], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![sum, sum], vec![Placeholder; 3], vec![Placeholder; 2])
                .unwrap()
        };
        let scan = ScanOperation::new(1, 4);

        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let xs_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(sharding)
            .unwrap();
        let xs = Array::from_host_buffer(
            &domain,
            xs_type,
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let carry = f32_scalar(&domain, &mesh, 0.0);
        let outputs = domain.bind(Operation::Scan(scan), vec![body], &[Value::Array(carry), Value::Array(xs)]).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(read_f32s(&client, program_array(&outputs[0])), vec![10.0]);
        assert_eq!(program_array(&outputs[1]).shape(), StaticShape::new(vec![4]));
        assert_eq!(program_array(&outputs[1]).sharding(), &Sharding::replicated(mesh.logical_mesh().clone(), 1),);
        assert_eq!(read_f32s(&client, program_array(&outputs[1])), vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_eager_bind_executes_jit_call_and_reuses_cached_executable() {
        use std::sync::Arc;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let vector_type = replicated_vector_type(&mesh, 4);

        // A staged jitted callee `x -> x * x` bound eagerly on concrete arrays dispatches straight through the
        // compiled per-operation path, mirroring JAX calling a jitted function from eager code.
        let callee = {
            let mut builder = ProgramBuilder::new();
            let input = builder.add_input(vector_type.clone().into());
            let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };
        let operation = Operation::JitCall(JitCallOperation::new(0));
        let callee = Arc::new(callee);

        let input = f32_vector(&domain, &mesh, &[1.0, 2.0, 3.0, 4.0]);
        let first = domain
            .bind(operation.clone(), AttachedRegions::Callees(&[callee.clone()]), &[Value::Array(input.clone())])
            .unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(read_f32s(&client, program_array(&first[0])), vec![1.0, 4.0, 9.0, 16.0]);
        assert_eq!(domain.program_count(), 1);

        // A repeated eager `jit_call` at the same input signature is a dispatch-cache hit.
        let second = domain.bind(operation, AttachedRegions::Callees(&[callee]), &[Value::Array(input)]).unwrap();
        assert_eq!(read_f32s(&client, program_array(&second[0])), vec![1.0, 4.0, 9.0, 16.0]);
        assert_eq!(domain.program_count(), 1, "a repeated eager jit_call must be a compile-cache hit");
    }

    #[test]
    fn test_eager_bind_executes_shard_map_over_sharded_inputs() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .into_iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices).unwrap();
        let domain = Session::new(&client).domain();

        // Manual shard-map body `local -> local + local` over `f32[4]` sharded across the 2-device mesh: each device
        // doubles its own 2-element shard inside the manual region.
        let sharding = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding.clone()).unwrap();
        let local_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(sharding.local_sharding(&["x".to_string()]).unwrap())
            .unwrap();
        let body_program = {
            let mut builder = ProgramBuilder::new();
            let input = builder.add_input(local_type.clone().into());
            let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };
        let operation = ShardMapOperation::from_program(
            &body_program,
            vec![global_type.into()],
            ShardMapBoundary::new(logical_mesh, vec![sharding.clone()], vec![sharding.clone()], vec!["x".to_string()])
                .unwrap(),
        )
        .unwrap();
        let operation = Operation::ShardMap(Box::new(operation));
        let body_region = body_program;

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(sharding.clone())
            .unwrap();
        let input = Array::from_host_buffer(
            &domain,
            input_type,
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let outputs = domain.bind(operation, vec![body_region], &[Value::Array(input)]).unwrap();

        assert_eq!(outputs.len(), 1);
        let output = program_array(&outputs[0]);
        assert_eq!(output.sharding(), &sharding);
        assert_eq!(output.shards().len(), 2);
        let shard_values = output
            .addressable_shards()
            .map(|shard| {
                let bytes = shard.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap();
                values_from_bytes::<f32>(bytes.as_slice())
            })
            .collect::<Vec<_>>();
        assert_eq!(shard_values, vec![vec![2.0, 4.0], vec![6.0, 8.0]]);
    }

    #[test]
    fn test_eager_bind_rejects_collective_outside_a_mapping_context() {
        use ryft_core::{ParallelReduceOperation, ReductionKind};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = array_domain(&client);

        // A collective on a concrete array outside any `batch` / `shard_map` binder has no axis to resolve against,
        // mirroring JAX's "unbound axis name" error for a top-level `psum`. The value-level `ParallelReduce` capability
        // is not even implemented for `Array` (its domain carries no named-axis environment), so this binds the
        // operation directly and asserts the axis-resolution failure surfaced at compile time.
        let input = f32_vector(domain.parent(), &mesh, &[1.0, 2.0]);
        let operation = ParallelReduceOperation::new(ReductionKind::Sum, "i".to_string());
        assert!(matches!(
            domain.bind(operation, Vec::new(), &[input]),
            Err(ProgramError::InvalidArgument { message })
                if message == "collective over axis `i` can only be lowered inside a shard_map manual region",
        ));
    }

    #[test]
    fn test_eager_bind_surfaces_shape_mismatch_as_type_error() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = array_domain(&client);

        // Mismatched operand shapes fail at bind time through type inference on the traced single-instruction
        // program — never reaching PJRT compilation or execution.
        let left = f32_vector(domain.parent(), &mesh, &[1.0, 2.0]);
        let right = f32_vector(domain.parent(), &mesh, &[1.0, 2.0, 3.0]);
        assert!(matches!(
            domain.bind(AddOperation::new(), Vec::new(), &[left, right]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`add` input types are not broadcast-compatible",
        ));
    }
}
