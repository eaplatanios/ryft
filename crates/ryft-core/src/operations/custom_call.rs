use std::fmt::Display;
use std::marker::PhantomData;

use crate::arrays::{
    ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayType, Dimension, DimensionType, DimensionVariable,
};
use crate::batching::{BatchAxis, BatchableOperation, BatchingContext, BatchingDriver, BatchingError, BatchingPolicy};
use crate::contexts::{Context, Domain, EagerContext};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::dimensions::dimension_size::DimensionSize;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    Effect, Effects, Operation, OperationFormatter, ProgramError, RegionInterface, Type, TypeError,
    TypeIdentityRenaming, Typed, Value, ValueProjection,
};

/// Canonical operation name for [`CustomCallOperation`].
pub const CUSTOM_CALL_OPERATION_NAME: &str = "custom_call";

/// Typed configuration attribute value carried by a [`CustomCallOperation`] and forwarded to the foreign kernel.
/// The variants deliberately cover only the encodings every supporting backend must decode: strings, Booleans,
/// 64-bit signed integers, and 64-bit floating-point values. The `From` conversions let attribute values be passed
/// directly to [`CustomCallOperation::with_attribute`] (e.g., `.with_attribute("scale", 2.0)`); `&str` and `String`
/// convert into [`String`](Self::String), `bool` into [`Boolean`](Self::Boolean), `i64` into [`I64`](Self::I64),
/// and `f64` into [`F64`](Self::F64).
#[derive(Clone, Debug, PartialEq)]
pub enum CustomCallAttribute {
    /// UTF-8 string value.
    String(String),

    /// Boolean value.
    Boolean(bool),

    /// 64-bit signed-integer value.
    I64(i64),

    /// 64-bit floating-point value.
    F64(f64),
}

impl Display for CustomCallAttribute {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String(string) => formatter.write_str(string),
            Self::Boolean(boolean) => write!(formatter, "{boolean}"),
            Self::I64(integer) => write!(formatter, "{integer}"),
            Self::F64(float) => write!(formatter, "{float:?}"),
        }
    }
}

impl From<&str> for CustomCallAttribute {
    fn from(value: &str) -> Self {
        Self::String(value.to_string())
    }
}

impl From<String> for CustomCallAttribute {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<bool> for CustomCallAttribute {
    fn from(value: bool) -> Self {
        Self::Boolean(value)
    }
}

impl From<i64> for CustomCallAttribute {
    fn from(value: i64) -> Self {
        Self::I64(value)
    }
}

impl From<f64> for CustomCallAttribute {
    fn from(value: f64) -> Self {
        Self::F64(value)
    }
}

/// Declares that one flat array output of a [`CustomCallOperation`] aliases one flat array input.
///
/// Aliasing requires the input and output to describe the same logical array. It allows a backend to reuse the input
/// buffer for the output without changing Ryft's functional SSA semantics. The indices never include mixed trailing
/// dimension operands or backend-internal effect tokens.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct CustomCallInputOutputAlias {
    /// Index of the aliased array input.
    input_index: usize,

    /// Index of the array output that aliases the input.
    output_index: usize,
}

impl CustomCallInputOutputAlias {
    /// Creates an alias from the array input at `input_index` to the array output at `output_index`.
    #[inline]
    pub fn new(input_index: usize, output_index: usize) -> Self {
        Self { input_index, output_index }
    }

    /// Returns the index of the aliased array input.
    #[inline]
    pub fn input_index(&self) -> usize {
        self.input_index
    }

    /// Returns the index of the array output that aliases the input.
    #[inline]
    pub fn output_index(&self) -> usize {
        self.output_index
    }
}

impl Display for CustomCallInputOutputAlias {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}->{}", self.input_index, self.output_index)
    }
}

/// [`Operation`] that calls a foreign kernel registered with the executing backend under a target name — the
/// analogue of [`jax.ffi.ffi_call`](https://docs.jax.dev/en/latest/ffi.html). The operation is opaque to Ryft:
/// its output types are declared up front instead of inferred, and typed [`CustomCallAttribute`]s are forwarded
/// verbatim to the kernel as its configuration.
///
/// In the array IR, each dynamic axis occurrence in the declared outputs requires one trailing first-class
/// dimension operand, ordered first by output and then by axis. Type inference verifies that each operand defines the
/// exact variable referenced by its corresponding output axis. These logical result extents do not enter the foreign
/// kernel ABI: only the leading array operands are passed to the kernel. Eager execution and backend lowering use the
/// trailing operands to verify or attach the declared logical sizes to the returned buffers.
///
/// The type parameter selects that operand contract without introducing a second semantic operation:
///
///   - `CustomCallOperation<ArrayType>` accepts only the foreign kernel's array operands and is suitable for
///     programs over homogeneous arrays whose result extents are fully described by their array types.
///   - `CustomCallOperation<ArrayIrType>` additionally accepts the trailing first-class dimension operands
///     described above and is suitable for mixed array/dimension programs.
///
/// Both forms carry the same target, output declarations, attributes, effects, rendering, and backend-kernel
/// semantics. Conversion into the mixed form only reparameterizes the operation family; it moves the existing
/// descriptor without copying its owned metadata.
///
/// The XLA backend lowers this operation to a
/// [`stablehlo.custom_call`](https://openxla.org/stablehlo/spec#custom_call) using the typed FFI calling convention
/// (`api_version = 4`), with the attributes carried as the `backend_config` dictionary. Handlers are registered with
/// the executing PJRT client under the same target name (e.g., via `ryft-pjrt`'s `Client::register_ffi_handler`).
/// The reference array backend cannot execute foreign kernels, so eager interpretation on it reports an error.
///
/// Because the kernel is opaque, Ryft cannot derive its transform rules. Differentiating it reports an error
/// directing users to wrap the call with [`custom_jvp`](crate::tracing_v2::CustomJvp) or
/// [`custom_vjp`](crate::tracing_v2::CustomVjp), which supply the missing derivative. Those wrappers do *not* supply
/// a batching rule: each of them structurally batches its own primal region, so a mapped operand reaches this same
/// operation and meets this same batching contract. Batching a call whose operands are all replicated binds it
/// unchanged, because a region-free foreign kernel cannot observe the transform's named axis. A mapped operand
/// instead reports an error naming that operand, and the remedies are to invoke a kernel that already understands
/// the batch axis or to select an explicit batching behavior for this call.
/// Marking the call as side-effecting via [`with_side_effect`](Self::with_side_effect)
/// reports [`Effect::OrderedIo`], which keeps the call alive through dead-code elimination and preserves its
/// execution order relative to other ordered effects; the lowered custom call is then also marked
/// `has_side_effect = true` so the XLA compiler never elides or reorders it.
///
/// # Backend Contract
///
/// This operation is backend-independent by design, and this payload is the entire portable contract: a target
/// name resolved in the executing backend's kernel registry, declared output types, typed configuration
/// attributes, and an effect flag. A backend supports the operation by providing (1) a process- or client-level
/// registry that resolves target names to executable kernels at execution time, (2) a calling convention that
/// hands the kernel its input buffers, output buffers matching the declared output types, and the decoded
/// attributes, and (3) an execution engine that honors [`Effect::OrderedIo`] for side-effecting calls. Backends
/// that cannot execute foreign kernels (like the reference array backend) reject interpretation with a clear
/// error instead of guessing.
///
/// Array layouts come from the canonical [`ArrayType`] descriptors of the operands and results. Portable flat-array
/// buffer aliases are declared with [`CustomCallInputOutputAlias`]. Backend-specific vocabulary must never grow on
/// this payload: encodings such as XLA's FFI API version, `backend_config` representation, tuple alias paths, result
/// tiling attributes, or called-computation references belong in the owning backend's lowering (or in a backend-owned
/// operation). If a configuration knob only makes sense for one backend, it does not belong on this operation.
#[derive(Clone, Debug)]
pub struct CustomCallOperation<T: Type> {
    /// Name under which the foreign kernel is registered with the executing backend.
    target_name: String,

    /// Declared output types of the call, returned verbatim by type inference.
    output_types: Vec<ArrayType>,

    /// Typed configuration attributes forwarded to the kernel, in insertion order.
    attributes: Vec<(String, CustomCallAttribute)>,

    /// Flat array input/output buffer aliases, in declaration order.
    input_output_aliases: Vec<CustomCallInputOutputAlias>,

    /// Whether the call has observable side effects beyond its returned outputs.
    has_side_effect: bool,

    /// Type universe that determines the operation's operand contract.
    marker: PhantomData<fn() -> T>,
}

impl CustomCallOperation<ArrayType> {
    /// Creates a new [`CustomCallOperation`] with the provided target name and declared output types.
    ///
    /// # Parameters
    ///
    ///   - `target_name`: Name under which the foreign kernel is registered with the executing backend.
    ///   - `output_types`: Declared output types of the call, returned verbatim by type inference.
    #[inline]
    pub fn new<N: Into<String>>(target_name: N, output_types: Vec<ArrayType>) -> Self {
        Self {
            target_name: target_name.into(),
            output_types,
            attributes: Vec::new(),
            input_output_aliases: Vec::new(),
            has_side_effect: false,
            marker: PhantomData,
        }
    }
}

impl<T: Type> CustomCallOperation<T> {
    /// Returns a copy of this [`CustomCallOperation`] with the provided typed configuration attribute appended.
    #[inline]
    pub fn with_attribute<N: Into<String>, V: Into<CustomCallAttribute>>(mut self, name: N, value: V) -> Self {
        self.attributes.push((name.into(), value.into()));
        self
    }

    /// Returns this operation with an alias from array input `input_index` to array output `output_index`.
    ///
    /// Index bounds and type compatibility are validated during type inference, when the input types are available.
    /// Each input and output can participate in at most one alias.
    pub fn with_input_output_alias(mut self, input_index: usize, output_index: usize) -> Result<Self, TypeError> {
        if let Some(alias) = self
            .input_output_aliases
            .iter()
            .find(|alias| alias.input_index == input_index || alias.output_index == output_index)
        {
            return Err(TypeError::invalid(format!(
                "'{CUSTOM_CALL_OPERATION_NAME}' cannot add alias {input_index}->{output_index} because alias \
                 '{alias}' already uses the same input or output",
            )));
        }
        self.input_output_aliases.push(CustomCallInputOutputAlias::new(input_index, output_index));
        Ok(self)
    }

    /// Returns a copy of this [`CustomCallOperation`] marked as having observable side effects.
    #[inline]
    pub fn with_side_effect(mut self) -> Self {
        self.has_side_effect = true;
        self
    }

    /// Returns the name under which the foreign kernel is registered with the executing backend.
    #[inline]
    pub fn target_name(&self) -> &str {
        self.target_name.as_str()
    }

    /// Returns the declared output types of the call.
    #[inline]
    pub fn output_types(&self) -> &[ArrayType] {
        self.output_types.as_slice()
    }

    /// Returns the typed configuration attributes forwarded to the kernel, in insertion order.
    #[inline]
    pub fn attributes(&self) -> &[(String, CustomCallAttribute)] {
        self.attributes.as_slice()
    }

    /// Returns the flat array input/output buffer aliases in declaration order.
    #[inline]
    pub fn input_output_aliases(&self) -> &[CustomCallInputOutputAlias] {
        self.input_output_aliases.as_slice()
    }

    /// Returns whether the call has observable side effects beyond its returned outputs.
    #[inline]
    pub fn has_side_effect(&self) -> bool {
        self.has_side_effect
    }

    /// Returns this payload with every declared output identity renamed according to `renaming`.
    fn renamed(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<Self, TypeError> {
        Ok(Self {
            target_name: self.target_name.clone(),
            output_types: self
                .output_types
                .iter()
                .map(|r#type| r#type.rename_identities(renaming))
                .collect::<Result<Vec<_>, _>>()?,
            attributes: self.attributes.clone(),
            input_output_aliases: self.input_output_aliases.clone(),
            has_side_effect: self.has_side_effect,
            marker: PhantomData,
        })
    }

    /// Returns the [`BatchingError`] reported when operand `index` carries the mapped `batch_axis` and this call has
    /// no way to thread that axis through its opaque kernel.
    fn mapped_operand_error(&self, index: usize, batch_axis: BatchAxis) -> BatchingError {
        BatchingError::UnsupportedOperation {
            message: format!(
                "custom call '{}' has no batching rule for operand {index} mapped at batch axis {}; invoke a kernel \
                 that understands the batch axis, or select an explicit batching behavior with \
                 `CustomCallOperation::with_batching`",
                self.target_name,
                batch_axis.axis().map(|axis| axis.to_string()).unwrap_or_else(|| "replicated".to_string()),
            ),
        }
    }

    /// Renders this payload independently of its homogeneous or composite operation contract.
    fn render_operation(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, CUSTOM_CALL_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("target", &self.target_name)?;
            for (name, value) in &self.attributes {
                operation.field(name, value)?;
            }
            for alias in &self.input_output_aliases {
                operation.field("input_output_alias", alias)?;
            }
            if self.has_side_effect {
                operation.field("has_side_effect", true)?;
            }
            Ok(())
        })
    }
}

impl From<CustomCallOperation<ArrayType>> for CustomCallOperation<ArrayIrType> {
    fn from(operation: CustomCallOperation<ArrayType>) -> Self {
        Self {
            target_name: operation.target_name,
            output_types: operation.output_types,
            attributes: operation.attributes,
            input_output_aliases: operation.input_output_aliases,
            has_side_effect: operation.has_side_effect,
            marker: PhantomData,
        }
    }
}

impl From<CustomCallOperation<ArrayIrType>> for CustomCallOperation<ArrayType> {
    fn from(operation: CustomCallOperation<ArrayIrType>) -> Self {
        Self {
            target_name: operation.target_name,
            output_types: operation.output_types,
            attributes: operation.attributes,
            input_output_aliases: operation.input_output_aliases,
            has_side_effect: operation.has_side_effect,
            marker: PhantomData,
        }
    }
}

impl<T: Type> Display for CustomCallOperation<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render_operation(formatter, 0)
    }
}

impl<T: Type> CustomCallOperation<T> {
    /// Validates flat input/output aliases against the array operands of this operation.
    fn validate_input_output_aliases(&self, input_types: &[&ArrayType]) -> Result<(), TypeError> {
        for alias in &self.input_output_aliases {
            let Some(input_type) = input_types.get(alias.input_index) else {
                return Err(TypeError::invalid(format!(
                    "'{CUSTOM_CALL_OPERATION_NAME}' alias '{alias}' refers to input {} but the call has {} array \
                     inputs",
                    alias.input_index,
                    input_types.len(),
                )));
            };
            let Some(output_type) = self.output_types.get(alias.output_index) else {
                return Err(TypeError::invalid(format!(
                    "'{CUSTOM_CALL_OPERATION_NAME}' alias '{alias}' refers to output {} but the call has {} outputs",
                    alias.output_index,
                    self.output_types.len(),
                )));
            };
            if *input_type != output_type {
                return Err(TypeError::invalid(format!(
                    "'{CUSTOM_CALL_OPERATION_NAME}' alias '{alias}' requires matching input and output types but \
                     input {} has type '{}' and output {} has type '{}'",
                    alias.input_index, input_type, alias.output_index, output_type,
                )));
            }
        }
        Ok(())
    }
}

impl Operation for CustomCallOperation<ArrayType> {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        CUSTOM_CALL_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("region", region_interfaces, 0, TypeError);
        self.validate_input_output_aliases(&input_types.iter().collect::<Vec<_>>())?;
        Ok(self.output_types.clone())
    }

    #[inline]
    fn effects(&self) -> Effects {
        if self.has_side_effect { Effects::single(Effect::OrderedIo) } else { Effects::PURE }
    }

    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<ArrayType as crate::Type>::Identity>,
    ) -> Result<Self, TypeError> {
        self.renamed(renaming)
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        self.render_operation(formatter, indentation)
    }
}

impl Operation for CustomCallOperation<ArrayIrType> {
    type Type = ArrayIrType;

    #[inline]
    fn name(&self) -> &'static str {
        CUSTOM_CALL_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("region", region_interfaces, 0, TypeError);
        let dynamic_output_dimensions = self
            .output_types
            .iter()
            .flat_map(|output_type| output_type.shape().dimensions())
            .filter_map(Dimension::variable)
            .collect::<Vec<_>>();
        let Some(array_input_count) = input_types.len().checked_sub(dynamic_output_dimensions.len()) else {
            return Err(TypeError::invalid(format!(
                "'{CUSTOM_CALL_OPERATION_NAME}' expects {} trailing output-extent dimensions but only {} inputs were \
                 provided",
                dynamic_output_dimensions.len(),
                input_types.len(),
            )));
        };
        let array_input_types =
            input_types[..array_input_count].iter().map(<&ArrayType>::try_from).collect::<Result<Vec<_>, _>>()?;
        self.validate_input_output_aliases(array_input_types.as_slice())?;
        for (input_type, expected_variable) in input_types[array_input_count..].iter().zip(dynamic_output_dimensions) {
            let actual_variable = <&crate::arrays::DimensionType>::try_from(input_type)?.variable();
            if actual_variable != expected_variable {
                return Err(TypeError::invalid(format!(
                    "'{CUSTOM_CALL_OPERATION_NAME}' output-extent operand defines dimension variable \
                     '{actual_variable}', but the corresponding declared output axis refers to \
                     '{expected_variable}'",
                )));
            }
        }
        Ok(self.output_types.iter().cloned().map(Into::into).collect())
    }

    #[inline]
    fn effects(&self) -> Effects {
        if self.has_side_effect { Effects::single(Effect::OrderedIo) } else { Effects::PURE }
    }

    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<Self, TypeError> {
        self.renamed(renaming)
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        self.render_operation(formatter, indentation)
    }
}

impl<C: Domain<Type = ArrayType, Value: CustomCall>> InterpretableOperation<C> for CustomCallOperation<ArrayType> {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        C::Value::custom_call(self, inputs)
    }
}

impl<A: CustomCall + DimensionSize<usize> + Value<Type = ArrayType>>
    InterpretableOperation<EagerContext<ArrayIrValue<A>, ArrayIrOperation<A>>> for CustomCallOperation<ArrayIrType>
{
    fn interpret<D: InterpretationDriver<EagerContext<ArrayIrValue<A>, ArrayIrOperation<A>>>>(
        &self,
        _context: &EagerContext<ArrayIrValue<A>, ArrayIrOperation<A>>,
        driver: &D,
        inputs: &[ArrayIrValue<A>],
    ) -> Result<Vec<ArrayIrValue<A>>, ProgramError> {
        if driver.region_count() != 0 {
            return Err(TypeError::invalid(format!("expected 0 regions but got {}", driver.region_count())).into());
        }
        self.infer_output_types(&inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(), &[])?;
        let dynamic_output_dimension_count = self
            .output_types
            .iter()
            .flat_map(|output_type| output_type.shape().dimensions())
            .filter(|dimension| matches!(dimension, Dimension::Dynamic(_)))
            .count();
        let array_input_count = inputs.len() - dynamic_output_dimension_count;
        let array_inputs = inputs[..array_input_count]
            .iter()
            .map(<ArrayIrValue<A> as ValueProjection<ArrayType>>::projected)
            .collect::<Result<Vec<_>, _>>()?;
        let output_extents = inputs[array_input_count..]
            .iter()
            .map(<ArrayIrValue<A> as ValueProjection<DimensionType>>::projected)
            .collect::<Result<Vec<_>, _>>()?;
        let kernel_operation = CustomCallOperation::<ArrayType>::from(self.clone());
        let outputs = A::custom_call(&kernel_operation, array_inputs.iter().copied())?;
        check_count!("output", outputs, self.output_types.len(), ProgramError);
        let mut output_extents = output_extents.into_iter();
        for (output_index, (output, output_type)) in outputs.iter().zip(&self.output_types).enumerate() {
            for (axis, dimension) in output_type.shape().dimensions().iter().enumerate() {
                if matches!(dimension, Dimension::Dynamic(_)) {
                    let expected_extent = output_extents.next().unwrap().extent();
                    let actual_extent = output.dimension_size(axis)?;
                    if actual_extent != expected_extent {
                        return Err(ProgramError::InvalidArgument {
                            message: format!(
                                "'{CUSTOM_CALL_OPERATION_NAME}' output {output_index} axis {axis} has extent \
                                 {actual_extent}, but its explicit extent operand is {expected_extent}",
                            ),
                        });
                    }
                }
            }
        }
        Ok(outputs.into_iter().map(ArrayIrValue::Array).collect())
    }
}

// Partial evaluation defers to the default fold-or-residualize behavior of `Program::partially_evaluate`. An all-known
// custom call folds into the known side (executing there only if the known-side context can run foreign kernels), and a
// side-effecting residual call survives dead-code elimination because `Operation::effects` is not `Effects::PURE`.
impl<T: Type, C: Context<Type = T, Operation: From<CustomCallOperation<T>>>> PartiallyEvaluatableOperation<C>
    for CustomCallOperation<T>
where
    CustomCallOperation<T>: Operation<Type = T>,
{
}

impl_differentiable_operation! {
    <T> CustomCallOperation<T>,
    jvp<C>
    where
        T: Type,
    {
        |operation, _context, _driver, _inputs| {
            // Foreign kernels are opaque, so there is no derivative to derive: differentiation reports an error
            // directing users to wrap the call with [`custom_jvp`](crate::tracing_v2::CustomJvp) or
            // [`custom_vjp`](crate::tracing_v2::CustomVjp), which is also how JAX handles `ffi_call` differentiation.
            Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "custom call '{}' has no differentiation rule; wrap it with `custom_jvp` or `custom_vjp` to \
                     provide one",
                    operation.target_name,
                ),
            }
            .into())
        }
    },
    transpose = @nonlinear,
}

/// Batching rule for [`CustomCallOperation`]. A foreign kernel is opaque, so Ryft cannot derive how a batch axis
/// threads through it. A call whose operands are *all replicated* is nevertheless bound unchanged through the parent
/// context and reports replicated outputs, matching JAX, which only invokes a batching rule once some operand is
/// actually mapped. Any mapped operand reports a [`BatchingError::UnsupportedOperation`] that names the operand and
/// its mapped axis.
///
/// The all-replicated shortcut is sound *for this operation specifically* because a custom call is region-free by
/// construction: [`Operation::infer_output_types`] rejects every attached region, so the kernel is a leaf whose only
/// observable inputs are its operands. A foreign kernel therefore cannot observe the transform's named axis, and
/// running it unchanged over replicated operands computes exactly what each batch item would have computed on its
/// own. The shortcut must never be generalized to region-carrying operations. A region can contain a named-axis
/// operation whose value differs per batch item even when every operand of the enclosing instruction is replicated;
/// `.tasks/plan_custom_derivative_batching_axis_parity.md` records the pinning JAX fixture for that counterexample
/// (`vmap` with `in_axes=None`, an explicit extent, and a named-axis index still produces `[0, 1, 2]`), which is why
/// the custom-derivative wrappers always batch their regions structurally.
impl<C: Context, P: BatchingPolicy<C>> BatchableOperation<C, P> for CustomCallOperation<C::Type>
where
    C::Operation: From<CustomCallOperation<C::Type>>,
    CustomCallOperation<C::Type>: Operation<Type = C::Type>,
{
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        _driver: &D,
        inputs: &[P::Batch],
    ) -> Result<Vec<P::Batch>, BatchingError> {
        if let Some((index, batch)) = inputs.iter().enumerate().find(|(_, input)| !P::batch_axis(input).is_replicated())
        {
            return Err(self.mapped_operand_error(index, P::batch_axis(batch)));
        }
        let inputs = inputs.iter().map(P::value).cloned().collect::<Vec<_>>();
        let outputs = context.parent().bind(self.clone(), Vec::new(), inputs.as_slice())?;
        Ok(outputs.into_iter().map(P::replicated).collect())
    }
}

/// Represents the ability to call foreign kernels registered with the executing backend. [`CustomCall`] stages or
/// executes a [`CustomCallOperation`]; refer to its documentation for the calling convention and the transform
/// rules. The capability method dispatches through the first input's context, so it needs at least one input
/// (zero-input custom calls can still be staged directly through a program builder).
pub trait CustomCall: Sized {
    /// Calls the foreign kernel described by `operation` with the provided inputs, returning one value per
    /// declared output type, and a [`ProgramError`] if something goes wrong.
    fn custom_call<'a, I: IntoIterator<Item = &'a Self>>(
        operation: &CustomCallOperation<ArrayType>,
        inputs: I,
    ) -> Result<Vec<Self>, ProgramError>
    where
        Self: 'a;
}

/// Any context-carrying value calls foreign kernels by binding a [`CustomCallOperation<ArrayType>`] through its own
/// context. The conversion bound makes this disjoint from the eager reference value types (whose context operation is
/// [`ConstantOperation`](crate::operations::constants::ConstantOperation)), so it covers the transform tracers and
/// backend-owned values without conflicting with concrete implementations.
impl<V: Value<Type = ArrayType>> CustomCall for V
where
    V::DispatchDomain: Context<Operation: From<CustomCallOperation<ArrayType>>>,
{
    fn custom_call<'a, I: IntoIterator<Item = &'a Self>>(
        operation: &CustomCallOperation<ArrayType>,
        inputs: I,
    ) -> Result<Vec<Self>, ProgramError>
    where
        Self: 'a,
    {
        let inputs = inputs.into_iter().cloned().collect::<Vec<_>>();
        let Some(first) = inputs.first() else {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "the custom-call capability dispatches through its first input's context, so calling '{}' with \
                     no inputs requires staging the operation through a program builder instead",
                    operation.target_name(),
                ),
            });
        };
        first.dispatch_domain().bind(operation.clone(), Vec::new(), inputs.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrBatch, ArrayIrBatching, ArrayOperation, DataType, Dimension, DimensionBounds, DimensionType,
        DimensionValue, DimensionVariable, Shape, ShardingDimension,
    };
    use crate::batching::{
        BatchAxis, BatchedProgram, BatchingContext, BatchingTracer, ProgramBatchingOutputAxesPolicy,
    };
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{DifferentiationError, TransposableOperation};
    use crate::interpretation::InterpretableOperation;
    use crate::parameters::Placeholder;
    use crate::programs::{EmptyRegionDriver, ProgramBuilder};
    use crate::tracing::{DomainTracer, Trace, TracingContext};

    use super::*;

    /// Returns the `f32[2]` array type used throughout these tests.
    fn vector_type() -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]))
    }

    #[test]
    fn test_custom_call_operation_contract() {
        let operation = CustomCallOperation::new("ryft.test.add_one", vec![vector_type()])
            .with_attribute("scale", 2.0)
            .with_attribute("count", 4i64)
            .with_attribute("verbose", true)
            .with_attribute("label", "x")
            .with_input_output_alias(0, 0)
            .unwrap()
            .with_side_effect();

        assert_eq!(operation.target_name(), "ryft.test.add_one");
        assert_eq!(operation.output_types(), &[vector_type()]);
        assert_eq!(
            operation.attributes(),
            &[
                ("scale".to_string(), CustomCallAttribute::F64(2.0)),
                ("count".to_string(), CustomCallAttribute::I64(4)),
                ("verbose".to_string(), CustomCallAttribute::Boolean(true)),
                ("label".to_string(), CustomCallAttribute::String("x".to_string())),
            ],
        );
        assert_eq!(operation.input_output_aliases(), &[CustomCallInputOutputAlias::new(0, 0)]);
        assert!(operation.has_side_effect());
        assert_eq!(operation.name(), CUSTOM_CALL_OPERATION_NAME);
        assert_eq!(operation.effects(), Effects::single(Effect::OrderedIo));
        assert_eq!(operation.infer_output_types(&[vector_type()], &[]), Ok(vec![vector_type()]),);
        // Long attribute lists wrap onto one line per field.
        assert_eq!(
            operation.to_string(),
            indoc! {"
                custom_call [
                    target=ryft.test.add_one,
                    scale=2.0,
                    count=4,
                    verbose=true,
                    label=x,
                    input_output_alias=0->0,
                    has_side_effect=true,
                ]
            "}
            .trim_end(),
        );

        let pure = CustomCallOperation::new("ryft.test.add_one", vec![vector_type()]);
        assert_eq!(pure.effects(), Effects::PURE);
        assert_eq!(pure.to_string(), "custom_call [target=ryft.test.add_one]");
        let roundtrip =
            CustomCallOperation::<ArrayType>::from(CustomCallOperation::<ArrayIrType>::from(operation.clone()));
        assert_eq!(roundtrip.input_output_aliases(), operation.input_output_aliases());
        assert!(matches!(
            operation.clone().with_input_output_alias(0, 1),
            Err(TypeError::Invalid { message })
                if message == "'custom_call' cannot add alias 0->1 because alias '0->0' already uses the same input \
                               or output",
        ));
        assert!(matches!(
            operation.clone().with_input_output_alias(1, 0),
            Err(TypeError::Invalid { message })
                if message == "'custom_call' cannot add alias 1->0 because alias '0->0' already uses the same input \
                               or output",
        ));
        assert_eq!(
            CustomCallOperation::new("ryft.test.add_one", vec![vector_type()])
                .with_input_output_alias(1, 0)
                .unwrap()
                .infer_output_types(&[vector_type()], &[]),
            Err(TypeError::invalid("'custom_call' alias '1->0' refers to input 1 but the call has 1 array inputs",)),
        );
        assert_eq!(
            CustomCallOperation::new("ryft.test.add_one", vec![vector_type()])
                .with_input_output_alias(0, 1)
                .unwrap()
                .infer_output_types(&[vector_type()], &[]),
            Err(TypeError::invalid("'custom_call' alias '0->1' refers to output 1 but the call has 1 outputs",)),
        );
        assert_eq!(
            CustomCallOperation::new("ryft.test.add_one", vec![ArrayType::scalar(DataType::F32)])
                .with_input_output_alias(0, 0)
                .unwrap()
                .infer_output_types(&[vector_type()], &[]),
            Err(TypeError::invalid(
                "'custom_call' alias '0->0' requires matching input and output types but input 0 has type 'f32[2]' \
                 and output 0 has type 'f32[]'",
            )),
        );

        // Composite output extents are positional SSA operands: output-major and then axis-major.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        let columns = DimensionVariable::new("columns", DimensionBounds::new(2, Some(17)).unwrap());
        let dynamic_output_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![
                Dimension::Dynamic(rows.clone()),
                Dimension::Static(3),
                Dimension::Dynamic(columns.clone()),
            ]),
        );
        let dynamic_operation = CustomCallOperation::<ArrayIrType>::from(CustomCallOperation::new(
            "ryft.test.dynamic",
            vec![dynamic_output_type.clone()],
        ));
        let input_types = vec![
            vector_type().into(),
            DimensionType::new(rows.clone()).into(),
            DimensionType::new(columns.clone()).into(),
        ];
        let aliased_dynamic_operation = CustomCallOperation::<ArrayIrType>::from(
            CustomCallOperation::new("ryft.test.dynamic", vec![dynamic_output_type.clone()])
                .with_input_output_alias(0, 0)
                .unwrap(),
        );
        assert_eq!(
            aliased_dynamic_operation.infer_output_types(
                &[
                    dynamic_output_type.clone().into(),
                    DimensionType::new(rows.clone()).into(),
                    DimensionType::new(columns.clone()).into(),
                ],
                &[],
            ),
            Ok(vec![dynamic_output_type.clone().into()]),
        );
        assert_eq!(dynamic_operation.infer_output_types(&input_types, &[]), Ok(vec![dynamic_output_type.into()]));
        assert_eq!(
            dynamic_operation.infer_output_types(
                &[vector_type().into(), DimensionType::new(columns).into(), DimensionType::new(rows).into()],
                &[],
            ),
            Err(TypeError::invalid(
                "'custom_call' output-extent operand defines dimension variable 'columns', but the corresponding \
                 declared output axis refers to 'rows'",
            )),
        );
        assert_eq!(
            dynamic_operation.infer_output_types(
                &[DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap(),))
                    .into()],
                &[],
            ),
            Err(TypeError::invalid(
                "'custom_call' expects 2 trailing output-extent dimensions but only 1 inputs were provided",
            )),
        );
    }

    #[test]
    fn test_custom_call_stages_through_the_tracer_capability() {
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| {
                let operation = CustomCallOperation::new("ryft.test.add_one", vec![vector_type()]);
                Ok(CustomCall::custom_call(&operation, std::slice::from_ref(&x))?.remove(0))
            },
            vector_type(),
        )
        .unwrap();
        let program = program.to_flat_program();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2] = custom_call [target=ryft.test.add_one] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_call_is_rejected_by_the_reference_backend() {
        let operation = CustomCallOperation::new("ryft.test.add_one", vec![vector_type()]);
        let result = InterpretableOperation::<EagerContext<Array>>::interpret(
            &operation,
            &EagerContext::new(),
            &EmptyRegionDriver,
            &[Array::vector(vec![1.0, 2.0])],
        );
        assert!(matches!(
            result,
            Err(ProgramError::UnsupportedOperation { message })
                if message == "the reference array backend cannot execute the foreign kernel 'ryft.test.add_one'",
        ));
    }

    #[test]
    fn test_custom_call_rejects_differentiation_and_batching() {
        let operation = CustomCallOperation::new("ryft.test.add_one", vec![vector_type()]);

        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(vector_type());
        let output = builder.add_instruction(operation.clone(), Vec::new(), vec![input]).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        assert!(matches!(
            program.jvp(),
            Err(error)
                if error.to_string().contains("custom call 'ryft.test.add_one' has no differentiation rule"),
        ));
        assert!(matches!(
            program.batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            ),
            Err(error)
                if error.to_string()
                    == "custom call 'ryft.test.add_one' has no batching rule for operand 0 mapped at batch axis 0; \
                        invoke a kernel that understands the batch axis, or select an explicit batching behavior \
                        with `CustomCallOperation::with_batching`",
        ));
    }

    /// A custom call whose operands are all replicated is bound unchanged and reports replicated outputs. This is the
    /// JAX-parity behavior: a batching rule is consulted only once an operand is actually mapped, and this shortcut is
    /// sound because the region-free foreign kernel cannot observe the transform's axis.
    #[test]
    fn test_custom_call_batches_all_replicated_operands_unchanged() {
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>| {
                let operation = CustomCallOperation::new("ryft.test.add_one", vec![vector_type()]);
                Ok(vec![CustomCall::custom_call(&operation, inputs.iter())?.remove(0)])
            },
            vec![vector_type(), vector_type()],
        )
        .unwrap();

        let (batched, output_axes) = program
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::replicated(), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::replicated()]);
        let batched = batched.to_flat_program();
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[2] .
                let %2:f32[2] = custom_call [target=ryft.test.add_one] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    /// The mixed universe applies the same all-replicated shortcut, including the trailing first-class output-extent
    /// operand, which stays an ordinary replicated operand of the unchanged call.
    #[test]
    fn test_array_ir_custom_call_batches_all_replicated_operands_unchanged() -> Result<(), ProgramError> {
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows.clone())]));
        let operation = CustomCallOperation::<ArrayIrType>::from(CustomCallOperation::new(
            "ryft.test.dynamic",
            vec![output_type.clone()],
        ));

        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = trace.input(vector_type().into());
        let extent = trace.input(DimensionType::new(rows).into());
        let context = BatchingContext::<_, ArrayIrBatching>::new(
            trace.clone(),
            trace.constant(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap())),
        );
        let inputs = [
            BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(input)),
            BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(extent)),
        ];
        let [output] = context.bind(operation, Vec::new(), &inputs)?.try_into().unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::replicated());
        assert_eq!(output.r#type().as_ref(), &ArrayIrType::Array(output_type));

        let output_id = output.into_batch().into_value().atom_id().unwrap();
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![output_id],
            vec![Placeholder, Placeholder],
            vec![Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:dimension<rows ∈ [1, 9)> .
                let %2:dimension<2> = const
                    %3:f32[rows] = custom_call [target=ryft.test.dynamic] %0 %1
                in (%3)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_array_ir_custom_call_rejects_transforms() {
        let operation = CustomCallOperation::<ArrayIrType>::from(CustomCallOperation::new(
            "ryft.test.add_one",
            vec![vector_type()],
        ));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(vector_type().into());
        let output = builder.add_instruction(operation.clone(), Vec::new(), vec![input]).unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert!(matches!(
            program.jvp(),
            Err(error)
                if error.to_string()
                    == "custom call 'ryft.test.add_one' has no differentiation rule; wrap it with `custom_jvp` or \
                        `custom_vjp` to provide one",
        ));

        let batching_context = BatchingContext::<_, ArrayIrBatching>::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let mapped = ArrayIrBatch::new(
            ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0])),
            BatchAxis::new(0),
        )
        .unwrap();
        assert!(matches!(
            operation.batch(&batching_context, &EmptyRegionDriver, &[mapped]),
            Err(BatchingError::UnsupportedOperation { message })
                if message
                    == "custom call 'ryft.test.add_one' has no batching rule for operand 0 mapped at batch axis 0; \
                        invoke a kernel that understands the batch axis, or select an explicit batching behavior \
                        with `CustomCallOperation::with_batching`",
        ));

        let mut transposition_context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        assert!(matches!(
            operation.transpose(&mut transposition_context, &EmptyRegionDriver, &[], &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "operation `custom_call` is not transposable",
        ));
    }
}
