use std::marker::PhantomData;

use ryft_macros::Parameterized;

use crate::contexts::{Context, StagingContext};
use crate::differentiation::forward::{DifferentiableOperation, ForwardModeDifferentiate, LinearizationTracer};
use crate::differentiation::reverse::{ReverseModeDifferentiate, TransposableOperation, TransposedProgramCotangent};
use crate::differentiation::types::{DenseDifferentiableType, DifferentiableType};
use crate::differentiation::zeros::ResidualZeroProvider;
use crate::differentiation::{DerivativeTransform, DifferentiationError, DifferentiationParameterRole};
use crate::macros::check_count;
use crate::operations::{AddOperation, ReferenceAddUpdateOperation, ReferenceNewOperation, ZeroOperation};
use crate::parameters::{Parameter, ParameterPath, Parameterized, ParameterizedFamily};
use crate::partial::{PartialEvaluationContext, PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    OperationProvider, ProgramError, ReferenceAccessOperation, ReferenceMemberType, ReferenceTransform, Type, Typed,
};
use crate::tracing::TracingContext;

/// Jacobian of a function, represented as the Cartesian product of its output and input [`Parameter`] leaves. `I` and
/// `O` retain the input and output [`Type`] trees. Derivative values are stored in deterministic output-major /
/// input-minor order and remain [`Parameter`]s so that the complete Jacobian can cross tracing and compilation
/// boundaries as well as participate in higher-order transforms. The physical representation of a block is defined by
/// [`DenseDifferentiableType`]. For [`ArrayType`](crate::ArrayType), the block for an output leaf with shape `O` and
/// an input leaf with shape `I` has shape `O` concatenated with `I`.
#[derive(Clone, Debug, Parameterized)]
pub struct Jacobian<T: Type, V: Parameter, I: Parameterized<T>, O: Parameterized<T>> {
    /// [`Type`] of the differentiated inputs.
    input_type: I,

    /// [`Type`] of the differentiated outputs.
    output_type: O,

    /// Derivative values in output-major/input-minor order.
    values: Vec<V>,

    /// [`PhantomData`] marker for `T`, needed because the input and output fields use `T` only through their bounds.
    _type: PhantomData<fn() -> T>,
}

impl<T: Type, V: Parameter, I: Parameterized<T>, O: Parameterized<T>> Jacobian<T, V, I, O> {
    /// Creates a new [`Jacobian`].
    pub fn new(input_type: I, output_type: O, values: Vec<V>) -> Result<Self, ProgramError> {
        let input_count = input_type.parameter_count();
        let output_count = output_type.parameter_count();
        let expected_count = input_count.checked_mul(output_count).ok_or_else(|| ProgramError::InvalidArgument {
            message: format!("Jacobian block count ({input_count} x {output_count}) overflows usize"),
        })?;
        if values.len() != expected_count {
            return Err(ProgramError::InvalidArgument {
                message: format!("Jacobian requires {} derivative values but got {}", expected_count, values.len()),
            });
        }
        Ok(Self { input_type, output_type, values, _type: PhantomData })
    }

    /// Returns the [`Type`] of the differentiated inputs.
    #[inline]
    pub fn input_type(&self) -> &I {
        &self.input_type
    }

    /// Returns the [`Type`] of the differentiated outputs.
    #[inline]
    pub fn output_type(&self) -> &O {
        &self.output_type
    }

    /// Returns the derivative values in output-major/input-minor order.
    #[inline]
    pub fn values(&self) -> &[V] {
        self.values.as_slice()
    }

    /// Consumes this [`Jacobian`] and returns its derivative values in output-major/input-minor order.
    #[inline]
    pub fn into_values(self) -> Vec<V> {
        self.values
    }

    /// Returns the [`JacobianBlock`] of this [`Jacobian`] for the specified output and input [`ParameterPath`]s,
    /// or `None` if either path is absent.
    pub fn block(&self, output_path: &ParameterPath, input_path: &ParameterPath) -> Option<JacobianBlock<'_, T, V>> {
        let input_count = self.input_type.parameter_count();
        let (output_index, (_, output_type)) =
            self.output_type.named_parameters().enumerate().find(|(_, (path, _))| path == output_path)?;
        let (input_index, (_, input_type)) =
            self.input_type.named_parameters().enumerate().find(|(_, (path, _))| path == input_path)?;
        Some(JacobianBlock {
            output_path: output_path.clone(),
            output_type,
            input_path: input_path.clone(),
            input_type,
            value: &self.values[output_index * input_count + input_index],
        })
    }

    /// Returns borrowed views of all [`JacobianBlock`]s of this [`Jacobian`] in output-major/input-minor order.
    pub fn iter_blocks(&self) -> impl Iterator<Item = JacobianBlock<'_, T, V>> {
        let input_count = self.input_type.parameter_count();
        self.output_type
            .named_parameters()
            .enumerate()
            .flat_map(move |(output_index, (output_path, output_type))| {
                self.input_type.named_parameters().enumerate().map(move |(input_index, (input_path, input_type))| {
                    JacobianBlock {
                        output_path: output_path.clone(),
                        output_type,
                        input_path,
                        input_type,
                        value: &self.values[output_index * input_count + input_index],
                    }
                })
            })
    }
}

/// Borrowed view of one output/input block in a [`Jacobian`].
#[derive(Debug)]
pub struct JacobianBlock<'o, T: Type, V> {
    /// [`ParameterPath`] of the differentiated output [`Parameter`] that this [`JacobianBlock`] corresponds to.
    output_path: ParameterPath,

    /// [`Type`] of the differentiated output [`Parameter`] that this [`JacobianBlock`] corresponds to.
    output_type: &'o T,

    /// [`ParameterPath`] of the differentiated input [`Parameter`] that this [`JacobianBlock`] corresponds to.
    input_path: ParameterPath,

    /// [`Type`] of the differentiated input [`Parameter`] that this [`JacobianBlock`] corresponds to.
    input_type: &'o T,

    /// Derivative value for this [`JacobianBlock`].
    value: &'o V,
}

impl<'o, T: Type, V> JacobianBlock<'o, T, V> {
    /// Returns the [`ParameterPath`] of the differentiated output [`Parameter`] that this [`JacobianBlock`]
    /// corresponds to.
    #[inline]
    pub fn output_path(&self) -> &ParameterPath {
        &self.output_path
    }

    /// Returns the [`Type`] of the differentiated output [`Parameter`] that this [`JacobianBlock`] corresponds to.
    #[inline]
    pub fn output_type(&self) -> &'o T {
        self.output_type
    }

    /// Returns the [`ParameterPath`] of the differentiated input [`Parameter`] that this [`JacobianBlock`]
    /// corresponds to.
    #[inline]
    pub fn input_path(&self) -> &ParameterPath {
        &self.input_path
    }

    /// Returns the [`Type`] of the differentiated input [`Parameter`] that this [`JacobianBlock`] corresponds to.
    #[inline]
    pub fn input_type(&self) -> &'o T {
        self.input_type
    }

    /// Returns the derivative value for this [`JacobianBlock`].
    #[inline]
    pub fn value(&self) -> &'o V {
        self.value
    }
}

impl<'o, T: Type, V> Clone for JacobianBlock<'o, T, V> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            output_path: self.output_path.clone(),
            output_type: self.output_type,
            input_path: self.input_path.clone(),
            input_type: self.input_type,
            value: self.value,
        }
    }
}

/// Implements the forward-mode differentiation transform used by
/// [`jacobian_forward`](crate::DifferentiationBuilder::jacobian_forward) in an explicitly
/// provided [`Context`]. It linearizes once, replays a packed active-input basis, and returns
/// auxiliary output unchanged.
///
/// # Parameters
///
///   - `context`: Context in which to trace and replay the transform.
///   - `function`: Function returning the differentiated output and auxiliary output.
///   - `primal`: Structured input value specifying the linearization point.
///   - `capture`: Structured runtime value held fixed while differentiating the input.
///   - `holomorphic`: Whether to validate all differentiated leaves under a holomorphy promise.
#[allow(clippy::type_complexity)]
pub(crate) fn jacobian_forward_in_context<
    C: Context,
    I: Parameterized<
            C::Value,
            To<C::Value> = I,
            Family: ParameterizedFamily<LinearizationTracer<C>> + ParameterizedFamily<C::Type>,
        >,
    O: Parameterized<LinearizationTracer<C>, Family: ParameterizedFamily<C::Value> + ParameterizedFamily<C::Type>>,
    AuxiliaryOutput: Parameterized<LinearizationTracer<C>, Family: ParameterizedFamily<C::Value>>,
    Capture: Parameterized<C::Value, To<C::Value> = Capture, Family: ParameterizedFamily<LinearizationTracer<C>>>,
    F: FnOnce(
        I::To<LinearizationTracer<C>>,
        Capture::To<LinearizationTracer<C>>,
    ) -> Result<(O, AuxiliaryOutput), ProgramError>,
>(
    context: &C,
    function: F,
    primal: I,
    capture: Capture,
    holomorphic: bool,
) -> Result<
    (Jacobian<C::Type, C::Value, I::To<C::Type>, O::To<C::Type>>, AuxiliaryOutput::To<C::Value>),
    DifferentiationError,
>
where
    C::Type: DenseDifferentiableType<C>,
    C::Operation: PartiallyEvaluatableOperation<C>
        + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
        + ResidualZeroProvider<C::Type, Operation = C::Operation>,
    TracingContext<C::Constant, C::Operation>:
        StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
{
    // Preserve the input tree while deriving an isomorphic tree of input types. Validate differentiability and the
    // ordinary-versus-holomorphic complex-type contract before tracing the derivative program.
    let input_structure = primal.parameter_structure();
    let input_values = primal.into_parameters().collect::<Vec<_>>();
    let input_types = I::To::<C::Type>::from_parameters(
        input_structure.clone(),
        input_values.iter().map(|value| value.r#type().into_owned()),
    )?;
    JacobianMode::Forward.validate_types(&input_types, holomorphic, DifferentiationParameterRole::Input)?;
    let primals = I::from_parameters(input_structure, input_values)?;

    // Assign each input parameter a contiguous range in the flattened input coordinate space. The final prefix offset
    // is the total number of basis directions packed into the replay batch. Validating this before linearization also
    // reports a non-finite input coordinate space with its input role before the closure observes the outputs.
    let input_offsets = coordinate_prefix_offsets::<C, _>(
        &input_types,
        DerivativeTransform::JacobianForward,
        DifferentiationParameterRole::Input,
    )?;
    let batch_size = input_offsets.last().copied().unwrap();

    // Evaluate and linearize the differentiated output once. `linearize` returns only the differentiated output,
    // so capture the auxiliary primals while its closure runs and move them back out after constructing the Jacobian.
    // Output types are validated inside the closure, before the forward Jacobian materializes a structural zero output
    // tangent in a nonzero dynamic array differential space, so a non-finite coordinate space retains the precise
    // Jacobian diagnostic. Validate here because a nullary constructor, which sees only the type, still cannot
    // materialize a dynamic structural zero. A future input-relative zero operation may make this early validation
    // redundant.
    let mut auxiliary = None;
    let (output, pushforward) = context.linearize(
        |input, capture| {
            let (output, value) = function(input, capture)?;
            auxiliary = Some(extract_auxiliary_primals(value).map_err(ProgramError::from)?);
            let output_structure = output.parameter_structure();
            let output_values = output.into_parameters().collect::<Vec<_>>();
            let output_types = O::To::<C::Type>::from_parameters(
                output_structure.clone(),
                output_values.iter().map(|value| value.r#type().into_owned()),
            )?;
            JacobianMode::Forward
                .validate_types(&output_types, holomorphic, DifferentiationParameterRole::Output)
                .map_err(ProgramError::from)?;
            coordinate_prefix_offsets::<C, _>(
                &output_types,
                DerivativeTransform::JacobianForward,
                DifferentiationParameterRole::Output,
            )
            .map_err(ProgramError::from)?;
            Ok(O::from_parameters(output_structure, output_values)?)
        },
        primals,
        capture,
    )?;

    // Recover and validate the output type tree from the primal output returned by linearization. These types later
    // determine how each packed pushforward output is partitioned into output/input Jacobian blocks.
    let output_structure = output.parameter_structure();
    let output_values = output.into_parameters().collect::<Vec<_>>();
    let output_types = O::To::<C::Type>::from_parameters(
        output_structure,
        output_values.iter().map(|value| value.r#type().into_owned()),
    )?;
    JacobianMode::Forward.validate_types(&output_types, holomorphic, DifferentiationParameterRole::Output)?;

    // A pushforward program consumes one tangent per input parameter followed by the closed-over primal residuals.
    // Verify that the derived program still satisfies this contract before constructing its packed inputs.
    let (_, program, residuals, _, _, tangent_output_slots, _) = pushforward.into_parts();
    let program_input_types = program.input_types();
    let tangent_input_count = program_input_types.len().checked_sub(residuals.len()).ok_or_else(|| {
        ProgramError::MalformedProgram(format!(
            "pushforward program consumes {} inputs which is fewer than its {} residuals",
            program_input_types.len(),
            residuals.len(),
        ))
    })?;
    if tangent_input_count != input_types.parameter_count() {
        return Err(ProgramError::MalformedProgram(format!(
            "pushforward program consumes {} tangent inputs but the differentiated input has {} leaves",
            tangent_input_count,
            input_types.parameter_count(),
        ))
        .into());
    }
    for (index, (program_input_type, (path, input_type))) in
        program_input_types[..tangent_input_count].iter().zip(input_types.named_parameters()).enumerate()
    {
        let tangent_type = input_type.tangent()?;
        if tangent_type.is_zero_space() {
            return Err(DifferentiationError::NonDifferentiableParameter {
                transform: DerivativeTransform::JacobianForward,
                role: DifferentiationParameterRole::Input,
                path: path.to_string(),
                r#type: input_type.to_string(),
            });
        }
        if program_input_type != &tangent_type {
            return Err(ProgramError::MalformedProgram(format!(
                "pushforward tangent input {index} has type {program_input_type} but the differentiated input \
                parameter has tangent type {tangent_type}",
            ))
            .into());
        }
    }

    // For every input parameter, construct a packed standard basis occupying that parameter's coordinate range in the
    // shared batch. Replicate residuals because the same linearization point is reused for every tangent direction.
    let mut packed_inputs = input_types
        .named_parameters()
        .enumerate()
        .map(|(index, (path, r#type))| {
            let tangent_type = r#type.tangent()?;
            if tangent_type.is_zero_space() {
                return Err(DifferentiationError::NonDifferentiableParameter {
                    transform: DerivativeTransform::JacobianForward,
                    role: DifferentiationParameterRole::Input,
                    path: path.to_string(),
                    r#type: r#type.to_string(),
                });
            }
            C::Type::coordinate_basis(context, r#type, &tangent_type, input_offsets[index], batch_size)
        })
        .collect::<Result<Vec<_>, _>>()?;
    packed_inputs.extend(residuals.into_iter().map(C::Type::replicated));

    // Replay all basis tangents together. Each packed output now contains the derivative of one output parameter
    // with respect to every flattened input coordinate. An output parameter without a tangent slot has a structural
    // zero tangent, so its packed derivative is a zero shared across every direction. Dense coordinate spaces were
    // validated above, so a zero constructed from the type alone is sufficient.
    let packed_outputs =
        C::Type::replay_derivative_region(context, program.entry_region_ref(), batch_size, packed_inputs)?;
    check_count!("output", tangent_output_slots, output_types.parameter_count(), ProgramError);
    check_count!(
        "output",
        packed_outputs,
        tangent_output_slots.iter().filter(|&&has_slot| has_slot).count(),
        ProgramError,
    );
    let mut packed_outputs = packed_outputs.into_iter();
    let packed_outputs = output_types
        .parameters()
        .zip(&tangent_output_slots)
        .map(|(output_type, &has_slot)| {
            if has_slot {
                return Ok(packed_outputs.next().unwrap());
            }
            let mut zero = context.bind(
                C::Operation::provide(ZeroOperation::new(output_type.tangent()?), &[])?,
                Vec::new(),
                &[],
            )?;
            check_count!("output", zero, 1, ProgramError);
            Ok(C::Type::replicated(zero.remove(0)))
        })
        .collect::<Result<Vec<_>, DifferentiationError>>()?;

    // Slice each packed output by the coordinate range of each input parameter. Iterating outputs outside inputs
    // produces the output-major/input-minor block order required by `Jacobian`.
    let mut values = Vec::new();
    for (output_index, output_type) in output_types.parameters().enumerate() {
        for (input_index, input_type) in input_types.parameters().enumerate() {
            let value = C::Type::extract_forward_jacobian_block(
                &packed_outputs[output_index],
                batch_size,
                input_offsets[input_index],
                input_type,
                output_type,
            )?;
            values.push(value);
        }
    }

    // Reattach the type trees to the extracted blocks and return the auxiliary primals captured during linearization.
    let jacobian = Jacobian::new(input_types, output_types, values)?;
    let auxiliary = auxiliary.ok_or_else(|| {
        ProgramError::MalformedProgram(
            "the forward-mode Jacobian computation did not evaluate its function".to_string(),
        )
    })?;
    Ok((jacobian, auxiliary))
}

/// Implements the reverse-mode differentiation transform used by
/// [`jacobian_reverse`](crate::DifferentiationBuilder::jacobian_reverse) in an explicitly
/// provided [`Context`]. It constructs one pullback, replays a packed differentiated-output basis,
/// and returns auxiliary output unchanged.
///
/// # Parameters
///
///   - `context`: Context in which to trace and replay the transform.
///   - `function`: Function returning the differentiated output and auxiliary output.
///   - `primal`: Structured input value specifying the linearization point.
///   - `capture`: Structured runtime value held fixed while differentiating the input.
///   - `holomorphic`: Whether to validate all differentiated leaves under a holomorphy promise.
#[allow(clippy::type_complexity)]
pub(crate) fn jacobian_reverse_in_context<
    C: Context,
    I: Parameterized<
            C::Value,
            To<C::Value> = I,
            Family: ParameterizedFamily<LinearizationTracer<C>> + ParameterizedFamily<C::Type>,
        >,
    O: Parameterized<LinearizationTracer<C>, Family: ParameterizedFamily<C::Value> + ParameterizedFamily<C::Type>>,
    AuxiliaryOutput: Parameterized<LinearizationTracer<C>, Family: ParameterizedFamily<C::Value>>,
    Capture: Parameterized<C::Value, To<C::Value> = Capture, Family: ParameterizedFamily<LinearizationTracer<C>>>,
    F: FnOnce(
        I::To<LinearizationTracer<C>>,
        Capture::To<LinearizationTracer<C>>,
    ) -> Result<(O, AuxiliaryOutput), ProgramError>,
>(
    context: &C,
    function: F,
    primal: I,
    capture: Capture,
    holomorphic: bool,
) -> Result<
    (Jacobian<C::Type, C::Value, I::To<C::Type>, O::To<C::Type>>, AuxiliaryOutput::To<C::Value>),
    DifferentiationError,
>
where
    C::Type: DenseDifferentiableType<C> + ReferenceMemberType,
    C::Operation: PartiallyEvaluatableOperation<C>
        + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
        + DifferentiableOperation<PartialEvaluationContext<C>>
        + TransposableOperation<C::Constant, C::Operation>
        + ResidualZeroProvider<C::Type, Operation = C::Operation>
        + ReferenceAccessOperation<Transform: ReferenceTransform<Referent = <C::Type as ReferenceMemberType>::Referent>>
        + OperationProvider<
            C::Type,
            ReferenceNewOperation<<C::Type as ReferenceMemberType>::Referent, C::Type>,
            Operation = C::Operation,
        > + OperationProvider<
            C::Type,
            ReferenceAddUpdateOperation<
                <C::Type as ReferenceMemberType>::Referent,
                C::Type,
                <C::Operation as ReferenceAccessOperation>::Transform,
            >,
            Operation = C::Operation,
        > + OperationProvider<C::Type, AddOperation<C::Type>, Operation = C::Operation>,
    TracingContext<C::Constant, C::Operation>:
        StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
{
    // Preserve the input tree while deriving and validating its isomorphic type tree. Reverse mode permits complex
    // inputs in the ordinary case, but still requires each input parameter to have a nonzero cotangent space.
    let input_structure = primal.parameter_structure();
    let input_values = primal.into_parameters().collect::<Vec<_>>();
    let input_types = I::To::<C::Type>::from_parameters(
        input_structure.clone(),
        input_values.iter().map(|value| value.r#type().into_owned()),
    )?;
    JacobianMode::Reverse.validate_types(&input_types, holomorphic, DifferentiationParameterRole::Input)?;
    let primals = I::from_parameters(input_structure, input_values)?;

    // Dense Jacobians enumerate every input coordinate, including disconnected inputs. Validate those coordinate
    // spaces before tracing the pullback so unsupported dynamic inputs report their original boundary role.
    coordinate_prefix_offsets::<C, _>(
        &input_types,
        DerivativeTransform::JacobianReverse,
        DifferentiationParameterRole::Input,
    )?;

    // Evaluate the differentiated output once and construct its reusable pullback. `vjp` returns only the
    // differentiated output, so capture the auxiliary primals while its closure runs and move them back out after
    // constructing the Jacobian.
    let mut auxiliary = None;
    let (output, pullback) = context.vjp(
        |input, capture| {
            let (output, value) = function(input, capture)?;
            auxiliary = Some(extract_auxiliary_primals(value).map_err(ProgramError::from)?);
            Ok(output)
        },
        primals,
        capture,
    )?;

    // Recover and validate the output type tree from the primal output. In reverse mode these types determine the
    // cotangent basis used to enumerate the rows of the Jacobian.
    let output_structure = output.parameter_structure();
    let output_values = output.into_parameters().collect::<Vec<_>>();
    let output_types = O::To::<C::Type>::from_parameters(
        output_structure,
        output_values.iter().map(|value| value.r#type().into_owned()),
    )?;
    JacobianMode::Reverse.validate_types(&output_types, holomorphic, DifferentiationParameterRole::Output)?;

    // Assign each output parameter a contiguous range in the flattened output coordinate space. The final output
    // offset is the number of cotangent basis directions in the replay batch.
    let output_offsets = coordinate_prefix_offsets::<C, _>(
        &output_types,
        DerivativeTransform::JacobianReverse,
        DifferentiationParameterRole::Output,
    )?;
    let batch_size = output_offsets.last().copied().unwrap();

    // A pullback program consumes one cotangent per output parameter with a tangent slot followed by its closed-over
    // primal residuals. Verify that the derived program exposes that input contract before constructing the packed
    // cotangents. The reverse Jacobian materializes returned cotangents only, so it takes the reference-free,
    // all-`Return` transposition of the retained linear program. An output parameter without a tangent slot has a
    // structural zero tangent and seeds nothing, so its coordinates receive no contribution and its rows of the
    // Jacobian are zero.
    let tangent_output_slots = pullback.tangent_output_slots().to_vec();
    let (transposed, residuals) = pullback.into_transposed_parts()?;
    let program = transposed.program();
    let program_input_types = program.input_types();
    let cotangent_input_count = program_input_types.len().checked_sub(residuals.len()).ok_or_else(|| {
        ProgramError::MalformedProgram(format!(
            "pullback program consumes {} inputs which is fewer than its {} residuals",
            program_input_types.len(),
            residuals.len(),
        ))
    })?;
    check_count!("output", tangent_output_slots, output_types.parameter_count(), ProgramError);
    let seeded_output_count = tangent_output_slots.iter().filter(|&&has_slot| has_slot).count();
    if cotangent_input_count != seeded_output_count {
        return Err(ProgramError::MalformedProgram(format!(
            "pullback program consumes {} cotangent inputs but the differentiated output has {} leaves with tangent \
             slots",
            cotangent_input_count, seeded_output_count,
        ))
        .into());
    }

    // Construct one packed standard cotangent basis in its assigned coordinate range for each output parameter that
    // the pullback program seeds. Replicate residuals so every cotangent direction reuses the same primal
    // linearization point.
    let mut packed_inputs = Vec::with_capacity(seeded_output_count + residuals.len());
    for (index, ((path, r#type), &has_slot)) in output_types.named_parameters().zip(&tangent_output_slots).enumerate() {
        let cotangent_type = r#type.cotangent()?;
        if cotangent_type.is_zero_space() {
            return Err(DifferentiationError::NonDifferentiableParameter {
                transform: DerivativeTransform::JacobianReverse,
                role: DifferentiationParameterRole::Output,
                path: path.to_string(),
                r#type: r#type.to_string(),
            });
        }
        if !has_slot {
            continue;
        }
        let program_input_type = &program_input_types[packed_inputs.len()];
        if program_input_type != &cotangent_type {
            return Err(ProgramError::MalformedProgram(format!(
                "pullback cotangent input {} has type {} but output parameter {} has cotangent type {}",
                packed_inputs.len(),
                program_input_type,
                path,
                cotangent_type,
            ))
            .into());
        }
        packed_inputs.push(C::Type::coordinate_basis(
            context,
            r#type,
            &cotangent_type,
            output_offsets[index],
            batch_size,
        )?);
    }
    packed_inputs.extend(residuals.into_iter().map(C::Type::replicated));

    // Replay all output cotangent directions together. Each packed result corresponds to one input parameter and
    // contains that input parameter's sensitivities to every flattened output coordinate.
    let packed_outputs =
        C::Type::replay_derivative_region(context, program.entry_region_ref(), batch_size, packed_inputs)?;
    check_count!("output", packed_outputs, program.output_ids().len(), ProgramError);
    if transposed.cotangents().len() != input_types.parameter_count() {
        return Err(ProgramError::MalformedProgram(format!(
            "reverse Jacobian pullback has {} cotangent slots for {} input leaves",
            transposed.cotangents().len(),
            input_types.parameter_count(),
        ))
        .into());
    }

    let mut outputs = packed_outputs.into_iter();
    let packed_outputs = transposed
        .cotangents()
        .iter()
        .zip(input_types.parameters())
        .map(|(slot, input_type)| {
            match slot {
                TransposedProgramCotangent::Output => Ok(outputs.next().unwrap()),
                TransposedProgramCotangent::Zero => {
                    // Dense coordinate spaces were validated above, so a type-only zero is sufficient. Sharing it
                    // across packed directions avoids materializing a zero in the compact pullback program.
                    let mut zero = context.bind(
                        C::Operation::provide(ZeroOperation::new(input_type.cotangent()?), &[])?,
                        Vec::new(),
                        &[],
                    )?;
                    check_count!("output", zero, 1, ProgramError);
                    Ok(C::Type::replicated(zero.remove(0)))
                }
                TransposedProgramCotangent::Absent => Err(ProgramError::MalformedProgram(
                    "reverse Jacobian pullback omitted a returned cotangent".to_string(),
                )
                .into()),
            }
        })
        .collect::<Result<Vec<_>, DifferentiationError>>()?;

    // Slice each packed input sensitivity by output-coordinate range and orient it as an output/input block.
    // The loop order restores the same output-major/input-minor representation produced by forward mode.
    let mut values = Vec::new();
    for (output_index, output_type) in output_types.parameters().enumerate() {
        for (input_index, input_type) in input_types.parameters().enumerate() {
            let value = C::Type::extract_reverse_jacobian_block(
                &packed_outputs[input_index],
                batch_size,
                output_offsets[output_index],
                output_type,
                input_type,
            )?;
            values.push(value);
        }
    }

    // Reattach the type trees to the extracted blocks and return the auxiliary primals captured
    // while building the pullback.
    let jacobian = Jacobian::new(input_types, output_types, values)?;
    let auxiliary = auxiliary.ok_or_else(|| {
        ProgramError::MalformedProgram(
            "the reverse-mode Jacobian computation did not evaluate its function".to_string(),
        )
    })?;
    Ok((jacobian, auxiliary))
}

/// Direction of a dense Jacobian materialization.
#[derive(Copy, Clone)]
enum JacobianMode {
    Forward,
    Reverse,
}

impl JacobianMode {
    /// Validates the differential representations and complex-type requirements of the provided parameter types.
    ///
    /// # Parameters
    ///
    ///   - `types`: Parameter types to validate.
    ///   - `holomorphic`: Whether the Jacobian is being materialized under a holomorphy promise.
    ///   - `role`: Role of `types` in the Jacobian transform.
    fn validate_types<T: DifferentiableType, Types: Parameterized<T>>(
        self,
        types: &Types,
        holomorphic: bool,
        role: DifferentiationParameterRole,
    ) -> Result<(), DifferentiationError> {
        let transform = match self {
            Self::Forward => DerivativeTransform::JacobianForward,
            Self::Reverse => DerivativeTransform::JacobianReverse,
        };
        for (path, r#type) in types.named_parameters() {
            let differential_type = match self {
                Self::Forward => r#type.tangent(),
                Self::Reverse => r#type.cotangent(),
            }?;
            if differential_type.is_zero_space() {
                return Err(DifferentiationError::NonDifferentiableParameter {
                    transform,
                    role,
                    path: path.to_string(),
                    r#type: r#type.to_string(),
                });
            }
            if holomorphic && !r#type.is_complex() {
                return Err(DifferentiationError::NonComplexParameter {
                    transform,
                    role,
                    path: path.to_string(),
                    r#type: r#type.to_string(),
                });
            }
            if !holomorphic
                && r#type.is_complex()
                && !matches!(
                    (self, role),
                    (Self::Forward, DifferentiationParameterRole::Output)
                        | (Self::Reverse, DifferentiationParameterRole::Input)
                )
            {
                return Err(DifferentiationError::ComplexParameter {
                    transform,
                    role,
                    path: path.to_string(),
                    r#type: r#type.to_string(),
                });
            }
        }
        Ok(())
    }
}

/// Returns the exclusive prefix offsets of the flattened coordinate spaces represented by `types`. If the parameter
/// leaves have coordinate-space dimensions `[d0, d1, ..., dn]`, the returned vector is `[0, d0, d0 + d1, ...,
/// d0 + d1 + ... + dn]`. Consequently, element `i` is the first packed coordinate direction belonging to leaf `i`,
/// and the final element is the total number of packed directions. Each leaf dimension is obtained from
/// [`DenseDifferentiableType::coordinate_space_dimension`]. Returns a [`DifferentiationError`] if any
/// leaf does not have a finite coordinate space or if a cumulative offset does not fit in [`usize`].
///
/// # Parameters
///
///   - `types`: Structured parameter types whose leaf coordinate spaces are flattened in parameter order.
///   - `transform`: Derivative transform requesting the offsets, used in diagnostics.
///   - `role`: Role of `types` in the derivative transform, used in diagnostics.
fn coordinate_prefix_offsets<C: Context, Types: Parameterized<C::Type>>(
    types: &Types,
    transform: DerivativeTransform,
    role: DifferentiationParameterRole,
) -> Result<Vec<usize>, DifferentiationError>
where
    C::Type: DenseDifferentiableType<C>,
{
    let mut offsets = Vec::new();
    offsets.push(0usize);
    for (path, r#type) in types.named_parameters() {
        let dimension = C::Type::coordinate_space_dimension(r#type, transform, role, &path)?;
        offsets.push(offsets.last().copied().unwrap().checked_add(dimension).ok_or_else(|| {
            DifferentiationError::CoordinateCountOverflow {
                transform,
                role,
                path: path.to_string(),
                r#type: r#type.to_string(),
            }
        })?);
    }
    Ok(offsets)
}

/// Extracts the known underlying primal value from each auxiliary [`LinearizationTracer`] and reconstructs the
/// auxiliary parameter structure.  This function does not evaluate or otherwise materialize the auxiliary outputs.
/// Every tracer must already carry a known primal because auxiliary outputs are not differentiated.
fn extract_auxiliary_primals<
    C: Context<
        Operation: PartiallyEvaluatableOperation<C>
                       + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>,
    >,
    A: Parameterized<LinearizationTracer<C>, Family: ParameterizedFamily<C::Value>>,
>(
    auxiliary: A,
) -> Result<A::To<C::Value>, DifferentiationError>
where
    TracingContext<C::Constant, C::Operation>:
        StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
{
    let structure = auxiliary.parameter_structure();
    let values = auxiliary
        .into_parameters()
        .map(|tracer| {
            let (primal, _) = tracer.into_dual().into_parts();
            match primal.into_value()?.value().clone() {
                PartialValue::Known(value) => Ok(value),
                PartialValue::Unknown(r#type) => Err(ProgramError::MalformedProgram(format!(
                    "auxiliary output has unknown primal type {type} but depends only on known primal inputs",
                ))),
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(A::To::<C::Value>::from_parameters(structure, values)?)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::sync::Arc;

    use approx::assert_abs_diff_eq;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::DataType::{F32, F64};
    use crate::arrays::{
        Array, ArrayOperation, ArrayType, DataType, Dimension, DimensionBounds, DimensionVariable, Shape, f8e8m0fnu,
    };
    use crate::batching::{BatchAxis, batch};
    use crate::contexts::{Context, EagerContext};
    use crate::differentiation::{
        DerivativeTransform, Differentiate, DifferentiationError, DifferentiationParameterRole, differentiate_at,
    };
    use crate::operations::{Add, Compare, ComparisonDirection, Select, Sin, ZeroLike};
    use crate::parameters::{ParameterPath, Parameterized};
    use crate::programs::{Instruction, Operation, Program, Provenance, ProvenanceScope, Typed, Value};

    use super::*;

    /// Returns `2x` for positive `x` and `3x` otherwise, expressed generically so both dense Jacobian modes exercise
    /// comparison, selection, and arithmetic while constructing their coordinate-basis replays.
    fn piecewise_select<V: Value<Type = ArrayType> + Add + Compare + Select + ZeroLike>(x: V) -> V {
        let condition = x.compare(&x.zero_like().unwrap(), ComparisonDirection::GreaterThan).unwrap();
        let doubled = x.add(&x).unwrap();
        let tripled = doubled.add(&x).unwrap();
        condition.select(&doubled, &tripled).unwrap()
    }

    #[test]
    fn test_jacobian() {
        // Parameterization preserves input/output metadata and the flattened block values.
        let jacobian = Jacobian::new((F32, vec![F64, F32]), F64, vec![1.0_f32, 2.0, 3.0]).unwrap();
        assert_eq!(jacobian.parameter_count(), 3);
        assert_eq!(jacobian.values(), &[1.0, 2.0, 3.0]);
        let reparameterized =
            <Jacobian<DataType, f64, _, _>>::from_parameters(jacobian.parameter_structure(), [4.0, 5.0, 6.0]).unwrap();
        assert_eq!(reparameterized.input_type(), &(F32, vec![F64, F32]));
        assert_eq!(reparameterized.output_type(), &F64);
        assert_eq!(reparameterized.values(), &[4.0, 5.0, 6.0]);

        // Block iteration uses output-major/input-minor order and block lookup addresses the same structure by path.
        let input_types = (ArrayType::scalar(F32), ArrayType::new(F32, Shape::new(vec![2.into()])));
        let output_types = ArrayType::new(F32, Shape::new(vec![3.into()]));
        let jacobian = Jacobian::new(input_types.clone(), output_types.clone(), vec![10_i32, 20]).unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].output_path(), &ParameterPath::root());
        assert_eq!(blocks[0].input_path().to_string(), "$.0");
        assert_eq!(blocks[0].output_type(), &output_types);
        assert_eq!(blocks[0].input_type(), &input_types.0);
        assert_eq!(*blocks[0].value(), 10);
        assert_eq!(blocks[1].input_path().to_string(), "$.1");
        assert_eq!(*blocks[1].value(), 20);
        let second_input_path = blocks[1].input_path().clone();
        assert_eq!(*jacobian.block(&ParameterPath::root(), &second_input_path).unwrap().value(), 20);
        assert!(jacobian.block(&ParameterPath::root(), &ParameterPath::root().field("missing")).is_none());
    }

    #[test]
    fn test_jacobian_forward() {
        // A vector identity function packs every input coordinate direction into one replay and reconstructs the
        // complete identity matrix as a single output/input block.
        let jacobian = differentiate_at(Array::vector(vec![1.0, 2.0, 3.0]).unwrap())
            .jacobian_forward(|input| Ok(input))
            .unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(
            block.value().r#type().into_owned(),
            ArrayType::new(F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(3)])),
        );
        assert_eq!(block.value().to_f64s(), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

        // Structured inputs and outputs produce blocks in output-major/input-minor order.
        let jacobian = differentiate_at((Array::scalar(2.0).unwrap(), Array::scalar(3.0).unwrap()))
            .jacobian_forward(|(x, y)| Ok((x.clone() * y.clone() + x.sin()?, x + y)))
            .unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 4);
        assert_eq!(blocks[0].output_path().to_string(), "$.0");
        assert_eq!(blocks[0].input_path().to_string(), "$.0");
        assert_abs_diff_eq!(blocks[0].value().to_f64s()[0], 3.0 + 2.0f64.cos(), epsilon = 1e-9);
        assert_eq!(blocks[1].output_path().to_string(), "$.0");
        assert_eq!(blocks[1].input_path().to_string(), "$.1");
        assert_abs_diff_eq!(blocks[1].value().to_f64s()[0], 2.0, epsilon = 1e-9);
        assert_eq!(blocks[2].output_path().to_string(), "$.1");
        assert_eq!(blocks[2].input_path().to_string(), "$.0");
        assert_abs_diff_eq!(blocks[2].value().to_f64s()[0], 1.0, epsilon = 1e-9);
        assert_eq!(blocks[3].output_path().to_string(), "$.1");
        assert_eq!(blocks[3].input_path().to_string(), "$.1");
        assert_abs_diff_eq!(blocks[3].value().to_f64s()[0], 1.0, epsilon = 1e-9);

        // An output independent of one input retains an explicit zero block in the Cartesian product.
        let jacobian = differentiate_at((Array::scalar(2.0).unwrap(), Array::scalar(3.0).unwrap()))
            .jacobian_forward(|(x, y)| Ok((x.clone() * y.clone() + x.sin()?, y.clone(), x + y)))
            .unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 6);
        assert_abs_diff_eq!(blocks[0].value().to_f64s()[0], 3.0 + 2.0f64.cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(blocks[1].value().to_f64s()[0], 2.0, epsilon = 1e-9);
        assert_abs_diff_eq!(blocks[2].value().to_f64s()[0], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(blocks[3].value().to_f64s()[0], 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(blocks[4].value().to_f64s()[0], 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(blocks[5].value().to_f64s()[0], 1.0, epsilon = 1e-9);

        // Forward replay follows primal control flow and selects the tangent of the branch taken at the primal point.
        let jacobian =
            differentiate_at(Array::scalar(2.0).unwrap()).jacobian_forward(|x| Ok(piecewise_select(x))).unwrap();
        assert_abs_diff_eq!(jacobian.iter_blocks().next().unwrap().value().to_f64s()[0], 2.0, epsilon = 1e-9);

        let jacobian = differentiate_at(Array::scalar(-2.0).unwrap())
            .jacobian_forward(|x| Ok(piecewise_select(x)))
            .unwrap();
        assert_abs_diff_eq!(jacobian.iter_blocks().next().unwrap().value().to_f64s()[0], 3.0, epsilon = 1e-9);

        // Narrow primal element types use their widened differential representation for dense Jacobian blocks.
        let input = Array::from_elements::<f8e8m0fnu>(
            ArrayType::scalar(DataType::F8E8M0FNU),
            &[2.0].map(|value| f8e8m0fnu::from_f64(value).unwrap()),
        )
        .unwrap();
        let jacobian = differentiate_at(input).jacobian_forward(|value| value.sin()).unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.value().r#type().as_ref(), &ArrayType::scalar(F32));
        assert_abs_diff_eq!(block.value().to_f64s()[0], 2.0f64.cos(), epsilon = 1e-6);

        // Scalar inputs broadcast into vector outputs are unbroadcast when their dense block is reconstructed.
        let jacobian = differentiate_at((Array::scalar(2.0).unwrap(), Array::vector(vec![3.0, 4.0]).unwrap()))
            .jacobian_forward(|(scalar, vector)| Ok(scalar.clone() * vector + scalar))
            .unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].value().r#type().static_shape().unwrap().as_slice(), &[2]);
        assert_eq!(blocks[0].value().to_f64s(), vec![4.0, 5.0]);
        assert_eq!(blocks[1].value().r#type().static_shape().unwrap().as_slice(), &[2, 2]);
        assert_eq!(blocks[1].value().to_f64s(), vec![2.0, 0.0, 0.0, 2.0]);

        // Zero-sized inputs and outputs remain concrete, honestly typed dense blocks.
        let r#type = ArrayType::new(F64, Shape::new(vec![Dimension::Static(0)]));
        let jacobian = differentiate_at(Array::from_elements::<f64>(r#type.clone(), &[]).unwrap())
            .jacobian_forward(|input| Ok(input.clone() + input))
            .unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.output_type().static_shape().unwrap().as_slice(), &[0]);
        assert_eq!(block.input_type().static_shape().unwrap().as_slice(), &[0]);
        assert_eq!(block.value().r#type().as_ref(), &r#type.with_inserted_dimension(0, Dimension::Static(0)).unwrap(),);
        assert!(block.value().storage_bytes().is_empty());

        // The holomorphic entry point treats a complex derivative as complex linear.
        let input = Array::scalar(ComplexNumber::new(2.0f32, 1.0)).unwrap();
        let jacobian = differentiate_at(input).holomorphic().jacobian_forward(|x| Ok(x.clone() * x)).unwrap();
        assert_eq!(
            jacobian.iter_blocks().next().unwrap().value().elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(4.0, 2.0)]),
        );
    }

    #[test]
    fn test_jacobian_reverse() {
        // Structured inputs and outputs produce the same output-major/input-minor blocks as forward mode.
        let jacobian = differentiate_at((Array::scalar(2.0).unwrap(), Array::scalar(3.0).unwrap()))
            .jacobian_reverse(|(x, y)| Ok((x.clone() * y.clone() + x.sin()?, x + y)))
            .unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 4);
        assert_eq!(blocks[0].output_path().to_string(), "$.0");
        assert_eq!(blocks[0].input_path().to_string(), "$.0");
        assert_abs_diff_eq!(blocks[0].value().to_f64s()[0], 3.0 + 2.0f64.cos(), epsilon = 1e-9);
        assert_eq!(blocks[1].output_path().to_string(), "$.0");
        assert_eq!(blocks[1].input_path().to_string(), "$.1");
        assert_abs_diff_eq!(blocks[1].value().to_f64s()[0], 2.0, epsilon = 1e-9);
        assert_eq!(blocks[2].output_path().to_string(), "$.1");
        assert_eq!(blocks[2].input_path().to_string(), "$.0");
        assert_abs_diff_eq!(blocks[2].value().to_f64s()[0], 1.0, epsilon = 1e-9);
        assert_eq!(blocks[3].output_path().to_string(), "$.1");
        assert_eq!(blocks[3].input_path().to_string(), "$.1");
        assert_abs_diff_eq!(blocks[3].value().to_f64s()[0], 1.0, epsilon = 1e-9);

        // Reverse replay routes output cotangents through the branch selected at the primal point.
        let jacobian =
            differentiate_at(Array::scalar(2.0).unwrap()).jacobian_reverse(|x| Ok(piecewise_select(x))).unwrap();
        assert_abs_diff_eq!(jacobian.iter_blocks().next().unwrap().value().to_f64s()[0], 2.0, epsilon = 1e-9);

        let jacobian = differentiate_at(Array::scalar(-2.0).unwrap())
            .jacobian_reverse(|x| Ok(piecewise_select(x)))
            .unwrap();
        assert_abs_diff_eq!(jacobian.iter_blocks().next().unwrap().value().to_f64s()[0], 3.0, epsilon = 1e-9);

        // Per-element masking over a vector input makes the Jacobian diagonal, with entries 2 for positive inputs and
        // 3 otherwise.
        let jacobian = differentiate_at(Array::vector(vec![1.0, -1.0]).unwrap())
            .jacobian_reverse(|x| Ok(piecewise_select(x)))
            .unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.output_type().static_shape().unwrap().as_slice(), &[2]);
        assert_eq!(block.input_type().static_shape().unwrap().as_slice(), &[2]);
        assert_abs_diff_eq!(block.value().to_f64s()[0], 2.0, epsilon = 1e-9);
        assert_abs_diff_eq!(block.value().to_f64s()[1], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(block.value().to_f64s()[2], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(block.value().to_f64s()[3], 3.0, epsilon = 1e-9);

        // Pullback replay unbroadcasts a selected scalar branch and preserves each input's differential data type.
        let scalar = Array::from_elements::<f32>(ArrayType::scalar(F32), &[5.0]).unwrap();
        let f32_vector_type = ArrayType::new(F32, Shape::new(vec![Dimension::Static(2)]));
        let vector =
            Array::from_elements::<f64>(ArrayType::new(F64, Shape::new(vec![Dimension::Static(2)])), &[2.0, -3.0])
                .unwrap();

        let jacobian = differentiate_at((scalar.clone(), vector.clone()))
            .jacobian_reverse(|(scalar, vector)| {
                let condition = vector.compare(&vector.zero_like()?, ComparisonDirection::GreaterThan)?;
                condition.select(&scalar, &vector)
            })
            .unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].value().r#type().into_owned(), f32_vector_type);
        assert_eq!(blocks[0].value().to_f64s(), vec![1.0, 0.0]);
        assert_eq!(
            blocks[1].value().r#type().into_owned(),
            ArrayType::new(F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)])),
        );
        assert_eq!(blocks[1].value().to_f64s(), vec![0.0, 0.0, 0.0, 1.0]);

        let jacobian = differentiate_at((scalar, vector))
            .jacobian_reverse(|(scalar, vector)| {
                let condition = vector.compare(&vector.zero_like()?, ComparisonDirection::GreaterThan)?;
                condition.select(&vector, &scalar)
            })
            .unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].value().r#type().into_owned(), f32_vector_type);
        assert_eq!(blocks[0].value().to_f64s(), vec![0.0, 1.0]);
        assert_eq!(
            blocks[1].value().r#type().into_owned(),
            ArrayType::new(F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)])),
        );
        assert_eq!(blocks[1].value().to_f64s(), vec![1.0, 0.0, 0.0, 0.0]);

        // Promoted elementwise cotangents are converted back to the differential type of each input leaf.
        let f32 = Array::from_elements::<f32>(ArrayType::scalar(F32), &[2.0]).unwrap();
        let f64 = Array::from_elements::<f64>(ArrayType::scalar(F64), &[3.0]).unwrap();
        let jacobian = differentiate_at((f32.clone(), f64.clone()))
            .jacobian_reverse(|(left, right)| Ok(left + right))
            .unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].value().r#type().data_type(), F32);
        assert_eq!(blocks[1].value().r#type().data_type(), F64);
        assert_abs_diff_eq!(blocks[0].value().to_f64s()[0], 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(blocks[1].value().to_f64s()[0], 1.0, epsilon = 1e-9);

        let jacobian = differentiate_at((f32.clone(), f64.clone()))
            .jacobian_reverse(|(left, right)| Ok(left - right))
            .unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        assert_abs_diff_eq!(blocks[0].value().to_f64s()[0], 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(blocks[1].value().to_f64s()[0], -1.0, epsilon = 1e-9);

        let jacobian = differentiate_at((f32, f64)).jacobian_reverse(|(left, right)| Ok(left * right)).unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        assert_abs_diff_eq!(blocks[0].value().to_f64s()[0], 3.0, epsilon = 1e-9);
        assert_abs_diff_eq!(blocks[1].value().to_f64s()[0], 2.0, epsilon = 1e-9);

        // Narrow primal element types use their widened differential representation for dense Jacobian blocks.
        let input = Array::from_elements::<f8e8m0fnu>(
            ArrayType::scalar(DataType::F8E8M0FNU),
            &[2.0].map(|value| f8e8m0fnu::from_f64(value).unwrap()),
        )
        .unwrap();
        let jacobian = differentiate_at(input).jacobian_reverse(|value| value.sin()).unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.value().r#type().as_ref(), &ArrayType::scalar(F32));
        assert_abs_diff_eq!(block.value().to_f64s()[0], 2.0f64.cos(), epsilon = 1e-6);

        // Scalar inputs broadcast into vector outputs are unbroadcast when their dense block is reconstructed.
        let jacobian = differentiate_at((Array::scalar(2.0).unwrap(), Array::vector(vec![3.0, 4.0]).unwrap()))
            .jacobian_reverse(|(scalar, vector)| Ok(scalar.clone() * vector + scalar))
            .unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].value().r#type().static_shape().unwrap().as_slice(), &[2]);
        assert_eq!(blocks[0].value().to_f64s(), vec![4.0, 5.0]);
        assert_eq!(blocks[1].value().r#type().static_shape().unwrap().as_slice(), &[2, 2]);
        assert_eq!(blocks[1].value().to_f64s(), vec![2.0, 0.0, 0.0, 2.0]);

        // Zero-sized inputs and outputs remain concrete, honestly typed dense blocks.
        let r#type = ArrayType::new(F64, Shape::new(vec![Dimension::Static(0)]));
        let jacobian = differentiate_at(Array::from_elements::<f64>(r#type.clone(), &[]).unwrap())
            .jacobian_reverse(|input| Ok(input.clone() + input))
            .unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.output_type().static_shape().unwrap().as_slice(), &[0]);
        assert_eq!(block.input_type().static_shape().unwrap().as_slice(), &[0]);
        assert_eq!(block.value().r#type().as_ref(), &r#type.with_inserted_dimension(0, Dimension::Static(0)).unwrap(),);
        assert!(block.value().storage_bytes().is_empty());

        // The holomorphic entry point transposes a complex-linear pushforward without conjugating it.
        let input = Array::scalar(ComplexNumber::new(2.0f32, 1.0)).unwrap();
        let jacobian = differentiate_at(input).holomorphic().jacobian_reverse(|x| Ok(x.clone() * x)).unwrap();
        assert_eq!(
            jacobian.iter_blocks().next().unwrap().value().elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(4.0, 2.0)]),
        );
    }

    #[test]
    fn test_jacobian_reverse_disconnected_inputs() {
        let jacobian = differentiate_at((Array::vector(vec![1f64, 2.0]).unwrap(), Array::scalar(3f64).unwrap()))
            .jacobian_reverse(|(_, live)| Ok(live))
            .unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].value(), &Array::vector(vec![0f64, 0.0]).unwrap());
        assert_eq!(blocks[1].value(), &Array::scalar(1f64).unwrap());

        // All cotangents can be structural zeros: dense block construction still creates the complete input tree.
        let jacobian = differentiate_at(Array::vector(vec![1f64, 2.0]).unwrap())
            .jacobian_reverse(|input| input.context().lift(Array::scalar(7f64).unwrap()))
            .unwrap();
        assert_eq!(jacobian.values(), &[Array::vector(vec![0f64, 0.0]).unwrap()]);
    }

    #[test]
    fn test_jacobian_with_auxiliary_outputs() {
        let forward_evaluations = Cell::new(0);
        let (jacobian, auxiliary) = differentiate_at(Array::scalar(2.0).unwrap())
            .with_captures(())
            .with_auxiliary_output()
            .in_context(&EagerContext::<Array, ArrayOperation<Array>>::new())
            .jacobian_forward(|x, ()| {
                forward_evaluations.set(forward_evaluations.get() + 1);
                Ok((x.clone() * x.clone(), x))
            })
            .unwrap();
        assert_eq!(forward_evaluations.get(), 1);
        assert_abs_diff_eq!(jacobian.iter_blocks().next().unwrap().value().to_f64s()[0], 4.0, epsilon = 1e-9);
        assert_eq!(auxiliary.to_f64s(), vec![2.0]);

        let reverse_evaluations = Cell::new(0);
        let (jacobian, auxiliary) = differentiate_at(Array::scalar(2.0).unwrap())
            .with_captures(())
            .with_auxiliary_output()
            .in_context(&EagerContext::<Array, ArrayOperation<Array>>::new())
            .jacobian_reverse(|x, ()| {
                reverse_evaluations.set(reverse_evaluations.get() + 1);
                Ok((x.clone() * x.clone(), x))
            })
            .unwrap();
        assert_eq!(reverse_evaluations.get(), 1);
        assert_abs_diff_eq!(jacobian.iter_blocks().next().unwrap().value().to_f64s()[0], 4.0, epsilon = 1e-9);
        assert_eq!(auxiliary.to_f64s(), vec![2.0]);

        let input = Array::scalar(ComplexNumber::new(2.0f32, 1.0)).unwrap();
        let (jacobian, auxiliary) = differentiate_at(input.clone())
            .with_captures(())
            .with_auxiliary_output()
            .holomorphic()
            .in_context(&EagerContext::<Array, ArrayOperation<Array>>::new())
            .jacobian_forward(|x, ()| Ok((x.clone() * x.clone(), x)))
            .unwrap();
        assert_eq!(
            jacobian.iter_blocks().next().unwrap().value().elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(4.0, 2.0)]),
        );
        assert_eq!(auxiliary, input);

        let (jacobian, auxiliary) = differentiate_at(input.clone())
            .with_captures(())
            .with_auxiliary_output()
            .holomorphic()
            .in_context(&EagerContext::<Array, ArrayOperation<Array>>::new())
            .jacobian_reverse(|x, ()| Ok((x.clone() * x.clone(), x)))
            .unwrap();
        assert_eq!(
            jacobian.iter_blocks().next().unwrap().value().elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(4.0, 2.0)]),
        );
        assert_eq!(auxiliary, input);
    }

    #[test]
    fn test_jacobian_validation() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();

        assert_eq!(
            context
                .differentiate_at(Vec::<Array>::new())
                .jacobian_forward(|inputs: Vec<_>| Ok(inputs))
                .unwrap_err(),
            DifferentiationError::EmptyInput,
        );

        let integer = Array::from_elements::<i32>(ArrayType::scalar(DataType::I32), &[2]).unwrap();
        assert_eq!(
            context.differentiate_at(integer).jacobian_forward(|x| Ok(x)).unwrap_err(),
            DifferentiationError::NonDifferentiableParameter {
                transform: DerivativeTransform::JacobianForward,
                role: DifferentiationParameterRole::Input,
                path: "$".to_string(),
                r#type: "i32[]".to_string(),
            },
        );

        let complex = Array::scalar(ComplexNumber::new(2.0f32, 0.0)).unwrap();
        assert_eq!(
            context.differentiate_at(complex).jacobian_forward(Ok).unwrap_err(),
            DifferentiationError::ComplexParameter {
                transform: DerivativeTransform::JacobianForward,
                role: DifferentiationParameterRole::Input,
                path: "$".to_string(),
                r#type: "c64[]".to_string(),
            },
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap())
                .holomorphic()
                .in_context(&context)
                .jacobian_reverse(Ok)
                .unwrap_err(),
            DifferentiationError::NonComplexParameter {
                transform: DerivativeTransform::JacobianReverse,
                role: DifferentiationParameterRole::Input,
                path: "$".to_string(),
                r#type: "f64[]".to_string(),
            },
        );

        let complex_output_error = context
            .differentiate_at(Array::scalar(2.0).unwrap())
            .jacobian_reverse(|input| {
                Ok(input.context().lift(Array::scalar(ComplexNumber::new(1.0f32, 0.0)).unwrap())?)
            })
            .unwrap_err();
        assert_eq!(
            complex_output_error,
            DifferentiationError::ComplexParameter {
                transform: DerivativeTransform::JacobianReverse,
                role: DifferentiationParameterRole::Output,
                path: "$".to_string(),
                r#type: "c64[]".to_string(),
            },
        );

        let dynamic_type = ArrayType::new(
            F64,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded()))]),
        );
        let dynamic = Array::new_unchecked(dynamic_type.clone(), Arc::new(1.0f64.to_le_bytes().to_vec()));
        assert_eq!(
            context.differentiate_at(dynamic).jacobian_forward(|x| Ok(x)).unwrap_err(),
            DifferentiationError::NonFiniteCoordinateSpace {
                transform: DerivativeTransform::JacobianForward,
                role: DifferentiationParameterRole::Input,
                path: "$".to_string(),
                r#type: "f64[dynamic]".to_string(),
            },
        );
        assert_eq!(
            context
                .differentiate_at(Array::scalar(1.0).unwrap())
                .jacobian_forward(|input| Ok(input
                    .context()
                    .lift(Array::new_unchecked(dynamic_type.clone(), Arc::new(1.0f64.to_le_bytes().to_vec())))?))
                .unwrap_err(),
            DifferentiationError::NonFiniteCoordinateSpace {
                transform: DerivativeTransform::JacobianForward,
                role: DifferentiationParameterRole::Output,
                path: "$".to_string(),
                r#type: "f64[dynamic]".to_string(),
            },
        );

        let dynamic = Array::new_unchecked(dynamic_type, Arc::new(1.0f64.to_le_bytes().to_vec()));
        assert_eq!(
            context
                .differentiate_at(dynamic)
                .jacobian_reverse(|input| Ok(input.context().lift(Array::scalar(1.0).unwrap())?))
                .unwrap_err(),
            DifferentiationError::NonFiniteCoordinateSpace {
                transform: DerivativeTransform::JacobianReverse,
                role: DifferentiationParameterRole::Input,
                path: "$".to_string(),
                r#type: "f64[dynamic]".to_string(),
            },
        );
    }

    #[test]
    fn test_jacobian_forward_nested_in_batch() {
        let jacobian = batch(
            |input| {
                let context = input.context().clone();
                context
                    .differentiate_at(input)
                    .jacobian_forward(|value| Ok(value.clone() * value))
                    .map_err(|error| ProgramError::MalformedProgram(error.to_string()))
            },
            Array::vector(vec![1.0, 2.0, 3.0]).unwrap(),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(jacobian.iter_blocks().next().unwrap().value().to_f64s(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_jacobian_reverse_nested_in_batch() {
        let jacobian = batch(
            |input| {
                let context = input.context().clone();
                context
                    .differentiate_at(input)
                    .jacobian_reverse(|value| Ok(value.clone() * value))
                    .map_err(|error| ProgramError::MalformedProgram(error.to_string()))
            },
            Array::vector(vec![1.0, 2.0, 3.0]).unwrap(),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(jacobian.iter_blocks().next().unwrap().value().to_f64s(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_jacobian_forward_nested_in_jacobian_reverse() {
        // For f(x) = x², the forward Jacobian is f′(x) = 2x. Differentiating that materialized Jacobian with a
        // reverse Jacobian computes the second derivative f″(x) = 2.
        let derivative = differentiate_at(Array::scalar(3.0).unwrap())
            .jacobian_reverse(|input| {
                let context = input.context().clone();
                let jacobian = context
                    .differentiate_at(input)
                    .jacobian_forward(|value| Ok(value.clone() * value))
                    .map_err(|error| ProgramError::MalformedProgram(error.to_string()))?;
                Ok(jacobian.into_values().remove(0))
            })
            .unwrap();
        assert_abs_diff_eq!(derivative.iter_blocks().next().unwrap().value().to_f64s()[0], 2.0, epsilon = 1e-9);
    }

    #[test]
    fn test_jacobian_reverse_nested_in_jacobian_forward() {
        // This is the opposite nesting order from the one in `test_jacobian_forward_nested_in_jacobian_reverse`. The
        // reverse Jacobian materializes 2x, and the enclosing forward Jacobian differentiates that primitive-only basis
        // replay to obtain 2.
        let derivative = differentiate_at(Array::scalar(3.0).unwrap())
            .jacobian_forward(|input| {
                let context = input.context().clone();
                let jacobian = context
                    .differentiate_at(input)
                    .jacobian_reverse(|value| Ok(value.clone() * value))
                    .map_err(|error| ProgramError::MalformedProgram(error.to_string()))?;
                Ok(jacobian.into_values().remove(0))
            })
            .unwrap();
        assert_abs_diff_eq!(derivative.iter_blocks().next().unwrap().value().to_f64s()[0], 2.0, epsilon = 1e-9);
    }

    #[test]
    fn test_staged_jacobian_scopes_its_coordinate_basis() {
        // Tracing a Jacobian stages its internal jvp/batching pipeline into the enclosing program, so the coordinate
        // basis primitives must still be recognizable there. The rank-two input makes each basis rank greater than one
        // and the second input carries a nonzero coordinate offset, which is the case that adds the offsetting
        // `constant`/`broadcast`/`add` primitives to the basis construction.
        let matrix_type = ArrayType::new(F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));

        // Partitions a staged program's instructions into the ones carrying the framework-owned coordinate-basis
        // scopes and the remaining ones, returning both operation-name sequences in program order. The remaining
        // instructions must all carry unknown provenance, because the user computation, its batched derivative
        // replay, and the block extraction are staged outside any framework scope.
        let coordinate_basis = Provenance::scope(
            ProvenanceScope::new("ryft"),
            Provenance::scope(
                ProvenanceScope::new("differentiation"),
                Provenance::scope(ProvenanceScope::new("coordinate_basis"), Provenance::unknown()),
            ),
        );
        let partition = |program: &Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>| {
            let (basis, other): (Vec<_>, Vec<_>) =
                program.instructions().iter().partition(|instruction| instruction.provenance() == &coordinate_basis);
            assert!(other.iter().all(|instruction| instruction.provenance().is_unknown()));
            let names = |instructions: Vec<&Instruction<ArrayOperation<Array>>>| {
                instructions.iter().map(|instruction| instruction.operation().name()).collect::<Vec<_>>().join(" ")
            };
            (names(basis), names(other))
        };

        // Forward mode builds one basis per input, so the second input's basis carries the nonzero coordinate offset
        // that adds the `constant`/`broadcast`/`add` primitives.
        let (_, forward) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<_>| {
                let context = inputs[0].context().clone();
                let jacobian = context
                    .differentiate_at(inputs)
                    .jacobian_forward(|values: Vec<_>| Ok(vec![values[0].clone() * values[1].clone()]))
                    .map_err(|error| ProgramError::MalformedProgram(error.to_string()))?;
                Ok(jacobian.into_values())
            },
            vec![matrix_type.clone(), matrix_type.clone()],
        )
        .unwrap();
        let (basis, user) = partition(&forward);
        assert_eq!(
            basis,
            "iota iota compare zero one select reshape \
             iota iota constant broadcast add compare zero one select reshape",
        );
        assert_eq!(user, "mul broadcast mul broadcast mul add slice reshape transpose slice reshape transpose");

        // Reverse mode builds its basis over the single output instead, so one offset-free basis is staged.
        let (_, reverse) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<_>| {
                let context = inputs[0].context().clone();
                let jacobian = context
                    .differentiate_at(inputs)
                    .jacobian_reverse(|values: Vec<_>| Ok(vec![values[0].clone() * values[1].clone()]))
                    .map_err(|error| ProgramError::MalformedProgram(error.to_string()))?;
                Ok(jacobian.into_values())
            },
            vec![matrix_type.clone(), matrix_type],
        )
        .unwrap();
        let (basis, user) = partition(&reverse);
        assert_eq!(basis, "iota iota compare zero one select reshape");
        assert_eq!(user, "mul broadcast mul broadcast mul reshape reshape");
    }

    #[test]
    fn test_coordinate_prefix_offsets() {
        let input_types =
            (ArrayType::new(F32, Shape::new(vec![Dimension::Static(usize::MAX)])), ArrayType::scalar(F32));
        assert_eq!(
            coordinate_prefix_offsets::<EagerContext<Array, ArrayOperation<Array>>, _>(
                &input_types,
                DerivativeTransform::JacobianForward,
                DifferentiationParameterRole::Input,
            )
            .unwrap_err(),
            DifferentiationError::CoordinateCountOverflow {
                transform: DerivativeTransform::JacobianForward,
                role: DifferentiationParameterRole::Input,
                path: "$.1".to_string(),
                r#type: "f32[]".to_string(),
            },
        );
        let empty_input_types = ArrayType::new(
            F32,
            Shape::new(vec![Dimension::Static(usize::MAX), Dimension::Static(usize::MAX), Dimension::Static(0)]),
        );
        assert_eq!(
            coordinate_prefix_offsets::<EagerContext<Array, ArrayOperation<Array>>, _>(
                &empty_input_types,
                DerivativeTransform::JacobianForward,
                DifferentiationParameterRole::Input,
            )
            .unwrap(),
            vec![0, 0],
        );
    }
}
