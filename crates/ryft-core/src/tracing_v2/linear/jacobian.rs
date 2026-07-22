use std::cell::RefCell;
use std::fmt::Debug;
use std::marker::PhantomData;

use ryft_macros::Parameterized;

use crate::contexts::{Context, Domain};
use crate::differentiation::forward::{DifferentiableOperation, ForwardModeDifferentiate, LinearizationTracer};
use crate::differentiation::reverse::{ReverseModeDifferentiate, TransposableOperation};
use crate::differentiation::{
    DenseDifferentiableType, DerivativeTransform, DifferentiableType, DifferentiationError,
    DifferentiationParameterRole,
};
use crate::macros::check_count;
use crate::operations::constants::ZeroOperation;
use crate::operations::math::AddOperation;
use crate::parameters::{Parameter, ParameterPath, Parameterized, ParameterizedFamily};
use crate::partial::{PartialEvaluationContext, PartialValue, PartiallyEvaluatableOperation};
use crate::programs::ProgramError;
use crate::programs::types::{Type, Typed};
use crate::programs::values::Value;
use crate::tracing::TracingContext;

// TODO(eaplatanios): Should we move this?
use super::DenseDifferentiate;

/// Jacobian of a function, represented as the Cartesian product of its output and input [`Parameter`] leaves. `I`
/// and `O` retain the input and output [`Type`] trees. Derivative values are stored in deterministic output-major /
/// input-minor order and remain [`Parameter`]s so that the complete Jacobian can cross tracing and compilation
/// boundaries as well as participate in higher-order transforms. The physical representation of a block is defined by
/// [`DenseDifferentiableType`]. For [`ArrayType`](crate::ArrayType), the block for an output leaf with shape `O` and
/// an input leaf with shape `I` has shape `O` concatenated with `I`.
#[derive(Parameterized, Clone, Debug)]
pub struct Jacobian<T: Type, V: Parameter, I: Clone + Parameterized<T>, O: Clone + Parameterized<T>> {
    /// [`Type`] of the differentiated inputs.
    input_type: I,

    /// [`Type`] of the differentiated outputs.
    output_type: O,

    /// Derivative values in output-major/input-minor order.
    values: Vec<V>,

    /// [`PhantomData`] marker for `T`, needed because the input and output fields use `T` only through their bounds.
    _type: PhantomData<fn() -> T>,
}

impl<T: Type, V: Parameter, I: Clone + Parameterized<T>, O: Clone + Parameterized<T>> Jacobian<T, V, I, O> {
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
            };
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

// TODO(eaplatanios): Review from here onwards.

fn coordinate_offsets<C: Context, S: Parameterized<C::Type>>(
    types: &S,
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

pub(super) fn jacfwd_in<C, F, I, O>(
    context: &C,
    function: F,
    primals: I,
    holomorphic: bool,
) -> Result<Jacobian<C::Type, C::Value, I::To<C::Type>, O::To<C::Type>>, DifferentiationError>
where
    C: Context,
    C::Type: DenseDifferentiableType<C>,
    I: Parameterized<
            C::Value,
            To<C::Value> = I,
            Family: ParameterizedFamily<LinearizationTracer<C>> + ParameterizedFamily<C::Type>,
        >,
    I::ParameterStructure: Debug + PartialEq,
    O: Parameterized<LinearizationTracer<C>, Family: ParameterizedFamily<C::Value> + ParameterizedFamily<C::Type>>,
    O::To<C::Value>: Parameterized<C::Value, To<C::Value> = O::To<C::Value>>,
    F: FnOnce(I::To<LinearizationTracer<C>>) -> Result<O, ProgramError>,
    C::Operation: PartiallyEvaluatableOperation<C>
        + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
        + From<ZeroOperation<C::Type>>,
    I::To<C::Type>: Clone + Parameterized<C::Type>,
    O::To<C::Type>: Clone + Parameterized<C::Type>,
{
    let transform = DerivativeTransform::JacobianForward;
    let input_structure = primals.parameter_structure();
    let input_values = primals.into_parameters().collect::<Vec<_>>();
    let input_types = I::To::<C::Type>::from_parameters(
        input_structure.clone(),
        input_values.iter().map(|value| value.r#type().into_owned()),
    )?;
    JacobianMode::Forward.validate_types(&input_types, holomorphic, DifferentiationParameterRole::Input)?;
    let primals = I::from_parameters(input_structure, input_values)?;
    let (output, pushforward) = context.linearize(function, primals)?;
    let output_structure = output.parameter_structure();
    let output_values = output.into_parameters().collect::<Vec<_>>();
    let output_types = O::To::<C::Type>::from_parameters(
        output_structure,
        output_values.iter().map(|value| value.r#type().into_owned()),
    )?;
    JacobianMode::Forward.validate_types(&output_types, holomorphic, DifferentiationParameterRole::Output)?;

    let input_offsets = coordinate_offsets::<C, _>(&input_types, transform, DifferentiationParameterRole::Input)?;
    let _ = coordinate_offsets::<C, _>(&output_types, transform, DifferentiationParameterRole::Output)?;
    let batch_size = input_offsets.last().copied().unwrap();
    let (program, residuals) = pushforward.into_parts();
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
            "pushforward program consumes {tangent_input_count} tangent inputs but the differentiated input has {} \
             leaves",
            input_types.parameter_count(),
        ))
        .into());
    }
    for (index, (program_input_type, (path, input_type))) in
        program_input_types[..tangent_input_count].iter().zip(input_types.named_parameters()).enumerate()
    {
        let tangent_type = input_type.tangent();
        if tangent_type.is_zero_space() {
            return Err(DifferentiationError::NonDifferentiableParameter {
                transform,
                role: DifferentiationParameterRole::Input,
                path: path.to_string(),
                r#type: input_type.to_string(),
            });
        }
        if program_input_type != &tangent_type {
            return Err(ProgramError::MalformedProgram(format!(
                "pushforward tangent input {index} has type {program_input_type} but the differentiated input leaf \
                 has tangent type {tangent_type}",
            ))
            .into());
        }
    }
    let mut packed_inputs = input_types
        .named_parameters()
        .enumerate()
        .map(|(index, (path, r#type))| {
            let tangent_type = r#type.tangent();
            if tangent_type.is_zero_space() {
                return Err(DifferentiationError::NonDifferentiableParameter {
                    transform,
                    role: DifferentiationParameterRole::Input,
                    path: path.to_string(),
                    r#type: r#type.to_string(),
                });
            }
            C::Type::coordinate_basis(context, r#type, &tangent_type, input_offsets[index], batch_size)
        })
        .collect::<Result<Vec<_>, _>>()?;
    packed_inputs.extend(residuals.into_iter().map(C::Type::replicated));
    let packed_outputs =
        C::Type::replay_derivative_region(context, program.entry_region_ref(), batch_size, packed_inputs)?;
    check_count!("output", packed_outputs, output_types.parameter_count(), ProgramError);

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
    Ok(Jacobian::new(input_types, output_types, values)?)
}

pub(super) fn jacrev_in<C, F, I, O>(
    context: &C,
    function: F,
    primals: I,
    holomorphic: bool,
) -> Result<Jacobian<C::Type, C::Value, I::To<C::Type>, O::To<C::Type>>, DifferentiationError>
where
    C: Context,
    C::Type: DenseDifferentiableType<C>,
    I: Parameterized<
            C::Value,
            To<C::Value> = I,
            Family: ParameterizedFamily<LinearizationTracer<C>> + ParameterizedFamily<C::Type>,
        >,
    I::ParameterStructure: Debug + PartialEq,
    O: Parameterized<LinearizationTracer<C>, Family: ParameterizedFamily<C::Value> + ParameterizedFamily<C::Type>>,
    O::To<C::Value>: Parameterized<C::Value, To<C::Value> = O::To<C::Value>>,
    F: FnOnce(I::To<LinearizationTracer<C>>) -> Result<O, ProgramError>,
    C::Operation: PartiallyEvaluatableOperation<C>
        + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
        + DifferentiableOperation<PartialEvaluationContext<C>>
        + TransposableOperation<C::Constant, C::Operation>
        + From<ZeroOperation<C::Type>>
        + From<AddOperation>,
    I::To<C::Type>: Clone + Parameterized<C::Type>,
    O::To<C::Type>: Clone + Parameterized<C::Type>,
{
    let transform = DerivativeTransform::JacobianReverse;
    let input_structure = primals.parameter_structure();
    let input_values = primals.into_parameters().collect::<Vec<_>>();
    let input_types = I::To::<C::Type>::from_parameters(
        input_structure.clone(),
        input_values.iter().map(|value| value.r#type().into_owned()),
    )?;
    JacobianMode::Reverse.validate_types(&input_types, holomorphic, DifferentiationParameterRole::Input)?;
    let primals = I::from_parameters(input_structure, input_values)?;
    let (output, pullback) = context.vjp(function, primals)?;
    let output_structure = output.parameter_structure();
    let output_values = output.into_parameters().collect::<Vec<_>>();
    let output_types = O::To::<C::Type>::from_parameters(
        output_structure,
        output_values.iter().map(|value| value.r#type().into_owned()),
    )?;
    JacobianMode::Reverse.validate_types(&output_types, holomorphic, DifferentiationParameterRole::Output)?;

    let _ = coordinate_offsets::<C, _>(&input_types, transform, DifferentiationParameterRole::Input)?;
    let output_offsets = coordinate_offsets::<C, _>(&output_types, transform, DifferentiationParameterRole::Output)?;
    let batch_size = output_offsets.last().copied().unwrap();
    let (program, residuals) = pullback.into_parts();
    let program_input_types = program.input_types();
    let cotangent_input_count = program_input_types.len().checked_sub(residuals.len()).ok_or_else(|| {
        ProgramError::MalformedProgram(format!(
            "pullback program consumes {} inputs which is fewer than its {} residuals",
            program_input_types.len(),
            residuals.len(),
        ))
    })?;
    if cotangent_input_count != output_types.parameter_count() {
        return Err(ProgramError::MalformedProgram(format!(
            "pullback program consumes {cotangent_input_count} cotangent inputs but the differentiated output has {} \
             leaves",
            output_types.parameter_count(),
        ))
        .into());
    }
    let mut packed_inputs = output_types
        .named_parameters()
        .enumerate()
        .map(|(index, (path, r#type))| {
            let cotangent_type = r#type.cotangent();
            if cotangent_type.is_zero_space() {
                return Err(DifferentiationError::NonDifferentiableParameter {
                    transform,
                    role: DifferentiationParameterRole::Output,
                    path: path.to_string(),
                    r#type: r#type.to_string(),
                });
            }
            if program_input_types[index] != cotangent_type {
                return Err(ProgramError::MalformedProgram(format!(
                    "pullback cotangent input {index} has type {} but output leaf {path} has cotangent type \
                     {cotangent_type}",
                    program_input_types[index],
                ))
                .into());
            }
            C::Type::coordinate_basis(context, r#type, &cotangent_type, output_offsets[index], batch_size)
        })
        .collect::<Result<Vec<_>, _>>()?;
    packed_inputs.extend(residuals.into_iter().map(C::Type::replicated));
    let packed_outputs =
        C::Type::replay_derivative_region(context, program.entry_region_ref(), batch_size, packed_inputs)?;
    check_count!("output", packed_outputs, input_types.parameter_count(), ProgramError);

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
    Ok(Jacobian::new(input_types, output_types, values)?)
}

pub(super) fn jacfwd_with_aux_in<C, F, I, O, A>(
    context: &C,
    function: F,
    primals: I,
    holomorphic: bool,
) -> Result<(Jacobian<C::Type, C::Value, I::To<C::Type>, O::To<C::Type>>, A::To<C::Value>), DifferentiationError>
where
    C: Context,
    C::Type: DenseDifferentiableType<C>,
    I: Parameterized<
            C::Value,
            To<C::Value> = I,
            Family: ParameterizedFamily<LinearizationTracer<C>> + ParameterizedFamily<C::Type>,
            ParameterStructure: Debug + PartialEq,
        >,
    I::To<C::Type>: Clone + Parameterized<C::Type>,
    O: Parameterized<LinearizationTracer<C>, Family: ParameterizedFamily<C::Value> + ParameterizedFamily<C::Type>>,
    O::To<C::Value>: Parameterized<C::Value, To<C::Value> = O::To<C::Value>>,
    O::To<C::Type>: Clone + Parameterized<C::Type>,
    A: Parameterized<LinearizationTracer<C>, Family: ParameterizedFamily<C::Value>>,
    F: FnOnce(I::To<LinearizationTracer<C>>) -> Result<(O, A), ProgramError>,
    C::Operation: PartiallyEvaluatableOperation<C>
        + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
        + From<ZeroOperation<C::Type>>,
{
    let auxiliary = RefCell::new(None);
    let jacobian = jacfwd_in(
        context,
        |input| {
            let (output, value) = function(input)?;
            auxiliary.replace(Some(materialize_auxiliary(value).map_err(ProgramError::from)?));
            Ok(output)
        },
        primals,
        holomorphic,
    )?;
    let auxiliary = auxiliary
        .into_inner()
        .ok_or_else(|| ProgramError::MalformedProgram("jacfwd_with_aux did not evaluate its function".to_string()))?;
    Ok((jacobian, auxiliary))
}

pub(super) fn jacrev_with_aux_in<C, F, I, O, A>(
    context: &C,
    function: F,
    primals: I,
    holomorphic: bool,
) -> Result<(Jacobian<C::Type, C::Value, I::To<C::Type>, O::To<C::Type>>, A::To<C::Value>), DifferentiationError>
where
    C: Context,
    C::Type: DenseDifferentiableType<C>,
    I: Parameterized<
            C::Value,
            To<C::Value> = I,
            Family: ParameterizedFamily<LinearizationTracer<C>> + ParameterizedFamily<C::Type>,
            ParameterStructure: Debug + PartialEq,
        >,
    I::To<C::Type>: Clone + Parameterized<C::Type>,
    O: Parameterized<LinearizationTracer<C>, Family: ParameterizedFamily<C::Value> + ParameterizedFamily<C::Type>>,
    O::To<C::Value>: Parameterized<C::Value, To<C::Value> = O::To<C::Value>>,
    O::To<C::Type>: Clone + Parameterized<C::Type>,
    A: Parameterized<LinearizationTracer<C>, Family: ParameterizedFamily<C::Value>>,
    F: FnOnce(I::To<LinearizationTracer<C>>) -> Result<(O, A), ProgramError>,
    C::Operation: PartiallyEvaluatableOperation<C>
        + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
        + DifferentiableOperation<PartialEvaluationContext<C>>
        + TransposableOperation<C::Constant, C::Operation>
        + From<ZeroOperation<C::Type>>
        + From<AddOperation>,
{
    let auxiliary = RefCell::new(None);
    let jacobian = jacrev_in(
        context,
        |input| {
            let (output, value) = function(input)?;
            auxiliary.replace(Some(materialize_auxiliary(value).map_err(ProgramError::from)?));
            Ok(output)
        },
        primals,
        holomorphic,
    )?;
    let auxiliary = auxiliary
        .into_inner()
        .ok_or_else(|| ProgramError::MalformedProgram("jacrev_with_aux did not evaluate its function".to_string()))?;
    Ok((jacobian, auxiliary))
}

fn materialize_auxiliary<C, A>(auxiliary: A) -> Result<A::To<C::Value>, DifferentiationError>
where
    C: Context,
    A: Parameterized<LinearizationTracer<C>, Family: ParameterizedFamily<C::Value>>,
    C::Operation:
        PartiallyEvaluatableOperation<C> + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>,
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

/// Materializes the complete forward-mode Jacobian, recovering the active context from `primals`.
pub fn jacfwd<V, F, I, O>(
    function: F,
    primals: I,
) -> Result<Jacobian<V::Type, V, I::To<V::Type>, O::To<V::Type>>, DifferentiationError>
where
    V: Value,
    V::ExecutionDomain: Context<Type = V::Type, Value = V>,
    V::Type: DenseDifferentiableType<V::ExecutionDomain>,
    I: Parameterized<
            V,
            To<V> = I,
            Family: ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>> + ParameterizedFamily<V::Type>,
            ParameterStructure: Debug + PartialEq,
        >,
    I::To<V::Type>: Clone + Parameterized<V::Type>,
    O: Parameterized<
            LinearizationTracer<V::ExecutionDomain>,
            Family: ParameterizedFamily<V> + ParameterizedFamily<V::Type>,
        >,
    O::To<V>: Parameterized<V, To<V> = O::To<V>>,
    O::To<V::Type>: Clone + Parameterized<V::Type>,
    F: FnOnce(I::To<LinearizationTracer<V::ExecutionDomain>>) -> Result<O, ProgramError>,
    <V::ExecutionDomain as Domain>::Operation: PartiallyEvaluatableOperation<V::ExecutionDomain>
        + PartiallyEvaluatableOperation<
            TracingContext<<V::ExecutionDomain as Domain>::Constant, <V::ExecutionDomain as Domain>::Operation>,
        > + From<ZeroOperation<V::Type>>,
{
    let Some(context) = primals.parameters().next().map(Value::execution_domain) else {
        return Err(DifferentiationError::EmptyInput);
    };
    context.jacfwd(function, primals)
}

/// Materializes the complete reverse-mode Jacobian, recovering the active context from `primals`.
pub fn jacrev<V, F, I, O>(
    function: F,
    primals: I,
) -> Result<Jacobian<V::Type, V, I::To<V::Type>, O::To<V::Type>>, DifferentiationError>
where
    V: Value,
    V::ExecutionDomain: Context<Type = V::Type, Value = V>,
    V::Type: DenseDifferentiableType<V::ExecutionDomain>,
    I: Parameterized<
            V,
            To<V> = I,
            Family: ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>> + ParameterizedFamily<V::Type>,
            ParameterStructure: Debug + PartialEq,
        >,
    I::To<V::Type>: Clone + Parameterized<V::Type>,
    O: Parameterized<
            LinearizationTracer<V::ExecutionDomain>,
            Family: ParameterizedFamily<V> + ParameterizedFamily<V::Type>,
        >,
    O::To<V>: Parameterized<V, To<V> = O::To<V>>,
    O::To<V::Type>: Clone + Parameterized<V::Type>,
    F: FnOnce(I::To<LinearizationTracer<V::ExecutionDomain>>) -> Result<O, ProgramError>,
    <V::ExecutionDomain as Domain>::Operation: PartiallyEvaluatableOperation<V::ExecutionDomain>
        + PartiallyEvaluatableOperation<
            TracingContext<<V::ExecutionDomain as Domain>::Constant, <V::ExecutionDomain as Domain>::Operation>,
        > + DifferentiableOperation<PartialEvaluationContext<V::ExecutionDomain>>
        + TransposableOperation<<V::ExecutionDomain as Domain>::Constant, <V::ExecutionDomain as Domain>::Operation>
        + From<ZeroOperation<V::Type>>
        + From<AddOperation>,
{
    let Some(context) = primals.parameters().next().map(Value::execution_domain) else {
        return Err(DifferentiationError::EmptyInput);
    };
    context.jacrev(function, primals)
}

/// Materializes a holomorphic forward-mode Jacobian, recovering the active context from `primals`.
pub fn jacfwd_holomorphic<V, F, I, O>(
    function: F,
    primals: I,
) -> Result<Jacobian<V::Type, V, I::To<V::Type>, O::To<V::Type>>, DifferentiationError>
where
    V: Value,
    V::ExecutionDomain: Context<Type = V::Type, Value = V>,
    V::Type: DenseDifferentiableType<V::ExecutionDomain>,
    I: Parameterized<
            V,
            To<V> = I,
            Family: ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>> + ParameterizedFamily<V::Type>,
            ParameterStructure: Debug + PartialEq,
        >,
    I::To<V::Type>: Clone + Parameterized<V::Type>,
    O: Parameterized<
            LinearizationTracer<V::ExecutionDomain>,
            Family: ParameterizedFamily<V> + ParameterizedFamily<V::Type>,
        >,
    O::To<V>: Parameterized<V, To<V> = O::To<V>>,
    O::To<V::Type>: Clone + Parameterized<V::Type>,
    F: FnOnce(I::To<LinearizationTracer<V::ExecutionDomain>>) -> Result<O, ProgramError>,
    <V::ExecutionDomain as Domain>::Operation: PartiallyEvaluatableOperation<V::ExecutionDomain>
        + PartiallyEvaluatableOperation<
            TracingContext<<V::ExecutionDomain as Domain>::Constant, <V::ExecutionDomain as Domain>::Operation>,
        > + From<ZeroOperation<V::Type>>,
{
    let Some(context) = primals.parameters().next().map(Value::execution_domain) else {
        return Err(DifferentiationError::EmptyInput);
    };
    context.jacfwd_holomorphic(function, primals)
}

/// Materializes a holomorphic reverse-mode Jacobian, recovering the active context from `primals`.
pub fn jacrev_holomorphic<V, F, I, O>(
    function: F,
    primals: I,
) -> Result<Jacobian<V::Type, V, I::To<V::Type>, O::To<V::Type>>, DifferentiationError>
where
    V: Value,
    V::ExecutionDomain: Context<Type = V::Type, Value = V>,
    V::Type: DenseDifferentiableType<V::ExecutionDomain>,
    I: Parameterized<
            V,
            To<V> = I,
            Family: ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>> + ParameterizedFamily<V::Type>,
            ParameterStructure: Debug + PartialEq,
        >,
    I::To<V::Type>: Clone + Parameterized<V::Type>,
    O: Parameterized<
            LinearizationTracer<V::ExecutionDomain>,
            Family: ParameterizedFamily<V> + ParameterizedFamily<V::Type>,
        >,
    O::To<V>: Parameterized<V, To<V> = O::To<V>>,
    O::To<V::Type>: Clone + Parameterized<V::Type>,
    F: FnOnce(I::To<LinearizationTracer<V::ExecutionDomain>>) -> Result<O, ProgramError>,
    <V::ExecutionDomain as Domain>::Operation: PartiallyEvaluatableOperation<V::ExecutionDomain>
        + PartiallyEvaluatableOperation<
            TracingContext<<V::ExecutionDomain as Domain>::Constant, <V::ExecutionDomain as Domain>::Operation>,
        > + DifferentiableOperation<PartialEvaluationContext<V::ExecutionDomain>>
        + TransposableOperation<<V::ExecutionDomain as Domain>::Constant, <V::ExecutionDomain as Domain>::Operation>
        + From<ZeroOperation<V::Type>>
        + From<AddOperation>,
{
    let Some(context) = primals.parameters().next().map(Value::execution_domain) else {
        return Err(DifferentiationError::EmptyInput);
    };
    context.jacrev_holomorphic(function, primals)
}

macro_rules! define_forward_jacobian_with_aux {
    ($name:ident, $method:ident, $documentation:literal) => {
        #[doc = $documentation]
        pub fn $name<V, F, I, O, A>(
            function: F,
            primals: I,
        ) -> Result<(Jacobian<V::Type, V, I::To<V::Type>, O::To<V::Type>>, A::To<V>), DifferentiationError>
        where
            V: Value,
            V::ExecutionDomain: Context<Type = V::Type, Value = V>,
            V::Type: DenseDifferentiableType<V::ExecutionDomain>,
            I: Parameterized<
                    V,
                    To<V> = I,
                    Family: ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>> + ParameterizedFamily<V::Type>,
                    ParameterStructure: Debug + PartialEq,
                >,
            I::To<V::Type>: Clone + Parameterized<V::Type>,
            O: Parameterized<
                    LinearizationTracer<V::ExecutionDomain>,
                    Family: ParameterizedFamily<V> + ParameterizedFamily<V::Type>,
                >,
            O::To<V>: Parameterized<V, To<V> = O::To<V>>,
            O::To<V::Type>: Clone + Parameterized<V::Type>,
            A: Parameterized<LinearizationTracer<V::ExecutionDomain>, Family: ParameterizedFamily<V>>,
            F: FnOnce(I::To<LinearizationTracer<V::ExecutionDomain>>) -> Result<(O, A), ProgramError>,
            <V::ExecutionDomain as Domain>::Operation: PartiallyEvaluatableOperation<V::ExecutionDomain>
                + PartiallyEvaluatableOperation<
                    TracingContext<<V::ExecutionDomain as Domain>::Constant, <V::ExecutionDomain as Domain>::Operation>,
                > + From<ZeroOperation<V::Type>>,
        {
            let Some(context) = primals.parameters().next().map(Value::execution_domain) else {
                return Err(DifferentiationError::EmptyInput);
            };
            context.$method(function, primals)
        }
    };
}

define_forward_jacobian_with_aux!(
    jacfwd_with_aux,
    jacfwd_with_aux,
    "Materializes a forward-mode Jacobian and auxiliary outputs, recovering the active context from `primals`."
);
define_forward_jacobian_with_aux!(
    jacfwd_holomorphic_with_aux,
    jacfwd_holomorphic_with_aux,
    "Materializes a holomorphic forward-mode Jacobian and auxiliary outputs, recovering the active context from \
     `primals`."
);

macro_rules! define_reverse_jacobian_with_aux {
    ($name:ident, $method:ident, $documentation:literal) => {
        #[doc = $documentation]
        pub fn $name<V, F, I, O, A>(
            function: F,
            primals: I,
        ) -> Result<(Jacobian<V::Type, V, I::To<V::Type>, O::To<V::Type>>, A::To<V>), DifferentiationError>
        where
            V: Value,
            V::ExecutionDomain: Context<Type = V::Type, Value = V>,
            V::Type: DenseDifferentiableType<V::ExecutionDomain>,
            I: Parameterized<
                    V,
                    To<V> = I,
                    Family: ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>> + ParameterizedFamily<V::Type>,
                    ParameterStructure: Debug + PartialEq,
                >,
            I::To<V::Type>: Clone + Parameterized<V::Type>,
            O: Parameterized<
                    LinearizationTracer<V::ExecutionDomain>,
                    Family: ParameterizedFamily<V> + ParameterizedFamily<V::Type>,
                >,
            O::To<V>: Parameterized<V, To<V> = O::To<V>>,
            O::To<V::Type>: Clone + Parameterized<V::Type>,
            A: Parameterized<LinearizationTracer<V::ExecutionDomain>, Family: ParameterizedFamily<V>>,
            F: FnOnce(I::To<LinearizationTracer<V::ExecutionDomain>>) -> Result<(O, A), ProgramError>,
            <V::ExecutionDomain as Domain>::Operation: PartiallyEvaluatableOperation<V::ExecutionDomain>
                + PartiallyEvaluatableOperation<
                    TracingContext<<V::ExecutionDomain as Domain>::Constant, <V::ExecutionDomain as Domain>::Operation>,
                > + DifferentiableOperation<PartialEvaluationContext<V::ExecutionDomain>>
                + TransposableOperation<
                    <V::ExecutionDomain as Domain>::Constant,
                    <V::ExecutionDomain as Domain>::Operation,
                > + From<ZeroOperation<V::Type>>
                + From<AddOperation>,
        {
            let Some(context) = primals.parameters().next().map(Value::execution_domain) else {
                return Err(DifferentiationError::EmptyInput);
            };
            context.$method(function, primals)
        }
    };
}

define_reverse_jacobian_with_aux!(
    jacrev_with_aux,
    jacrev_with_aux,
    "Materializes a reverse-mode Jacobian and auxiliary outputs, recovering the active context from `primals`."
);
define_reverse_jacobian_with_aux!(
    jacrev_holomorphic_with_aux,
    jacrev_holomorphic_with_aux,
    "Materializes a holomorphic reverse-mode Jacobian and auxiliary outputs, recovering the active context from \
     `primals`."
);

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::contexts::{Context, EagerContext};
    use crate::differentiation::{DerivativeTransform, DifferentiationError, DifferentiationParameterRole};
    use crate::operations::compare::{Compare, ComparisonDirection};
    use crate::operations::constants::ZeroLike;
    use crate::operations::control_flow::Select;
    use crate::operations::math::Add;
    use crate::parameters::{ParameterPath, Parameterized};
    use crate::programs::types::Typed;
    use crate::programs::values::Value;
    use crate::types::DataType::{F32, F64};
    use crate::types::{ArrayType, DataType, Shape, Size};

    use super::{Jacobian, coordinate_offsets, jacfwd, jacrev};

    /// Returns `2x` for positive `x` and `3x` otherwise, expressed generically so both dense Jacobian modes exercise
    /// comparison, selection, and arithmetic while constructing their coordinate-basis replays.
    fn piecewise_select<V>(x: V) -> V
    where
        V: Value<Type = ArrayType>,
        V::DispatchDomain: Context<Type = ArrayType, Constant = Array, Operation = ArrayOperation<Array>>,
    {
        let condition = x.compare(&x.zero_like(), ComparisonDirection::GreaterThan).unwrap();
        let doubled = x.add(&x).unwrap();
        let tripled = doubled.add(&x).unwrap();
        Select::select(&condition, &doubled, &tripled).unwrap()
    }

    #[test]
    fn test_jacobian_parameterization_with_data_types() {
        let jacobian = Jacobian::new((F32, vec![F64, F32]), F64, vec![1.0_f32, 2.0, 3.0]).unwrap();
        assert_eq!(jacobian.parameter_count(), 3);
        assert_eq!(jacobian.values(), &[1.0, 2.0, 3.0]);

        let reparameterized =
            <Jacobian<DataType, f64, _, _>>::from_parameters(jacobian.parameter_structure(), [4.0, 5.0, 6.0]).unwrap();
        assert_eq!(reparameterized.input_type(), &(F32, vec![F64, F32]));
        assert_eq!(reparameterized.output_type(), &F64);
        assert_eq!(reparameterized.values(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_jacobian_blocks_with_array_types() {
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
    fn test_coordinate_offsets_reports_overflow_and_handles_empty_coordinate_spaces() {
        let input_types = (ArrayType::new(F32, Shape::new(vec![Size::Static(usize::MAX)])), ArrayType::scalar(F32));
        assert_eq!(
            coordinate_offsets::<EagerContext<Array, ArrayOperation<Array>>, _>(
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

        let empty_input_types =
            ArrayType::new(F32, Shape::new(vec![Size::Static(usize::MAX), Size::Static(usize::MAX), Size::Static(0)]));
        assert_eq!(
            coordinate_offsets::<EagerContext<Array, ArrayOperation<Array>>, _>(
                &empty_input_types,
                DerivativeTransform::JacobianForward,
                DifferentiationParameterRole::Input,
            )
            .unwrap(),
            vec![0, 0],
        );
    }

    #[test]
    fn test_jacfwd_packs_all_coordinate_directions() {
        let jacobian = jacfwd(|input| Ok(input), Array::vector(vec![1.0, 2.0, 3.0])).unwrap();

        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(
            block.value().r#type().into_owned(),
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3), Size::Static(3)])),
        );
        assert_eq!(block.value().values(), &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_select_jacfwd_computes_piecewise_derivative() {
        // Forward mode selects the branch tangent under the primal condition, giving derivative 2 for positive inputs
        // and 3 otherwise.
        let jacobian = jacfwd(|x| Ok(piecewise_select(x)), Array::scalar(2.0)).unwrap();
        assert_abs_diff_eq!(jacobian.iter_blocks().next().unwrap().value().values()[0], 2.0, epsilon = 1e-9);

        let jacobian = jacfwd(|x| Ok(piecewise_select(x)), Array::scalar(-2.0)).unwrap();
        assert_abs_diff_eq!(jacobian.iter_blocks().next().unwrap().value().values()[0], 3.0, epsilon = 1e-9);
    }

    #[test]
    fn test_select_jacrev_computes_piecewise_derivative() {
        // Reverse mode routes the output cotangent through the selected branch, giving the same piecewise derivative.
        let jacobian = jacrev(|x| Ok(piecewise_select(x)), Array::scalar(2.0)).unwrap();
        assert_abs_diff_eq!(jacobian.iter_blocks().next().unwrap().value().values()[0], 2.0, epsilon = 1e-9);

        let jacobian = jacrev(|x| Ok(piecewise_select(x)), Array::scalar(-2.0)).unwrap();
        assert_abs_diff_eq!(jacobian.iter_blocks().next().unwrap().value().values()[0], 3.0, epsilon = 1e-9);
    }

    #[test]
    fn test_select_jacrev_over_vector_masks_per_element() {
        // Per-element masking over a vector input makes the Jacobian diagonal, with entries 2 for positive inputs and
        // 3 otherwise.
        let jacobian = jacrev(|x| Ok(piecewise_select(x)), Array::vector(vec![1.0, -1.0])).unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.output_type().static_shape().unwrap().as_slice(), &[2]);
        assert_eq!(block.input_type().static_shape().unwrap().as_slice(), &[2]);
        assert_abs_diff_eq!(block.value().values()[0], 2.0, epsilon = 1e-9);
        assert_abs_diff_eq!(block.value().values()[1], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(block.value().values()[2], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(block.value().values()[3], 3.0, epsilon = 1e-9);
    }

    #[test]
    fn test_select_jacrev_unbroadcasts_mixed_precision_scalar_branches() {
        let scalar = Array::from_f64s(ArrayType::scalar(DataType::F32), vec![5.0]);
        let f32_vector_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)]));
        let vector =
            Array::from_f64s(ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)])), vec![2.0, -3.0]);

        let jacobian = jacrev(
            |(scalar, vector)| {
                let condition = vector.compare(&vector.zero_like(), ComparisonDirection::GreaterThan)?;
                Select::select(&condition, &scalar, &vector)
            },
            (scalar.clone(), vector.clone()),
        )
        .unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].value().r#type().into_owned(), f32_vector_type);
        assert_eq!(blocks[0].value().to_f64s(), vec![1.0, 0.0]);
        assert_eq!(
            blocks[1].value().r#type().into_owned(),
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2)])),
        );
        assert_eq!(blocks[1].value().to_f64s(), vec![0.0, 0.0, 0.0, 1.0]);

        let jacobian = jacrev(
            |(scalar, vector)| {
                let condition = vector.compare(&vector.zero_like(), ComparisonDirection::GreaterThan)?;
                Select::select(&condition, &vector, &scalar)
            },
            (scalar, vector),
        )
        .unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].value().r#type().into_owned(), f32_vector_type);
        assert_eq!(blocks[0].value().to_f64s(), vec![0.0, 1.0]);
        assert_eq!(
            blocks[1].value().r#type().into_owned(),
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2)])),
        );
        assert_eq!(blocks[1].value().to_f64s(), vec![1.0, 0.0, 0.0, 0.0]);
    }
}
