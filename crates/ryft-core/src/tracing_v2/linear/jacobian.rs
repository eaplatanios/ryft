use std::cell::RefCell;
use std::fmt::Debug;
use std::marker::PhantomData;

use ryft_macros::Parameterized;

use crate::batching::{ArrayBatch, BatchableOperation, BatchingContext};
use crate::contexts::{Context, Domain};
use crate::differentiation::forward::{DifferentiableOperation, ForwardModeDifferentiate, LinearizationTracer};
use crate::differentiation::reverse::{ReverseModeDifferentiate, TransposableOperation};
use crate::differentiation::types::DifferentiableType;
use crate::differentiation::{DerivativeTransform, DifferentiationError, DifferentiationParameterRole};
use crate::macros::check_count;
use crate::operations::constants::ZeroOperation;
use crate::operations::differentiation::CoordinateBasisOperation;
use crate::operations::manipulation::{Broadcast, BroadcastOperation, Reshape, Slice, Transpose, TransposeOperation};
use crate::operations::math::AddOperation;
use crate::parameters::{Parameter, ParameterPath, Parameterized, ParameterizedFamily};
use crate::partial::{PartialEvaluationContext, PartialValue, PartiallyEvaluatableOperation};
use crate::programs::ProgramError;
use crate::programs::regions::RegionRef;
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::Value;
use crate::sharding::ShardingDimension;
use crate::tracing::TracingContext;
use crate::types::{ArrayType, Shape, Size, StaticShape};

// TODO(eaplatanios): Should we move this?
use super::DenseDifferentiate;

/// Jacobian of a function, represented as the Cartesian product of its output and input [`Parameter`] leaves. `I`
/// and `O` retain the input and output [`Type`] trees. Derivative values are stored in deterministic output-major /
/// input-minor order and remain [`Parameter`]s so that the complete Jacobian can cross tracing and compilation
/// boundaries as well as participate in higher-order transforms. The physical representation of a block is defined by
/// [`DenseDifferentiableType`]. For [`ArrayType`], the block for an output leaf with shape `O` and an input leaf with
/// shape `I` has shape `O` concatenated with `I`.
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

/// A [`Type`] that is supported by dense differentiation functions that compute [`Jacobian`](crate::Jacobian)s and
/// [`Hessian`](crate::Hessian)s. [`Type`] and [`DifferentiableType`] describe individual primal and cotangent values,
/// but they do not state that a leaf (i.e., a [`Parameter`]) has a finite coordinate space, that several directions
/// can be represented by one value, or how packed replay results become public derivative blocks. Implementations
/// of this trait provide only those representation-specific operations. The Jacobian and Hessian algorithms retain
/// ownership of structure traversal, differentiation, ordering, and result construction.
pub trait DenseDifferentiableType<C: Context<Type = Self>>: DifferentiableType {
    /// Intermediate representation of one logical [`C::Value`](Domain::Value) during a packed, multi-direction
    /// derivative replay. A value is either mapped over the packed coordinate directions or replicated unchanged across
    /// them. Its logical per-direction type remains `Self`, while its physical representation may carry an additional
    /// axis that indexes those directions. [`coordinate_basis`](Self::coordinate_basis)
    /// and [`replicated`](Self::replicated) construct these values,
    /// [`replay_derivative_region`](Self::replay_derivative_region) propagates them through the derivative program,
    /// and the block extraction methods convert them back into ordinary [`C::Value`](Domain::Value)s. Implementations
    /// must preserve which physical axis, if any, carries the packed directions.
    type PackedValue;

    /// Returns the dimension of the finite coordinate space represented by `r#type` (i.e., the number of independent
    /// scalar basis directions that a dense Jacobian or Hessian replay must pack for this value). This is distinct from
    /// array rank; the [`ArrayType`] implementation returns the product of all static shape extents, with dimension `1`
    /// for a scalar and `0` when any extent is zero. Returns [`DifferentiationError::NonFiniteCoordinateSpace`] when
    /// the space cannot be enumerated statically and [`DifferentiationError::CoordinateCountOverflow`] when its
    /// dimension does not fit in [`usize`].
    ///
    /// # Example
    ///
    /// An array with shape `[2, 3]` has rank `2`, but its coordinate space has dimension `6`: one independent scalar
    /// basis direction for each array element. A scalar array has rank `0` and coordinate-space dimension `1`, while
    /// an array with shape `[2, 0, 3]` has rank `3` and coordinate-space dimension `0`.
    ///
    /// # Parameters
    ///
    ///   - `r#type`: Type of the value whose coordinates will be enumerated.
    ///   - `transform`: Derivative transform requesting the coordinate space, used in diagnostics.
    ///   - `role`: Whether the parameter belongs to the transform's input or output structure.
    ///   - `path`: [`ParameterPath`] of the value within the owning [`Parameterized`] structure.
    fn coordinate_space_dimension(
        r#type: &Self,
        transform: DerivativeTransform,
        role: DifferentiationParameterRole,
        path: &ParameterPath,
    ) -> Result<usize, DifferentiationError>;

    /// Constructs the portion of a packed global coordinate basis that belongs to one differentiated value. If the
    /// value occupies `d` coordinates beginning at `coordinate_offset`, the returned value represents
    /// `packed_direction_count` directions (i.e., directions in `coordinate_offset..coordinate_offset + d` form
    /// the value's scalar identity basis, while every other direction is zero). `coordinate_type` determines which
    /// coordinates are enumerated, while `value_type` determines the differential values stored in the basis.
    ///
    /// # Parameters
    ///
    ///   - `context`: Context in which to construct the basis value.
    ///   - `coordinate_type`: Type whose finite coordinate space is being enumerated.
    ///   - `value_type`: Type of the basis values. Forward bases use the primal tangent type,
    ///     while reverse bases use the coordinate type's cotangent type.
    ///   - `coordinate_offset`: Index of the first packed direction belonging to `coordinate_type`.
    ///   - `packed_direction_count`: Number of coordinate directions packed across the differentiated structure.
    fn coordinate_basis(
        context: &C,
        coordinate_type: &Self,
        value_type: &Self,
        coordinate_offset: usize,
        packed_direction_count: usize,
    ) -> Result<Self::PackedValue, DifferentiationError>;

    /// Wraps an ordinary value as a packed replay value that is shared unchanged by every packed coordinate direction.
    /// Unlike [`coordinate_basis`](Self::coordinate_basis), this function does not introduce a mapped direction axis.
    /// It is instead used for derivative-program inputs such as residuals that do not vary between replay directions.
    ///
    /// # Parameters
    ///
    ///   - `value`: Ordinary value to share across all packed coordinate directions.
    fn replicated(value: C::Value) -> Self::PackedValue;

    /// Evaluates a derivative-[`Program`](crate::Program) [`Region`](crate::Region) once across all packed coordinate
    /// directions. Inputs constructed by [`coordinate_basis`](Self::coordinate_basis) vary across directions, while
    /// inputs constructed by [`replicated`](Self::replicated) are shared. The returned values must preserve the
    /// region's declared output order and retain enough direction-axis information for Jacobian block extraction.
    /// Implementations must replay nested regions using the arena owned by `region`.
    ///
    /// # Parameters
    ///
    ///   - `context`: Context in which to evaluate the derivative region.
    ///   - `region`: Derivative-program entry region, including access to its owning nested-region arena.
    ///   - `packed_direction_count`: Number of coordinate directions represented by `inputs`.
    ///   - `inputs`: Packed values in the derivative region's declared input order.
    fn replay_derivative_region(
        context: &C,
        region: RegionRef<'_, C::Constant, C::Operation>,
        packed_direction_count: usize,
        inputs: Vec<Self::PackedValue>,
    ) -> Result<Vec<Self::PackedValue>, DifferentiationError>;

    /// Validates the physical type of one forward-over-reverse dense Hessian block. Hessian blocks arrive as the values
    /// of the outer forward Jacobian rather than through a Hessian-specific extractor, so this composite layout check
    /// is the one block validation that cannot live inside
    /// [`extract_forward_jacobian_block`](Self::extract_forward_jacobian_block) or
    /// [`extract_reverse_jacobian_block`](Self::extract_reverse_jacobian_block), which validate their own results.
    /// The expected block consists of the output coordinate axes, followed by the first input's cotangent value axes,
    /// followed by the second input's coordinate axes. Layout metadata does not affect validation.
    ///
    /// # Example
    ///
    /// For an output with shape `[2]` and first and second inputs with shapes `[3]` and `[4]`, respectively, the
    /// expected Hessian block shape is `[2, 3, 4]`.
    ///
    /// # Parameters
    ///
    ///   - `block_type`: Physical type of the materialized second-derivative value.
    ///   - `output_type`: Type of the differentiated output leaf whose coordinate axes prefix the block.
    ///   - `first_input_type`: First input-leaf type whose cotangent supplies the block values.
    ///   - `second_input_type`: Second input-leaf type whose coordinate axes suffix the block.
    fn validate_hessian_block_type(
        block_type: &Self,
        output_type: &Self,
        first_input_type: &Self,
        second_input_type: &Self,
    ) -> Result<(), DifferentiationError>;

    /// Extracts one output/input Jacobian block from a packed forward-mode pushforward result. The method selects the
    /// packed directions belonging to `input_type`, restores its coordinate axes, places those axes after the output
    /// tangent value axes, and validates the resulting block type.
    ///
    /// # Examples
    ///
    /// For a function from an input with shape `[2]` to an output with shape `[3]`, this method extracts a Jacobian
    /// block with shape `[3, 2]` from the two packed input-coordinate directions.
    ///
    /// # Parameters
    ///
    ///   - `packed_output`: Packed pushforward result for one output value.
    ///   - `packed_direction_count`: Total number of packed input-coordinate directions.
    ///   - `input_coordinate_offset`: Index of the first packed direction belonging to `input_type`.
    ///   - `input_type`: Differentiated input-leaf type.
    ///   - `output_type`: Differentiated output-leaf type.
    fn extract_forward_jacobian_block(
        packed_output: &Self::PackedValue,
        packed_direction_count: usize,
        input_coordinate_offset: usize,
        input_type: &Self,
        output_type: &Self,
    ) -> Result<C::Value, DifferentiationError>;

    /// Extracts one output/input Jacobian block from a packed reverse-mode pullback result. The method selects the
    /// packed directions belonging to `output_type`, restores its coordinate axes before the input cotangent value
    /// axes, and validates the resulting block type.
    ///
    /// # Examples
    ///
    /// For a function from an input with shape `[2]` to an output with shape `[3]`, this method extracts a Jacobian
    /// block with shape `[3, 2]` from the three packed output-coordinate directions.
    ///
    /// # Parameters
    ///
    ///   - `packed_output`: Packed pullback result for one input value.
    ///   - `packed_direction_count`: Total number of packed output-coordinate directions.
    ///   - `output_coordinate_offset`: Index of the first packed direction belonging to `output_type`.
    ///   - `output_type`: Differentiated output-leaf type.
    ///   - `input_type`: Differentiated input-leaf type.
    fn extract_reverse_jacobian_block(
        packed_output: &Self::PackedValue,
        packed_direction_count: usize,
        output_coordinate_offset: usize,
        output_type: &Self,
        input_type: &Self,
    ) -> Result<C::Value, DifferentiationError>;
}

// TODO(eaplatanios): Review this.

impl<C: Context<Type = ArrayType>> DenseDifferentiableType<C> for ArrayType
where
    C::Value: Broadcast + Reshape + Slice + Transpose,
    C::Operation: BatchableOperation<C>
        + BatchableOperation<TracingContext<C::Constant, C::Operation>>
        + From<CoordinateBasisOperation<ArrayType>>
        + From<TransposeOperation>
        + From<BroadcastOperation>,
{
    type PackedValue = ArrayBatch<C::Value>;

    fn coordinate_space_dimension(
        r#type: &Self,
        transform: DerivativeTransform,
        role: DifferentiationParameterRole,
        path: &ParameterPath,
    ) -> Result<usize, DifferentiationError> {
        let shape = r#type.static_shape().ok_or_else(|| DifferentiationError::NonFiniteCoordinateSpace {
            transform,
            role,
            path: path.to_string(),
            r#type: r#type.to_string(),
        })?;
        if shape.dimensions().contains(&0) {
            return Ok(0);
        }
        shape.dimensions().iter().copied().try_fold(1usize, |count, size| {
            count.checked_mul(size).ok_or_else(|| DifferentiationError::CoordinateCountOverflow {
                transform,
                role,
                path: path.to_string(),
                r#type: r#type.to_string(),
            })
        })
    }

    fn coordinate_basis(
        context: &C,
        coordinate_type: &Self,
        value_type: &Self,
        coordinate_offset: usize,
        packed_direction_count: usize,
    ) -> Result<Self::PackedValue, DifferentiationError> {
        if coordinate_type.shape() != value_type.shape() {
            return Err(TypeError {
                message: format!(
                    "coordinate basis type {coordinate_type} and value type {value_type} have different shapes",
                ),
            }
            .into());
        }
        let expected_type = value_type.with_inserted_dimension(0, Size::Static(packed_direction_count))?;
        let mut outputs = context.bind(
            CoordinateBasisOperation::new(value_type.clone(), coordinate_offset, packed_direction_count),
            Vec::new(),
            &[],
        )?;
        check_count!("output", outputs, 1, ProgramError);
        let value = outputs.remove(0);
        if value.r#type().as_ref() != &expected_type {
            return Err(TypeError {
                message: format!(
                    "coordinate basis for leaf type {value_type} has type {} but expected {expected_type}",
                    value.r#type(),
                ),
            }
            .into());
        }
        Ok(ArrayBatch::new(expected_type, value, Some(0)).map_err(ProgramError::from)?)
    }

    #[inline]
    fn replicated(value: C::Value) -> Self::PackedValue {
        ArrayBatch::replicated(value)
    }

    fn replay_derivative_region(
        context: &C,
        region: RegionRef<'_, C::Constant, C::Operation>,
        packed_direction_count: usize,
        inputs: Vec<Self::PackedValue>,
    ) -> Result<Vec<Self::PackedValue>, DifferentiationError> {
        Ok(BatchingContext::new(context.clone(), packed_direction_count)
            .batch_region(region, inputs)
            .map_err(ProgramError::from)?)
    }

    fn validate_hessian_block_type(
        block_type: &Self,
        output_type: &Self,
        first_input_type: &Self,
        second_input_type: &Self,
    ) -> Result<(), DifferentiationError> {
        let first_input_cotangent_type = first_input_type.cotangent();
        if first_input_cotangent_type.is_zero_space() {
            return Err(
                TypeError { message: format!("hessian input type {first_input_type} has no cotangent type") }.into()
            );
        }
        validate_array_block_type(
            DerivativeTransform::Hessian,
            block_type,
            &first_input_cotangent_type,
            &[output_type],
            &[second_input_type],
        )
    }

    fn extract_forward_jacobian_block(
        packed_output: &Self::PackedValue,
        packed_direction_count: usize,
        input_coordinate_offset: usize,
        input_type: &Self,
        output_type: &Self,
    ) -> Result<C::Value, DifferentiationError> {
        let input_shape =
            static_shape(input_type, DerivativeTransform::JacobianForward, DifferentiationParameterRole::Input)?;
        let output_shape =
            static_shape(output_type, DerivativeTransform::JacobianForward, DifferentiationParameterRole::Output)?;
        let output_tangent_type = output_type.tangent();
        if output_tangent_type.is_zero_space() {
            return Err(TypeError {
                message: format!("forward Jacobian output type {output_type} has no tangent type"),
            }
            .into());
        }
        let value = basis_range_value(
            packed_output,
            packed_direction_count,
            input_coordinate_offset,
            input_shape.dimensions(),
            &output_tangent_type,
        )?;
        let permutation = (input_shape.rank()..input_shape.rank() + output_shape.rank())
            .chain(0..input_shape.rank())
            .collect::<Vec<_>>();
        let value = value.transpose(permutation)?;
        validate_array_block_type(
            DerivativeTransform::JacobianForward,
            value.r#type().as_ref(),
            &output_tangent_type,
            &[],
            &[input_type],
        )?;
        Ok(value)
    }

    fn extract_reverse_jacobian_block(
        packed_output: &Self::PackedValue,
        packed_direction_count: usize,
        output_coordinate_offset: usize,
        output_type: &Self,
        input_type: &Self,
    ) -> Result<C::Value, DifferentiationError> {
        let output_shape =
            static_shape(output_type, DerivativeTransform::JacobianReverse, DifferentiationParameterRole::Output)?;
        let input_cotangent_type = input_type.cotangent();
        if input_cotangent_type.is_zero_space() {
            return Err(TypeError {
                message: format!("reverse Jacobian input type {input_type} has no cotangent type"),
            }
            .into());
        }
        let value = basis_range_value(
            packed_output,
            packed_direction_count,
            output_coordinate_offset,
            output_shape.dimensions(),
            &input_cotangent_type,
        )?;
        validate_array_block_type(
            DerivativeTransform::JacobianReverse,
            value.r#type().as_ref(),
            &input_cotangent_type,
            &[output_type],
            &[],
        )?;
        Ok(value)
    }
}

#[derive(Copy, Clone)]
enum DenseMode {
    Forward,
    ForwardHolomorphic,
    Reverse,
    ReverseHolomorphic,
}

impl DenseMode {
    fn transform(self) -> DerivativeTransform {
        match self {
            Self::Forward | Self::ForwardHolomorphic => DerivativeTransform::JacobianForward,
            Self::Reverse | Self::ReverseHolomorphic => DerivativeTransform::JacobianReverse,
        }
    }

    fn is_holomorphic(self) -> bool {
        matches!(self, Self::ForwardHolomorphic | Self::ReverseHolomorphic)
    }

    fn permits_complex_input(self) -> bool {
        matches!(self, Self::ForwardHolomorphic | Self::Reverse | Self::ReverseHolomorphic)
    }

    fn permits_complex_output(self) -> bool {
        matches!(self, Self::Forward | Self::ForwardHolomorphic | Self::ReverseHolomorphic)
    }
}

fn validate_types<T: DifferentiableType, S: Parameterized<T>>(
    types: &S,
    mode: DenseMode,
    role: DifferentiationParameterRole,
) -> Result<(), DifferentiationError> {
    for (path, r#type) in types.named_parameters() {
        let differential_type = match mode {
            DenseMode::Forward | DenseMode::ForwardHolomorphic => r#type.tangent(),
            DenseMode::Reverse | DenseMode::ReverseHolomorphic => r#type.cotangent(),
        };
        if differential_type.is_zero_space() {
            return Err(DifferentiationError::NonDifferentiableParameter {
                transform: mode.transform(),
                role,
                path: path.to_string(),
                r#type: r#type.to_string(),
            });
        }
        if mode.is_holomorphic() && !r#type.is_complex() {
            return Err(DifferentiationError::NonComplexParameter {
                transform: mode.transform(),
                role,
                path: path.to_string(),
                r#type: r#type.to_string(),
            });
        }
        let permits_complex = if role == DifferentiationParameterRole::Input {
            mode.permits_complex_input()
        } else {
            mode.permits_complex_output()
        };
        if !mode.is_holomorphic() && r#type.is_complex() && !permits_complex {
            return Err(DifferentiationError::ComplexParameter {
                transform: mode.transform(),
                role,
                path: path.to_string(),
                r#type: r#type.to_string(),
            });
        }
    }
    Ok(())
}

fn coordinate_offsets<C: Context, S: Parameterized<C::Type>>(
    types: &S,
    mode: DenseMode,
    role: DifferentiationParameterRole,
) -> Result<Vec<usize>, DifferentiationError>
where
    C::Type: DenseDifferentiableType<C>,
{
    let mut offsets = Vec::new();
    offsets.push(0usize);
    for (path, r#type) in types.named_parameters() {
        let dimension = C::Type::coordinate_space_dimension(r#type, mode.transform(), role, &path)?;
        offsets.push(offsets.last().copied().unwrap().checked_add(dimension).ok_or_else(|| {
            DifferentiationError::CoordinateCountOverflow {
                transform: mode.transform(),
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
    let mode = if holomorphic { DenseMode::ForwardHolomorphic } else { DenseMode::Forward };
    let input_structure = primals.parameter_structure();
    let input_values = primals.into_parameters().collect::<Vec<_>>();
    let input_types = I::To::<C::Type>::from_parameters(
        input_structure.clone(),
        input_values.iter().map(|value| value.r#type().into_owned()),
    )?;
    validate_types(&input_types, mode, DifferentiationParameterRole::Input)?;
    let primals = I::from_parameters(input_structure, input_values)?;
    let (output, pushforward) = context.linearize(function, primals)?;
    let output_structure = output.parameter_structure();
    let output_values = output.into_parameters().collect::<Vec<_>>();
    let output_types = O::To::<C::Type>::from_parameters(
        output_structure,
        output_values.iter().map(|value| value.r#type().into_owned()),
    )?;
    validate_types(&output_types, mode, DifferentiationParameterRole::Output)?;

    let input_offsets = coordinate_offsets::<C, _>(&input_types, mode, DifferentiationParameterRole::Input)?;
    let _ = coordinate_offsets::<C, _>(&output_types, mode, DifferentiationParameterRole::Output)?;
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
                transform: mode.transform(),
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
                    transform: mode.transform(),
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
    let mode = if holomorphic { DenseMode::ReverseHolomorphic } else { DenseMode::Reverse };
    let input_structure = primals.parameter_structure();
    let input_values = primals.into_parameters().collect::<Vec<_>>();
    let input_types = I::To::<C::Type>::from_parameters(
        input_structure.clone(),
        input_values.iter().map(|value| value.r#type().into_owned()),
    )?;
    validate_types(&input_types, mode, DifferentiationParameterRole::Input)?;
    let primals = I::from_parameters(input_structure, input_values)?;
    let (output, pullback) = context.vjp(function, primals)?;
    let output_structure = output.parameter_structure();
    let output_values = output.into_parameters().collect::<Vec<_>>();
    let output_types = O::To::<C::Type>::from_parameters(
        output_structure,
        output_values.iter().map(|value| value.r#type().into_owned()),
    )?;
    validate_types(&output_types, mode, DifferentiationParameterRole::Output)?;

    let _ = coordinate_offsets::<C, _>(&input_types, mode, DifferentiationParameterRole::Input)?;
    let output_offsets = coordinate_offsets::<C, _>(&output_types, mode, DifferentiationParameterRole::Output)?;
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
                    transform: mode.transform(),
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

fn validate_array_block_type(
    transform: DerivativeTransform,
    block_type: &ArrayType,
    value_type: &ArrayType,
    prefix_coordinate_types: &[&ArrayType],
    suffix_coordinate_types: &[&ArrayType],
) -> Result<(), DifferentiationError> {
    let mut expected_type = value_type.clone().with_layout(None);
    let mut prefix_index = 0;
    for coordinate_type in prefix_coordinate_types {
        for size in static_shape(coordinate_type, transform, DifferentiationParameterRole::Derivative)?.dimensions() {
            expected_type = expected_type.with_inserted_dimension(prefix_index, Size::Static(*size))?;
            prefix_index += 1;
        }
    }
    for coordinate_type in suffix_coordinate_types {
        for size in static_shape(coordinate_type, transform, DifferentiationParameterRole::Derivative)?.dimensions() {
            let rank = expected_type.rank();
            expected_type = expected_type.with_inserted_dimension(rank, Size::Static(*size))?;
        }
    }
    let block_type_without_layout = block_type.clone().with_layout(None);
    if block_type_without_layout != expected_type {
        return Err(TypeError {
            message: format!("derivative block has type {block_type} but expected {expected_type}"),
        }
        .into());
    }
    Ok(())
}

fn static_shape(
    r#type: &ArrayType,
    transform: DerivativeTransform,
    role: DifferentiationParameterRole,
) -> Result<StaticShape, DifferentiationError> {
    r#type.static_shape().ok_or_else(|| DifferentiationError::NonFiniteCoordinateSpace {
        transform,
        role,
        path: ParameterPath::root().to_string(),
        r#type: r#type.to_string(),
    })
}

fn basis_range_value<V>(
    batch: &ArrayBatch<V>,
    batch_size: usize,
    basis_offset: usize,
    basis_shape: &[usize],
    expected_item_type: &ArrayType,
) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayType> + Broadcast + Reshape + Slice + Transpose,
{
    let aligned = batch.match_axis(0, batch_size, ShardingDimension::Replicated)?;
    let actual_item_type = aligned.unbatched_type();
    if actual_item_type.clone().with_layout(None) != expected_item_type.clone().with_layout(None) {
        return Err(TypeError {
            message: format!(
                "batched derivative output has per-item type {actual_item_type} but expected {expected_item_type}",
            ),
        }
        .into());
    }
    let item_shape = expected_item_type.static_shape().ok_or_else(|| TypeError {
        message: format!(
            "jacobian or hessian materialization requires a fully static array shape but got {expected_item_type}"
        ),
    })?;
    let basis_count = if basis_shape.contains(&0) {
        0
    } else {
        basis_shape.iter().try_fold(1usize, |count, size| {
            count.checked_mul(*size).ok_or_else(|| ProgramError::InvalidArgument {
                message: format!("coordinate basis shape {basis_shape:?} overflows usize"),
            })
        })?
    };
    let physical_type = aligned.r#type();
    let physical_shape = physical_type.static_shape().ok_or_else(|| TypeError {
        message: format!(
            "jacobian or hessian materialization requires a fully static array shape but got {physical_type}"
        ),
    })?;
    let mut start_indices = vec![0; physical_shape.rank()];
    start_indices[0] = basis_offset;
    let mut limit_indices = physical_shape.dimensions().to_vec();
    limit_indices[0] = basis_offset.checked_add(basis_count).ok_or_else(|| ProgramError::InvalidArgument {
        message: "coordinate basis range overflows usize".to_string(),
    })?;
    let strides = vec![1; limit_indices.len()];
    let sliced = aligned.value().slice(&start_indices, &limit_indices, &strides)?;
    let reshaped_shape =
        Shape::new(basis_shape.iter().chain(item_shape.dimensions()).copied().map(Size::Static).collect());
    sliced.reshape(reshaped_shape)
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
    use crate::batching::{ArrayBatch, BatchAxis};
    use crate::contexts::{Context, EagerContext};
    use crate::differentiation::{
        DerivativeTransform, DifferentiableType, DifferentiationError, DifferentiationParameterRole,
    };
    use crate::operations::compare::{Compare, ComparisonDirection};
    use crate::operations::constants::ZeroLike;
    use crate::operations::control_flow::Select;
    use crate::operations::math::Add;
    use crate::parameters::{ParameterPath, Parameterized};
    use crate::programs::types::Typed;
    use crate::programs::values::Value;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding};
    use crate::types::DataType::{F32, F64};
    use crate::types::{ArrayType, DataType, Shape, Size};

    use super::{DenseDifferentiableType, DenseMode, Jacobian, coordinate_offsets, jacfwd, jacrev};

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
    fn test_dense_type_validation_reports_block_and_coordinate_overflow_errors() {
        type TestContext = EagerContext<Array, ArrayOperation<Array>>;

        let error = <ArrayType as DenseDifferentiableType<TestContext>>::validate_hessian_block_type(
            &ArrayType::new(F32, Shape::new(vec![Size::Static(2)])),
            &ArrayType::scalar(F32),
            &ArrayType::scalar(F32),
            &ArrayType::scalar(F32),
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "derivative block has type f32[2] but expected f32[]");

        let input_types = (ArrayType::new(F32, Shape::new(vec![Size::Static(usize::MAX)])), ArrayType::scalar(F32));
        assert_eq!(
            coordinate_offsets::<TestContext, _>(
                &input_types,
                DenseMode::Forward,
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
            coordinate_offsets::<TestContext, _>(
                &empty_input_types,
                DenseMode::Forward,
                DifferentiationParameterRole::Input,
            )
            .unwrap(),
            vec![0, 0],
        );
    }

    #[test]
    fn test_dense_array_coordinate_basis_uses_the_requested_value_type() {
        type TestContext = EagerContext<Array, ArrayOperation<Array>>;

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let coordinate_type = ArrayType::scalar(F32)
            .with_sharding(Sharding::new(mesh, Vec::new()).unwrap().with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let value_type = coordinate_type.cotangent();
        let basis = <ArrayType as DenseDifferentiableType<TestContext>>::coordinate_basis(
            &TestContext::new(),
            &coordinate_type,
            &value_type,
            0,
            1,
        )
        .unwrap();

        assert_eq!(basis.unbatched_type(), value_type);
    }

    #[test]
    fn test_dense_array_block_validation_uses_transform_value_types() {
        type TestContext = EagerContext<Array, ArrayOperation<Array>>;

        let output_type = ArrayType::new(F64, Shape::new(vec![Size::Static(2)]));
        let input_type = ArrayType::new(F32, Shape::new(vec![Size::Static(3)]));
        let wrong_block_type = ArrayType::new(F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));

        // Forward blocks carry the output leaf's tangent values while reverse blocks carry the input leaf's
        // cotangent values, so a packed replay output that stays in the narrow `f8e8m0fnu` storage type is rejected
        // by both extractors, whose expected per-item type is the widened `f32` differential representation.
        let narrow_type = ArrayType::scalar(DataType::F8E8M0FNU);
        let physical_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Size::Static(1)]));
        let packed =
            ArrayBatch::new(physical_type.clone(), Array::from_f64s(physical_type, vec![2.0]), BatchAxis::new(0))
                .unwrap();
        assert_eq!(
            <ArrayType as DenseDifferentiableType<TestContext>>::extract_forward_jacobian_block(
                &packed,
                1,
                0,
                &narrow_type,
                &narrow_type,
            )
            .unwrap_err()
            .to_string(),
            "batched derivative output has per-item type f8e8m0fnu[] but expected f32[]",
        );
        assert_eq!(
            <ArrayType as DenseDifferentiableType<TestContext>>::extract_reverse_jacobian_block(
                &packed,
                1,
                0,
                &narrow_type,
                &narrow_type,
            )
            .unwrap_err()
            .to_string(),
            "batched derivative output has per-item type f8e8m0fnu[] but expected f32[]",
        );

        let first_input_type = ArrayType::scalar(F32);
        assert_eq!(
            <ArrayType as DenseDifferentiableType<TestContext>>::validate_hessian_block_type(
                &wrong_block_type,
                &output_type,
                &first_input_type,
                &input_type,
            )
            .unwrap_err()
            .to_string(),
            "derivative block has type f64[2, 3] but expected f32[2, 3]",
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
