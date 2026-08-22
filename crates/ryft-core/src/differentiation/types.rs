use std::ops::{Add, Mul};

use crate::arrays::{
    ArrayBatch, ArrayBatching, ArrayIrType, ArrayType, DataType, Dimension, Shape, Sharding, ShardingDimension,
};
use crate::batching::{BatchableOperation, BatchingContext, RecursiveBatchingPolicy};
use crate::contexts::Context;
use crate::differentiation::{DerivativeTransform, DifferentiationError, DifferentiationParameterRole};
use crate::operations::{
    Broadcast, BroadcastOperation, Compare, ComparisonDirection, Fill, Iota, One, Reshape, ReshapeParameters, Select,
    Slice, Transpose, TransposeOperation, Zero,
};
use crate::parameters::ParameterPath;
use crate::programs::{ProgramError, RegionRef, Type, TypeError, Typed, Value};
use crate::tracing::TracingContext;

/// A [`Type`] whose forward perturbations and reverse adjoints carry well-defined differential representations.
/// Differential values need not use the primal representation. A compact primal storage format may require a wider
/// signed type to support zero, addition, and negative linear contributions.
pub trait DifferentiableType: Type {
    /// Returns `true` if this [`Type`] represents the trivial differential space whose only possible value is zero
    /// (e.g., [`DataType::Zero`]). Generic transform code uses this property to distinguish a first-class zero-space
    /// type from a type that can carry live, potentially nonzero differential values.
    fn is_zero_space(&self) -> bool;

    /// Returns the [`Type`] that forward-mode tangents of values of this [`Type`] carry. The returned type is used
    /// for forward-mode inputs, outputs, structural zeros, and intermediate Jacobian-Vector Product (JVP) values. It
    /// preserves primal placement metadata because a tangent follows the same forward data flow as its primal. Most
    /// differentiable types use themselves, but specialized storage representations may use a wider differential
    /// representation. For example, [`DataType::F8E8M0FNU`] uses [`DataType::F32`] because its unsigned power-of-two
    /// representation cannot represent zero or negative linear contributions. Non-differentiable types return a
    /// first-class zero-space type, such as [`DataType::Zero`]. Generated linear programs are compact with respect to
    /// such types (i.e., where a program boundary would carry one tangent input or output per primal leaf, leaves whose
    /// tangent type [`is_zero_space`](Self::is_zero_space) get no boundary input or output at all). Structured callable
    /// transforms such as [`Pushforward`](crate::Pushforward) still expose the complete leaf-for-leaf public derivative
    /// tree and restore each omitted leaf as a typed zero at their public boundaries. Types that cannot participate in
    /// forward differentiation return an error instead of fabricating a zero-space representation.
    fn tangent(&self) -> Result<Self, DifferentiationError>;

    /// Returns the [`Type`] that reverse-mode cotangents of values of this [`Type`] carry. The returned type is the
    /// representation used for reverse-mode seeds, accumulation, structural zeros, and outputs. In most cases it is the
    /// type itself, but it may instead be a distinct representation that supports the required linear operations. For
    /// example, [`DataType::F8E8M0FNU`] uses [`DataType::F32`] cotangents because its unsigned power-of-two storage
    /// format cannot represent zero or negative values, while [`ArrayType`] also swaps the unreduced and reduced axes
    /// of its [`Sharding`]. Refer to [`Sharding::cotangent`] for more information. This mapping is not required to be
    /// an _involution_ (i.e., a specialized primal representation may map to a general-purpose cotangent representation
    /// that is itself a fixed point). Non-differentiable types return a first-class zero-space type. Reverse mode
    /// accumulates no live adjoint for values of those types, and generated linear programs are compact with respect
    /// to them (i.e., where a pullback boundary would carry one cotangent input per primal output and one cotangent
    /// output per primal input, leaves whose cotangent type [`is_zero_space`](Self::is_zero_space) get no boundary
    /// input or output at all). Structured callable transforms such as [`Pullback`](crate::Pullback) still expose the
    /// complete leaf-for-leaf public derivative tree and restore each omitted leaf as a typed zero at their public
    /// boundaries. Types that cannot participate in reverse differentiation return an error instead of fabricating a
    /// zero-space representation.
    fn cotangent(&self) -> Result<Self, DifferentiationError>;
}

impl DifferentiableType for DataType {
    #[inline]
    fn is_zero_space(&self) -> bool {
        *self == Self::Zero
    }

    #[inline]
    fn tangent(&self) -> Result<Self, DifferentiationError> {
        Ok(match self {
            Self::Token
            | Self::Boolean
            | Self::I1
            | Self::I2
            | Self::I4
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64
            | Self::U1
            | Self::U2
            | Self::U4
            | Self::U8
            | Self::U16
            | Self::U32
            | Self::U64
            | Self::Zero => Self::Zero,
            Self::F4E2M1FN
            | Self::F6E2M3FN
            | Self::F6E3M2FN
            | Self::F8E3M4
            | Self::F8E4M3
            | Self::F8E4M3FN
            | Self::F8E4M3FNUZ
            | Self::F8E4M3B11FNUZ
            | Self::F8E5M2
            | Self::F8E5M2FNUZ => *self,
            Self::F8E8M0FNU => Self::F32,
            Self::BF16 | Self::F16 | Self::F32 | Self::F64 | Self::C64 | Self::C128 => *self,
        })
    }

    #[inline]
    fn cotangent(&self) -> Result<Self, DifferentiationError> {
        self.tangent()
    }
}

impl DifferentiableType for ArrayType {
    #[inline]
    fn is_zero_space(&self) -> bool {
        self.data_type().is_zero_space()
    }

    #[inline]
    fn tangent(&self) -> Result<Self, DifferentiationError> {
        // Forward perturbations follow their primals' placement. An element-representation change clears explicit
        // layout because byte-level layout metadata cannot in general survive a change in element width.
        let data_type = self.data_type().tangent()?;
        let layout = if data_type == self.data_type() { self.layout.clone() } else { None };
        Ok(Self { data_type, layout, ..self.clone() })
    }

    #[inline]
    fn cotangent(&self) -> Result<Self, DifferentiationError> {
        // Use the element cotangent representation, clear explicit layout when that representation changes element
        // width, swap the unreduced and reduced sharding axes, and keep all other type metadata unchanged.
        let data_type = self.data_type().cotangent()?;
        let layout = if data_type == self.data_type() { self.layout.clone() } else { None };
        Ok(Self { data_type, layout, sharding: self.sharding.as_ref().map(Sharding::cotangent), ..self.clone() })
    }
}

impl DifferentiableType for ArrayIrType {
    #[inline]
    fn is_zero_space(&self) -> bool {
        match self {
            Self::Array(r#type) => r#type.is_zero_space(),
            Self::Dimension(_) => true,
            Self::Reference(_) => {
                // A reference is not itself a differential value. Returning `false` avoids misclassifying it as a
                // structural zero. `Self::tangent` and `Self::cotangent` provide the authoritative rejection.
                false
            }
        }
    }

    #[inline]
    fn tangent(&self) -> Result<Self, DifferentiationError> {
        match self {
            Self::Array(r#type) => Ok(Self::Array(r#type.tangent()?)),
            Self::Dimension(_) => {
                // First-class dimensions describe shapes rather than numerical data. Their tangent space is
                // structurally zero and uses the ordinary `Self::Array` variant so that generic zero materialization
                // has one canonical backend representation.
                Ok(Self::Array(ArrayType::scalar(DataType::Zero)))
            }
            Self::Reference(_) => Err(DifferentiationError::UndefinedTangentType { primal_type: self.to_string() }),
        }
    }

    #[inline]
    fn cotangent(&self) -> Result<Self, DifferentiationError> {
        match self {
            Self::Array(r#type) => Ok(Self::Array(r#type.cotangent()?)),
            Self::Dimension(_) => {
                // As in `Self::tangent`, dimensions receive no live adjoint. Keeping the zero space in the array
                // member prevents a zero value from being mistaken for a first-class dimension.
                Ok(Self::Array(ArrayType::scalar(DataType::Zero)))
            }
            Self::Reference(_) => Err(DifferentiationError::UndefinedCotangentType { primal_type: self.to_string() }),
        }
    }
}

/// A [`Type`] that is supported by dense differentiation functions that compute [`Jacobian`](crate::Jacobian)s and
/// [`Hessian`](crate::Hessian)s. [`Type`] and [`DifferentiableType`] describe individual primal and cotangent values,
/// but they do not state that a leaf (i.e., a [`Parameter`](crate::parameters::Parameter)) has a finite coordinate
/// space, that several directions can be represented by one value, or how packed replay results become public
/// derivative blocks. Implementations of this trait provide only those representation-specific operations. The
/// Jacobian and Hessian algorithms retain ownership of structure traversal, differentiation, ordering, and result
/// construction.
pub trait DenseDifferentiableType<C: Context<Type = Self>>: DifferentiableType {
    /// Intermediate representation of one logical [`C::Value`](crate::contexts::Domain::Value) during a packed,
    /// multi-direction derivative replay. A value is either mapped over the packed coordinate directions or replicated
    /// unchanged across them. Its logical per-direction type remains `Self`, while its physical representation may carry
    /// an additional axis that indexes those directions. [`coordinate_basis`](Self::coordinate_basis) and
    /// [`replicated`](Self::replicated) construct these values,
    /// [`replay_derivative_region`](Self::replay_derivative_region) propagates them through the derivative program,
    /// and the block extraction methods convert them back into ordinary
    /// [`C::Value`](crate::contexts::Domain::Value)s. Implementations must preserve which physical axis, if any, carries
    /// the packed directions.
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
    ///   - `path`: [`ParameterPath`] of the value within the owning
    ///     [`Parameterized`](crate::parameters::Parameterized) structure.
    fn coordinate_space_dimension(
        r#type: &Self,
        transform: DerivativeTransform,
        role: DifferentiationParameterRole,
        path: &ParameterPath,
    ) -> Result<usize, DifferentiationError>;

    /// Constructs the portion of a packed global coordinate basis that belongs to one differentiated value.
    /// If the value occupies `c` coordinates beginning at `coordinate_offset`, the returned value represents the
    /// `coordinate_count` standard-basis directions. Directions in `coordinate_offset..coordinate_offset + c` form
    /// the value's scalar identity basis, while every other direction is zero. `coordinate_type` determines which
    /// coordinates are enumerated, while `value_type` determines the differential values stored in the basis.
    ///
    /// For a value with shape `S`, this method returns a physical value with shape `[coordinate_count] ++ S`.
    /// Its element at `[k, i...]` is one exactly when `k == coordinate_offset + flatten_row_major(i...)`, and zero
    /// otherwise. For example, a two-element value at offset `0` followed by a scalar value at offset `2` produces
    /// three packed directions whose two fragments are:
    ///
    /// ```text
    /// two_element_value = [[1, 0],
    ///                      [0, 1],
    ///                      [0, 0]]
    /// scalar_value      = [0, 0, 1]
    /// ```
    ///
    /// Forward-mode Jacobians use these values as packed Jacobian-Vector Product (JVP) tangents, while reverse-mode
    /// Jacobians use them as packed Vector-Jacobian Product (VJP) cotangents. Implementations construct the basis
    /// through ordinary context/value capabilities so a staging context records the constituent operations directly
    /// in the surrounding program.
    ///
    /// # Parameters
    ///
    ///   - `context`: Context in which to construct the basis value.
    ///   - `coordinate_type`: Type whose finite coordinate space is being enumerated.
    ///   - `value_type`: Type of the basis values. Forward bases use the primal tangent type,
    ///     while reverse bases use the coordinate type's cotangent type.
    ///   - `coordinate_offset`: Index of `coordinate_type`'s first coordinate in the global coordinate space.
    ///   - `coordinate_count`: Total number of coordinates across the differentiated structure.
    fn coordinate_basis(
        context: &C,
        coordinate_type: &Self,
        value_type: &Self,
        coordinate_offset: usize,
        coordinate_count: usize,
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

impl<C: Context<Type = ArrayType>> DenseDifferentiableType<C> for ArrayType
where
    C: One<C::Value> + Zero<C::Value> + Iota<C::Value> + Fill<u64, C::Value>,
    C::Value: Add<Output = C::Value>
        + Mul<Output = C::Value>
        + Compare<C::Value>
        + Select
        + Broadcast
        + Reshape
        + Slice
        + Transpose,
    C::Operation: BatchableOperation<C, ArrayBatching>
        + BatchableOperation<TracingContext<C::Constant, C::Operation>, ArrayBatching>
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
        coordinate_count: usize,
    ) -> Result<Self::PackedValue, DifferentiationError> {
        if coordinate_type.shape() != value_type.shape() {
            return Err(TypeError::invalid(format!(
                "coordinate basis type {coordinate_type} and value type {value_type} have different shapes",
            ))
            .into());
        }
        let cotangent_data_type = value_type.data_type().cotangent()?;
        if cotangent_data_type.is_zero_space() {
            return Err(TypeError::invalid(format!(
                "coordinate basis requires a differentiable value type but got {value_type}",
            ))
            .into());
        }
        if cotangent_data_type != value_type.data_type() {
            return Err(TypeError::invalid(format!(
                "coordinate basis values of type {} cannot represent their own cotangents; use {} instead",
                value_type,
                value_type.clone().with_data_type(cotangent_data_type),
            ))
            .into());
        }
        let value_dimensions = value_type.static_shape().ok_or_else(|| {
            TypeError::invalid(format!("coordinate basis requires a fully static value type but got {value_type}"))
        })?;
        let value_coordinate_count = if value_dimensions.dimensions().contains(&0) {
            0
        } else {
            value_dimensions.dimensions().iter().copied().try_fold(1usize, |count, size| {
                count.checked_mul(size).ok_or_else(|| {
                    TypeError::invalid(format!("coordinate count overflows usize for value type {value_type}"))
                })
            })?
        };
        let coordinate_end = coordinate_offset.checked_add(value_coordinate_count).ok_or_else(|| {
            TypeError::invalid(format!("coordinate range overflows usize for value type {value_type}"))
        })?;
        if coordinate_end > coordinate_count {
            return Err(TypeError::invalid(format!(
                "coordinate range [{coordinate_offset}, {coordinate_end}) exceeds coordinate count {coordinate_count}",
            ))
            .into());
        }

        // TODO(eaplatanios): Review this portion.
        let expected_type = value_type.with_inserted_dimension(0, Dimension::Static(coordinate_count))?;
        let value = if value_coordinate_count == 0 {
            context.zero(&expected_type)?
        } else {
            // Prefer a rank-independent rectangular identity fragment. Plan both reshapes before emitting values so
            // this path is used only when flattening preserves the exact output placement and layout type.
            let rectangular_shape =
                Shape::new(vec![Dimension::Static(coordinate_count), Dimension::Static(value_coordinate_count)]);
            let rectangular_type = if expected_type.layout().is_none() {
                match expected_type.reshape(rectangular_shape) {
                    Ok(rectangular_type) => Some(rectangular_type),
                    Err(_reshape_error) => None,
                }
            } else {
                None
            };
            let rectangular_plan = rectangular_type.and_then(|rectangular_type| {
                let output_reshape_parameters =
                    if expected_type.sharding().is_some_and(|sharding| sharding.references_auto_axis()) {
                        ReshapeParameters::new(expected_type.shape().clone())
                    } else {
                        ReshapeParameters::new(expected_type.shape().clone())
                            .with_output_sharding(expected_type.sharding().cloned())
                    };
                rectangular_type
                    .reshape(output_reshape_parameters.clone())
                    .is_ok_and(|restored_type| restored_type == expected_type)
                    .then_some((rectangular_type, output_reshape_parameters))
            });
            if let Some((rectangular_type, output_reshape)) = rectangular_plan {
                let index_type = rectangular_type.clone().with_data_type(DataType::U64);
                let direction_index = context.iota(&index_type, 0)?;
                let mut value_coordinate_index = context.iota(&index_type, 1)?;
                if coordinate_offset != 0 {
                    let offset = u64::try_from(coordinate_offset).map_err(|_| ProgramError::InvalidArgument {
                        message: format!("coordinate offset {coordinate_offset} does not fit in u64"),
                    })?;
                    value_coordinate_index = value_coordinate_index + context.fill(&index_type, offset)?;
                }
                let selected = direction_index.compare(&value_coordinate_index, ComparisonDirection::Equal)?;
                let zero = context.zero(&rectangular_type)?;
                let one = context.one(&rectangular_type)?;
                C::Value::select(&selected, &one, &zero)?.reshape(output_reshape)?
            } else {
                // Flattening is not a placement-preserving reshape for every explicit layout or non-contiguous and
                // unconstrained sharding. Construct the same row-major coordinates directly in the output shape.
                let index_type = expected_type.clone().with_data_type(DataType::U64);
                let direction_index = context.iota(&index_type, 0)?;
                let mut flat_coordinate = None;
                let mut stride = 1u64;
                for (value_axis, dimension_size) in value_dimensions.dimensions().iter().copied().enumerate().rev() {
                    let coordinate = context.iota(&index_type, value_axis + 1)?;
                    let coordinate =
                        if stride == 1 { coordinate } else { coordinate * context.fill(&index_type, stride)? };
                    flat_coordinate = Some(match flat_coordinate {
                        Some(accumulated) => accumulated + coordinate,
                        None => coordinate,
                    });
                    stride = stride
                        .checked_mul(u64::try_from(dimension_size).map_err(|_| ProgramError::InvalidArgument {
                            message: format!("value dimension {dimension_size} does not fit in u64"),
                        })?)
                        .ok_or_else(|| ProgramError::InvalidArgument {
                            message: format!("coordinate count overflows u64 for value type {value_type}"),
                        })?;
                }
                let mut flat_coordinate = flat_coordinate.map_or_else(|| context.fill(&index_type, 0u64), Ok)?;
                if coordinate_offset != 0 {
                    let offset = u64::try_from(coordinate_offset).map_err(|_| ProgramError::InvalidArgument {
                        message: format!("coordinate offset {coordinate_offset} does not fit in u64"),
                    })?;
                    flat_coordinate = flat_coordinate + context.fill(&index_type, offset)?;
                }
                let selected = direction_index.compare(&flat_coordinate, ComparisonDirection::Equal)?;
                let one = context.one(&expected_type)?;
                let zero = context.zero(&expected_type)?;
                C::Value::select(&selected, &one, &zero)?
            }
        };

        if value.r#type().as_ref() != &expected_type {
            return Err(TypeError::invalid(format!(
                "coordinate basis for value type {} has type {} but expected {}",
                value_type,
                value.r#type().as_ref(),
                expected_type,
            ))
            .into());
        }

        Ok(ArrayBatch::new(value, Some(0)).map_err(ProgramError::from)?)
    }

    #[inline]
    fn replicated(value: C::Value) -> Self::PackedValue {
        ArrayBatch::replicated(value)
    }

    #[inline]
    fn replay_derivative_region(
        context: &C,
        region: RegionRef<'_, C::Constant, C::Operation>,
        packed_direction_count: usize,
        inputs: Vec<Self::PackedValue>,
    ) -> Result<Vec<Self::PackedValue>, DifferentiationError> {
        let context = BatchingContext::new(context.clone(), packed_direction_count);
        Ok(ArrayBatching::batch_region(&context, region, inputs).map_err(ProgramError::from)?)
    }

    fn validate_hessian_block_type(
        block_type: &Self,
        output_type: &Self,
        first_input_type: &Self,
        second_input_type: &Self,
    ) -> Result<(), DifferentiationError> {
        let first_input_cotangent_type = first_input_type.cotangent()?;
        if first_input_cotangent_type.is_zero_space() {
            return Err(
                TypeError::invalid(format!("hessian input type {first_input_type} has no cotangent type")).into()
            );
        }
        validate_array_derivative_block_type(
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
        let input_shape = input_type.static_shape().ok_or_else(|| DifferentiationError::NonFiniteCoordinateSpace {
            transform: DerivativeTransform::JacobianForward,
            role: DifferentiationParameterRole::Input,
            path: ParameterPath::root().to_string(),
            r#type: input_type.to_string(),
        })?;
        let output_shape =
            output_type.static_shape().ok_or_else(|| DifferentiationError::NonFiniteCoordinateSpace {
                transform: DerivativeTransform::JacobianForward,
                role: DifferentiationParameterRole::Output,
                path: ParameterPath::root().to_string(),
                r#type: output_type.to_string(),
            })?;
        let output_tangent_type = output_type.tangent()?;
        if output_tangent_type.is_zero_space() {
            return Err(
                TypeError::invalid(format!("forward Jacobian output type {output_type} has no tangent type")).into()
            );
        }
        let value = unpack_coordinate_range(
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
        validate_array_derivative_block_type(
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
            output_type.static_shape().ok_or_else(|| DifferentiationError::NonFiniteCoordinateSpace {
                transform: DerivativeTransform::JacobianReverse,
                role: DifferentiationParameterRole::Output,
                path: ParameterPath::root().to_string(),
                r#type: output_type.to_string(),
            })?;
        let input_cotangent_type = input_type.cotangent()?;
        if input_cotangent_type.is_zero_space() {
            return Err(
                TypeError::invalid(format!("reverse Jacobian input type {input_type} has no cotangent type")).into()
            );
        }
        let value = unpack_coordinate_range(
            packed_output,
            packed_direction_count,
            output_coordinate_offset,
            output_shape.dimensions(),
            &input_cotangent_type,
        )?;
        validate_array_derivative_block_type(
            DerivativeTransform::JacobianReverse,
            value.r#type().as_ref(),
            &input_cotangent_type,
            &[output_type],
            &[],
        )?;
        Ok(value)
    }
}

/// Validates that `block_type` is the [`ArrayType`] expected for a materialized derivative block. The expected shape
/// consists of every static coordinate dimension in `prefix_coordinate_types`, followed by the dimensions of
/// `value_type`, followed by every static coordinate dimension in `suffix_coordinate_types`. The expected type keeps
/// the element type, memory space, and sharding of `value_type`. Inserted coordinate dimensions are replicated in its
/// sharding. Layout metadata is ignored because derivative materialization may change physical storage order without
/// changing the logical block type.
///
/// Returns [`DifferentiationError::NonFiniteCoordinateSpace`] if a coordinate type does not have a fully static shape,
/// or a [`TypeError`] wrapped in [`DifferentiationError`] if `block_type` differs from the expected type after removing
/// layout metadata.
///
/// # Parameters
///
///   - `transform`: Derivative transform being materialized, used when reporting a non-static coordinate space.
///   - `block_type`: Actual type of the materialized derivative block.
///   - `value_type`: Per-coordinate differential value type whose dimensions form the center of the block.
///   - `prefix_coordinate_types`: Coordinate types whose static dimensions must precede the `value_type` dimensions,
///     in slice order.
///   - `suffix_coordinate_types`: Coordinate types whose static dimensions must follow the `value_type` dimensions,
///     in slice order.
fn validate_array_derivative_block_type(
    transform: DerivativeTransform,
    block_type: &ArrayType,
    value_type: &ArrayType,
    prefix_coordinate_types: &[&ArrayType],
    suffix_coordinate_types: &[&ArrayType],
) -> Result<(), DifferentiationError> {
    let mut expected_type = value_type.clone().with_layout(None);
    let mut prefix_index = 0;
    for (coordinate_type, is_prefix) in prefix_coordinate_types
        .iter()
        .map(|r#type| (*r#type, true))
        .chain(suffix_coordinate_types.iter().map(|r#type| (*r#type, false)))
    {
        let coordinate_shape =
            coordinate_type.static_shape().ok_or_else(|| DifferentiationError::NonFiniteCoordinateSpace {
                transform,
                role: DifferentiationParameterRole::Derivative,
                path: ParameterPath::root().to_string(),
                r#type: coordinate_type.to_string(),
            })?;
        for size in coordinate_shape.dimensions() {
            let index = if is_prefix { prefix_index } else { expected_type.rank() };
            expected_type = expected_type.with_inserted_dimension(index, Dimension::Static(*size))?;
            if is_prefix {
                prefix_index += 1;
            }
        }
    }
    let block_type_without_layout = block_type.clone().with_layout(None);
    if block_type_without_layout != expected_type {
        return Err(
            TypeError::invalid(format!("derivative block has type {block_type} but expected {expected_type}")).into()
        );
    }
    Ok(())
}

/// Extracts one value's coordinate directions from `packed_output` and restores their logical coordinate shape. The
/// packed direction axis is first aligned to physical axis `0`, broadcasting a replicated output when necessary. The
/// function then verifies that the unbatched value type equals `expected_value_type` after ignoring layout metadata,
/// slices the consecutive directions beginning at `coordinate_offset`, and reshapes that leading range into
/// `coordinate_shape`. The returned value therefore has `coordinate_shape` followed by the dimensions of
/// `expected_value_type`. A coordinate shape containing a zero dimension selects an empty range, while a
/// scalar coordinate shape selects one direction. Returns a [`ProgramError`] if axis alignment, type validation,
/// coordinate-count arithmetic, slicing, or reshaping fails.
///
/// # Parameters
///
///   - `packed_output`: Derivative replay output whose batch representation is interpreted over the packed coordinate
///     directions.
///   - `packed_direction_count`: Total number of coordinate directions represented by `packed_output`.
///   - `coordinate_offset`: Index of the first packed direction belonging to the value being extracted.
///   - `coordinate_shape`: Static logical shape whose flattened coordinates occupy the selected direction range.
///   - `expected_value_type`: Expected type of each unpacked derivative value, excluding the packed direction axis.
fn unpack_coordinate_range<V: Value<Type = ArrayType> + Broadcast + Reshape + Slice + Transpose>(
    packed_output: &ArrayBatch<V>,
    packed_direction_count: usize,
    coordinate_offset: usize,
    coordinate_shape: &[usize],
    expected_value_type: &ArrayType,
) -> Result<V, ProgramError> {
    let aligned = packed_output.match_axis(0, packed_direction_count, ShardingDimension::Replicated)?;
    let actual_item_type = aligned.unbatched_type();
    if actual_item_type.clone().with_layout(None) != expected_value_type.clone().with_layout(None) {
        return Err(TypeError::invalid(format!(
            "batched derivative output has per-item type {actual_item_type} but expected {expected_value_type}",
        ))
        .into());
    }
    let item_shape = expected_value_type.static_shape().ok_or_else(|| {
        TypeError::invalid(format!(
            "Jacobian or Hessian materialization requires a fully static array shape but got {expected_value_type}",
        ))
    })?;
    let coordinate_shape = Shape::new(coordinate_shape.iter().copied().map(Dimension::Static).collect());
    let coordinate_count = coordinate_shape.element_count()?.unwrap();
    let coordinate_limit = coordinate_offset
        .checked_add(coordinate_count)
        .ok_or_else(|| ProgramError::InvalidArgument { message: "coordinate range overflows usize".to_string() })?;
    let physical_type = aligned.r#type();
    let physical_shape = physical_type.static_shape().ok_or_else(|| {
        TypeError::invalid(format!(
            "Jacobian or Hessian materialization requires a fully static array shape but got {physical_type}",
        ))
    })?;
    let mut start_indices = vec![0; physical_shape.rank()];
    start_indices[0] = coordinate_offset;
    let mut limit_indices = physical_shape.dimensions().to_vec();
    limit_indices[0] = coordinate_limit;
    let strides = vec![1; limit_indices.len()];
    let sliced = aligned.value().slice(&start_indices, &limit_indices, &strides)?;
    let reshaped_shape = Shape::new(
        coordinate_shape
            .dimensions()
            .iter()
            .cloned()
            .chain(item_shape.dimensions().iter().copied().map(Dimension::Static))
            .collect(),
    );
    sliced.reshape(reshaped_shape)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use pretty_assertions::assert_eq;

    use crate::arrays::DataType::*;
    use crate::arrays::{
        Array, ArrayBatch, ArrayOperation, ArrayType, Dimension, DimensionBounds, DimensionVariable, Layout,
        LogicalMesh, Memory, MeshAxis, MeshAxisType, Shape, Sharding, ShardingDimension, StridedLayout, f6e2m3fn,
    };
    use crate::batching::BatchAxis;
    use crate::contexts::EagerContext;
    use crate::programs::{Operation, ReferenceType};
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_data_type_differential_representations() {
        let non_differentiable = [Token, Boolean, I1, I2, I4, I8, I16, I32, I64, U1, U2, U4, U8, U16, U32, U64];
        for data_type in non_differentiable {
            assert_eq!(data_type.tangent(), Ok(Zero));
            assert_eq!(data_type.cotangent(), Ok(Zero));
        }
        assert!(Zero.is_zero_space());
        assert_eq!(Zero.tangent(), Ok(Zero));
        assert_eq!(Zero.cotangent(), Ok(Zero));

        let self_differentiable = [
            F4E2M1FN,
            F6E2M3FN,
            F6E3M2FN,
            F8E3M4,
            F8E4M3,
            F8E4M3FN,
            F8E4M3FNUZ,
            F8E4M3B11FNUZ,
            F8E5M2,
            F8E5M2FNUZ,
            BF16,
            F16,
            F32,
            F64,
            C64,
            C128,
        ];
        for data_type in self_differentiable {
            assert_eq!(data_type.tangent(), Ok(data_type));
            assert_eq!(data_type.cotangent(), Ok(data_type));
            assert!(!data_type.is_zero_space());
        }

        assert_eq!(F8E8M0FNU.tangent(), Ok(F32));
        assert_eq!(F8E8M0FNU.cotangent(), Ok(F32));
    }

    #[test]
    fn test_array_type_tangent() {
        let boolean = ArrayType::new(Boolean, Shape::new(vec![Dimension::Static(4)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![1])));
        assert_eq!(boolean.tangent(), Ok(boolean.clone().with_data_type(Zero).with_layout(None)));

        let token = boolean.clone().with_data_type(Token).with_memory(Memory::Host { pinned: true });
        assert_eq!(
            token.tangent(),
            Ok(boolean.with_data_type(Zero).with_layout(None).with_memory(Memory::Host { pinned: true })),
        );

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let primal = ArrayType::new(F8E8M0FNU, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap()
            .with_layout(Layout::Strided(StridedLayout::new(vec![1])))
            .with_memory(Memory::Host { pinned: true });
        let tangent = primal.clone().with_data_type(F32).with_layout(None);
        assert_eq!(primal.tangent(), Ok(tangent));

        // An unchanged element representation retains its explicit physical layout.
        let laid_out = ArrayType::new(F32, Shape::new(vec![Dimension::Static(4)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        assert_eq!(laid_out.tangent(), Ok(laid_out));
    }

    #[test]
    fn test_array_type_cotangent() {
        // A non-differentiable element type maps to a zero cotangent space with the same structural metadata.
        let boolean = ArrayType::new(Boolean, Shape::new(vec![Dimension::Static(4)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![1])));
        assert_eq!(boolean.cotangent(), Ok(boolean.clone().with_data_type(Zero).with_layout(None)));

        // A different non-differentiable element representation also clears its element-dependent layout while
        // preserving shape and memory.
        let token = boolean.clone().with_data_type(Token).with_memory(Memory::Host { pinned: true });
        assert_eq!(
            token.cotangent(),
            Ok(boolean.clone().with_data_type(Zero).with_layout(None).with_memory(Memory::Host { pinned: true })),
        );

        // Without a sharding, the cotangent type is the type itself.
        let plain = ArrayType::new(F32, Shape::new(vec![Dimension::Static(4)]));
        assert_eq!(plain.cotangent(), Ok(plain.clone()));

        // With a sharding, the unreduced and reduced axes are swapped.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let unreduced = plain
            .clone()
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_unreduced_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        let reduced = plain
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_reduced_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(unreduced.cotangent(), Ok(reduced.clone()));
        assert_eq!(reduced.cotangent(), Ok(unreduced.clone()));

        // E8M0 arrays use F32 cotangent elements while also transforming sharding and preserving other metadata.
        let e8m0 = unreduced
            .with_data_type(F8E8M0FNU)
            .with_layout(Layout::Strided(StridedLayout::new(vec![1])))
            .with_memory(Memory::Host { pinned: true });
        let e8m0_cotangent = reduced.with_data_type(F32).with_layout(None).with_memory(Memory::Host { pinned: true });
        assert_eq!(e8m0.cotangent(), Ok(e8m0_cotangent));

        // An unchanged element representation retains its explicit physical layout.
        let laid_out = ArrayType::new(F32, Shape::new(vec![Dimension::Static(4)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        assert_eq!(laid_out.cotangent(), Ok(laid_out));
    }

    #[test]
    fn test_array_ir_type_reference_tangent_and_cotangent_are_undefined() {
        let r#type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(F32)));
        let primal_type = r#type.to_string();
        assert!(!r#type.is_zero_space());
        assert_eq!(
            r#type.tangent(),
            Err(DifferentiationError::UndefinedTangentType { primal_type: primal_type.clone() }),
        );
        assert_eq!(r#type.cotangent(), Err(DifferentiationError::UndefinedCotangentType { primal_type }));
    }

    #[test]
    fn test_sharding_tangent() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("data", 4, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("model", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("manual", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::new(mesh, vec![ShardingDimension::sharded(["data"]), ShardingDimension::replicated()])
            .unwrap()
            .with_unreduced_axes(["model"])
            .unwrap()
            .with_varying_manual_axes(["manual"])
            .unwrap();
        let primal = ArrayType::new(F32, Shape::new(vec![Dimension::Static(4), Dimension::Static(2)]))
            .with_sharding(sharding.clone())
            .unwrap();

        // Forward tangents follow the primal data flow, so every sharding component remains unchanged.
        assert_eq!(primal.tangent().unwrap().sharding(), Some(&sharding));
    }

    #[test]
    fn test_sharding_cotangent() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("data", 4, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("model", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("manual", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["data"]), ShardingDimension::replicated()])
                .unwrap()
                .with_unreduced_axes(["model"])
                .unwrap()
                .with_varying_manual_axes(["manual"])
                .unwrap();

        // The cotangent swaps the unreduced and reduced sets and keeps all other state unchanged.
        let cotangent = sharding.cotangent();
        assert_eq!(cotangent.dimensions(), sharding.dimensions());
        assert_eq!(cotangent.unreduced_axes(), &BTreeSet::new());
        assert_eq!(cotangent.reduced_axes(), &BTreeSet::from(["model".to_string()]));
        assert_eq!(cotangent.varying_manual_axes(), &BTreeSet::from(["manual".to_string()]));

        // The swap is an involution.
        assert_eq!(cotangent.cotangent(), sharding);

        // Shardings without reduction state are their own cotangents.
        let replicated = Sharding::replicated(mesh, 2);
        assert_eq!(replicated.cotangent(), replicated);
    }

    #[test]
    fn test_dense_array_coordinate_space_dimension() {
        let transform = DerivativeTransform::JacobianForward;
        let role = DifferentiationParameterRole::Input;
        let path = ParameterPath::root();
        assert_eq!(
            <ArrayType as DenseDifferentiableType<EagerContext<Array, ArrayOperation<Array>>>>::coordinate_space_dimension(
                &ArrayType::scalar(F32),
                transform,
                role,
                &path,
            )
            .unwrap(),
            1,
        );
        assert_eq!(
            <ArrayType as DenseDifferentiableType<EagerContext<Array, ArrayOperation<Array>>>>::coordinate_space_dimension(
                &ArrayType::new(F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)])),
                transform,
                role,
                &path,
            )
            .unwrap(),
            6,
        );
        assert_eq!(
            <ArrayType as DenseDifferentiableType<EagerContext<Array, ArrayOperation<Array>>>>::coordinate_space_dimension(
                &ArrayType::new(
                    F32,
                    Shape::new(vec![Dimension::Static(usize::MAX), Dimension::Static(usize::MAX), Dimension::Static(0)]),
                ),
                transform,
                role,
                &path,
            )
            .unwrap(),
            0,
        );

        let dynamic_type = ArrayType::new(
            F32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded()))]),
        );
        assert_eq!(
            <ArrayType as DenseDifferentiableType<EagerContext<Array, ArrayOperation<Array>>>>::coordinate_space_dimension(
                &dynamic_type,
                transform,
                role,
                &path,
            )
            .unwrap_err(),
            DifferentiationError::NonFiniteCoordinateSpace {
                transform,
                role,
                path: path.to_string(),
                r#type: dynamic_type.to_string(),
            },
        );

        let overflowing_type =
            ArrayType::new(F32, Shape::new(vec![Dimension::Static(usize::MAX), Dimension::Static(2)]));
        assert_eq!(
            <ArrayType as DenseDifferentiableType<EagerContext<Array, ArrayOperation<Array>>>>::coordinate_space_dimension(
                &overflowing_type,
                transform,
                role,
                &path,
            )
            .unwrap_err(),
            DifferentiationError::CoordinateCountOverflow {
                transform,
                role,
                path: path.to_string(),
                r#type: overflowing_type.to_string(),
            },
        );
    }

    #[test]
    fn test_dense_array_coordinate_basis_uses_the_requested_value_type() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let coordinate_type = ArrayType::scalar(F32)
            .with_sharding(Sharding::new(mesh, Vec::new()).unwrap().with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let value_type = coordinate_type.cotangent().unwrap();
        let basis =
            <ArrayType as DenseDifferentiableType<EagerContext<Array, ArrayOperation<Array>>>>::coordinate_basis(
                &EagerContext::<Array, ArrayOperation<Array>>::new(),
                &coordinate_type,
                &value_type,
                0,
                1,
            )
            .unwrap();
        assert_eq!(basis.unbatched_type(), value_type);
    }

    #[test]
    fn test_dense_array_coordinate_basis_materializes_packed_fragments() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();

        // A rank-two value with a nonzero global coordinate offset exercises the rectangular two-iota path.
        let value_type = ArrayType::new(F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let basis =
            <ArrayType as DenseDifferentiableType<_>>::coordinate_basis(&context, &value_type, &value_type, 1, 8)
                .unwrap();
        let expected =
            (0..48).map(|index| if index / 6 == index % 6 + 1 { 1.0f32 } else { 0.0f32 }).collect::<Vec<_>>();
        assert_eq!(basis.value().elements::<f32>().unwrap(), expected);
        assert_eq!(basis.value().r#type().as_ref().static_shape().unwrap().dimensions(), &[8, 2, 3]);
        assert_eq!(basis.batch_axis(), BatchAxis::new(0));

        // Low-precision differential values retain their exact typed zero/one encodings.
        let value_type = ArrayType::new(F6E2M3FN, Shape::new(vec![Dimension::Static(2)]));
        let basis =
            <ArrayType as DenseDifferentiableType<_>>::coordinate_basis(&context, &value_type, &value_type, 1, 4)
                .unwrap();
        let zero = f6e2m3fn::from_bits(0).unwrap();
        let one = f6e2m3fn::from_bits(0x08).unwrap();
        assert_eq!(basis.value().elements::<f6e2m3fn>().unwrap(), vec![zero, zero, one, zero, zero, one, zero, zero]);
    }

    #[test]
    fn test_dense_array_coordinate_basis_preserves_non_contiguous_sharding() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 3, MeshAxisType::Explicit).unwrap()]).unwrap();
        let value_type = ArrayType::new(F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])]).unwrap(),
            )
            .unwrap();
        let basis = <ArrayType as DenseDifferentiableType<_>>::coordinate_basis(
            &EagerContext::<Array, ArrayOperation<Array>>::new(),
            &value_type,
            &value_type,
            0,
            6,
        )
        .unwrap();
        let expected = (0..36).map(|index| if index / 6 == index % 6 { 1.0f32 } else { 0.0f32 }).collect::<Vec<_>>();
        assert_eq!(basis.value().elements::<f32>().unwrap(), expected);
        assert_eq!(basis.unbatched_type(), value_type);
    }

    #[test]
    fn test_dense_array_coordinate_basis_stages_ordinary_primitives() {
        type ArrayTracingContext = TracingContext<Array, ArrayOperation<Array>>;

        let value_type = ArrayType::new(F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let (_, program) = ArrayTracingContext::trace(
            |input| {
                let context = input.context().clone();
                <ArrayType as DenseDifferentiableType<_>>::coordinate_basis(&context, &value_type, &value_type, 0, 6)
                    .map(ArrayBatch::into_value)
                    .map_err(|error| ProgramError::MalformedProgram(error.to_string()))
            },
            ArrayType::scalar(F32),
        )
        .unwrap();

        assert_eq!(
            program.instructions().iter().map(|instruction| instruction.operation().name()).collect::<Vec<_>>(),
            vec!["iota", "iota", "compare", "zero", "one", "select", "reshape"],
        );
    }

    #[test]
    fn test_dense_array_coordinate_basis_validates_its_coordinate_range() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let boolean_type = ArrayType::scalar(Boolean);
        assert_eq!(
            <ArrayType as DenseDifferentiableType<_>>::coordinate_basis(&context, &boolean_type, &boolean_type, 0, 1,)
                .unwrap_err()
                .to_string(),
            "coordinate basis requires a differentiable value type but got bool[]",
        );

        let narrow_type = ArrayType::scalar(F8E8M0FNU);
        assert_eq!(
            <ArrayType as DenseDifferentiableType<_>>::coordinate_basis(&context, &narrow_type, &narrow_type, 0, 1,)
                .unwrap_err()
                .to_string(),
            "coordinate basis values of type f8e8m0fnu[] cannot represent their own cotangents; use f32[] instead",
        );

        let dynamic_type = ArrayType::new(
            F32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded()))]),
        );
        assert_eq!(
            <ArrayType as DenseDifferentiableType<_>>::coordinate_basis(&context, &dynamic_type, &dynamic_type, 0, 1,)
                .unwrap_err()
                .to_string(),
            "coordinate basis requires a fully static value type but got f32[dynamic]",
        );

        let value_type = ArrayType::new(F32, Shape::new(vec![Dimension::Static(3)]));
        assert_eq!(
            <ArrayType as DenseDifferentiableType<_>>::coordinate_basis(&context, &value_type, &value_type, 2, 4,)
                .unwrap_err()
                .to_string(),
            "coordinate range [2, 5) exceeds coordinate count 4",
        );
    }

    #[test]
    fn test_dense_array_coordinate_basis_handles_zero_sized_values_without_coordinate_overflow() {
        let value_type = ArrayType::new(
            F32,
            Shape::new(vec![Dimension::Static(usize::MAX), Dimension::Static(usize::MAX), Dimension::Static(0)]),
        );
        let basis = <ArrayType as DenseDifferentiableType<_>>::coordinate_basis(
            &EagerContext::<Array, ArrayOperation<Array>>::new(),
            &value_type,
            &value_type,
            usize::MAX,
            usize::MAX,
        )
        .unwrap();
        assert!(basis.value().elements::<f32>().unwrap().is_empty());
        assert_eq!(basis.unbatched_type(), value_type);
    }

    #[test]
    fn test_dense_array_derivative_block_validation_uses_transform_value_types() {
        let error = <ArrayType as DenseDifferentiableType<
            EagerContext<Array, ArrayOperation<Array>>,
        >>::validate_hessian_block_type(
            &ArrayType::new(F32, Shape::new(vec![Dimension::Static(2)])),
            &ArrayType::scalar(F32),
            &ArrayType::scalar(F32),
            &ArrayType::scalar(F32),
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "derivative block has type f32[2] but expected f32[]");

        // Forward blocks carry the output leaf's tangent values while reverse blocks carry the input leaf's cotangent
        // values, so a packed replay output that stays in the narrow `f8e8m0fnu` storage type is rejected by both
        // extractors, whose expected per-item type is the widened `f32` differential representation.
        let narrow_type = ArrayType::scalar(F8E8M0FNU);
        let physical_type = ArrayType::new(F8E8M0FNU, Shape::new(vec![Dimension::Static(1)]));
        let packed = ArrayBatch::new(Array::from_f64s(physical_type, vec![2.0]), BatchAxis::new(0)).unwrap();
        assert_eq!(
            <ArrayType as DenseDifferentiableType<
                EagerContext<Array, ArrayOperation<Array>>,
            >>::extract_forward_jacobian_block(
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
            <ArrayType as DenseDifferentiableType<
                EagerContext<Array, ArrayOperation<Array>>,
            >>::extract_reverse_jacobian_block(
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

        let output_type = ArrayType::new(F64, Shape::new(vec![Dimension::Static(2)]));
        let input_type = ArrayType::new(F32, Shape::new(vec![Dimension::Static(3)]));
        let wrong_block_type = ArrayType::new(F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        assert_eq!(
            <ArrayType as DenseDifferentiableType<
                EagerContext<Array, ArrayOperation<Array>>,
            >>::validate_hessian_block_type(
                &wrong_block_type,
                &output_type,
                &ArrayType::scalar(F32),
                &input_type,
            )
            .unwrap_err()
            .to_string(),
            "derivative block has type f64[2, 3] but expected f32[2, 3]",
        );
    }
}
