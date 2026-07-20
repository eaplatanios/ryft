use std::fmt::{Display, Formatter};
use std::ops::{Add, Mul};

use crate::backends::scalars::Scalar;
use crate::batching::{ArrayBatch, BatchableOperation, BatchingContext, BatchingDriver, BatchingError};
use crate::contexts::{Context, Domain};
use crate::differentiation::DifferentiableType;
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_non_differentiable_operation, impl_nullary_transposable_operation};
use crate::operations::compare::{Compare, ComparisonDirection};
use crate::operations::constants::{Fill, Iota, One, Zero};
use crate::operations::control_flow::Select;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::ProgramError;
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{Type, TypeError};
use crate::types::{ArrayType, DataType, Size};

// TODO(eaplatanios): Review this.

/// Canonical operation name for [`CoordinateBasisOperation`].
pub const COORDINATE_BASIS_OPERATION_NAME: &str = "coordinate_basis";

/// Nullary operation that materializes one leaf's fragment of a packed global coordinate basis.
///
/// For a leaf with `n` row-major coordinates, the array specialization produces one value with physical type
/// `[basis_size] ++ leaf_type.shape`. Global row `k` contains the leaf-local one-hot coordinate
/// `k - coordinate_offset` when that coordinate belongs to the leaf and zero otherwise. This representation lets a
/// dense derivative transform replay all structured input or output coordinates in one batching transform.
#[derive(Clone, Debug)]
pub struct CoordinateBasisOperation<T: Type> {
    /// Unpacked type of the represented leaf.
    leaf_type: T,

    /// Offset of this leaf's first coordinate in the global packed basis.
    coordinate_offset: usize,

    /// Total number of coordinates in the global packed basis.
    basis_size: usize,
}

impl<T: Type> CoordinateBasisOperation<T> {
    /// Creates a packed coordinate-basis operation.
    ///
    /// # Parameters
    ///
    ///   - `leaf_type`: Unpacked type of the represented leaf.
    ///   - `coordinate_offset`: Offset of this leaf's first coordinate in the global packed basis.
    ///   - `basis_size`: Total number of coordinates in the global packed basis.
    #[inline]
    pub fn new(leaf_type: T, coordinate_offset: usize, basis_size: usize) -> Self {
        Self { leaf_type, coordinate_offset, basis_size }
    }

    /// Returns the unpacked type of the represented leaf.
    #[inline]
    pub fn leaf_type(&self) -> &T {
        &self.leaf_type
    }

    /// Returns the offset of this leaf's first coordinate in the global packed basis.
    #[inline]
    pub fn coordinate_offset(&self) -> usize {
        self.coordinate_offset
    }

    /// Returns the total number of coordinates in the global packed basis.
    #[inline]
    pub fn basis_size(&self) -> usize {
        self.basis_size
    }
}

impl<T: Type> Display for CoordinateBasisOperation<T> {
    #[inline]
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        OperationFormatter::new(formatter, 0, COORDINATE_BASIS_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("leaf_type", &self.leaf_type)?;
            operation.field("coordinate_offset", self.coordinate_offset)?;
            operation.field("basis_size", self.basis_size)
        })
    }
}

impl Operation<ArrayType> for CoordinateBasisOperation<ArrayType> {
    #[inline]
    fn name(&self) -> &'static str {
        COORDINATE_BASIS_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 0, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let cotangent_data_type = self.leaf_type.data_type().cotangent();
        if cotangent_data_type.is_zero_space() {
            return Err(TypeError {
                message: format!("coordinate basis requires a differentiable leaf type but got {}", self.leaf_type,),
            });
        }
        if cotangent_data_type != self.leaf_type.data_type() {
            return Err(TypeError {
                message: format!(
                    "coordinate basis values of type {} cannot represent their own cotangents; use {} instead",
                    self.leaf_type,
                    self.leaf_type.clone().with_data_type(cotangent_data_type),
                ),
            });
        }
        let dimensions = self.leaf_type.shape().dimensions();
        if dimensions.iter().any(|size| matches!(size, Size::Dynamic(_))) {
            return Err(TypeError {
                message: format!("coordinate basis requires a fully static leaf type but got {}", self.leaf_type),
            });
        }
        let coordinate_count = if dimensions.contains(&Size::Static(0)) {
            0
        } else {
            dimensions.iter().try_fold(1usize, |count, size| match size {
                Size::Static(size) => count.checked_mul(*size).ok_or_else(|| TypeError {
                    message: format!("coordinate count overflows usize for leaf type {}", self.leaf_type),
                }),
                Size::Dynamic(_) => unreachable!("dynamic dimensions were rejected above"),
            })?
        };
        let coordinate_end = self.coordinate_offset.checked_add(coordinate_count).ok_or_else(|| TypeError {
            message: format!("coordinate range overflows usize for leaf type {}", self.leaf_type),
        })?;
        if coordinate_end > self.basis_size {
            return Err(TypeError {
                message: format!(
                    "coordinate range [{}, {coordinate_end}) exceeds basis size {}",
                    self.coordinate_offset, self.basis_size,
                ),
            });
        }
        Ok(vec![self.leaf_type.with_inserted_dimension(0, Size::Static(self.basis_size))?])
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, COORDINATE_BASIS_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("leaf_type", &self.leaf_type)?;
            operation.field("coordinate_offset", self.coordinate_offset)?;
            operation.field("basis_size", self.basis_size)
        })
    }
}

impl<C> InterpretableOperation<C> for CoordinateBasisOperation<ArrayType>
where
    C: Domain<Type = ArrayType> + Fill<Scalar, C::Value> + Iota<C::Value> + One<C::Value> + Zero<C::Value>,
    C::Value: Add<Output = C::Value> + Mul<Output = C::Value> + Compare<Output = C::Value> + Select,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 0, ProgramError);
        let basis_type = self.infer_output_types(&[], &[])?.remove(0);
        let index_type = basis_type.clone().with_data_type(DataType::U64);
        let leaf_dimensions = self
            .leaf_type
            .static_shape()
            .ok_or_else(|| TypeError {
                message: format!("coordinate basis requires a fully static leaf type but got {}", self.leaf_type),
            })?
            .dimensions()
            .to_vec();

        // A zero-sized leaf has no local coordinates, so its packed basis fragment is the typed empty zero value.
        // Returning before row-major stride construction also makes the result independent of where the zero-sized
        // dimension appears and prevents irrelevant dimensions from overflowing the stride accumulator.
        if leaf_dimensions.contains(&0) {
            return Ok(vec![context.zero(&basis_type)?]);
        }

        let basis_index = context.iota(&index_type, 0)?;
        let mut flat_coordinate = None;
        let mut stride = 1u64;
        for (leaf_axis, dimension_size) in leaf_dimensions.iter().copied().enumerate().rev() {
            let coordinate = context.iota(&index_type, leaf_axis + 1)?;
            let coordinate = if stride == 1 {
                coordinate
            } else {
                let stride_value = context.fill(&index_type, Scalar::U64(stride))?;
                coordinate * stride_value
            };
            flat_coordinate = Some(match flat_coordinate {
                Some(accumulated) => accumulated + coordinate,
                None => coordinate,
            });
            stride = stride
                .checked_mul(u64::try_from(dimension_size).map_err(|_| ProgramError::InvalidArgument {
                    message: format!("leaf dimension {dimension_size} does not fit in u64"),
                })?)
                .ok_or_else(|| ProgramError::InvalidArgument {
                    message: format!("coordinate count overflows u64 for leaf type {}", self.leaf_type),
                })?;
        }
        let mut flat_coordinate = match flat_coordinate {
            Some(flat_coordinate) => flat_coordinate,
            None => context.fill(&index_type, Scalar::U64(0))?,
        };
        if self.coordinate_offset != 0 {
            let offset = u64::try_from(self.coordinate_offset).map_err(|_| ProgramError::InvalidArgument {
                message: format!("coordinate offset {} does not fit in u64", self.coordinate_offset),
            })?;
            let offset_value = context.fill(&index_type, Scalar::U64(offset))?;
            flat_coordinate = flat_coordinate + offset_value;
        }

        let selected = basis_index.compare(&flat_coordinate, ComparisonDirection::Equal)?;
        let one = context.one(&basis_type)?;
        let zero = context.zero(&basis_type)?;
        Ok(vec![C::Value::select(&selected, &one, &zero)?])
    }
}

// Keep the dedicated operation intact when batching into a staging parent so that backends can lower the packed basis
// directly. The generic replicated-nullary rule interprets its operation and would expand this primitive instead.
impl<C> BatchableOperation<C> for CoordinateBasisOperation<ArrayType>
where
    C: Context<Type = ArrayType>,
    C::Operation: From<CoordinateBasisOperation<ArrayType>>,
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        check_count!("input", inputs, 0, ProgramError);
        Ok(context
            .parent()
            .bind(self.clone(), Vec::new(), &[])?
            .into_iter()
            .map(ArrayBatch::replicated)
            .collect())
    }
}

impl_non_differentiable_operation!(CoordinateBasisOperation<ArrayType>);
impl_nullary_transposable_operation!(CoordinateBasisOperation<ArrayType>);

impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for CoordinateBasisOperation<ArrayType> where
    C::Operation: From<CoordinateBasisOperation<ArrayType>>
{
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::Array;
    use crate::contexts::EagerContext;
    use crate::interpretation::InterpretableOperation;
    use crate::macros::check_operation_type_inference;
    use crate::programs::operations::Operation;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::types::DataType::{Boolean, F8E8M0FNU, F32, I32};
    use crate::types::{ArrayType, Shape, Size};

    use super::*;

    #[test]
    fn test_coordinate_basis_operation_infers_packed_type() {
        let leaf_type = ArrayType::new(F32, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let operation = CoordinateBasisOperation::new(leaf_type.clone(), 4, 10);
        check_operation_type_inference!(
            operation = operation,
            cases = [{
                type = ArrayType,
                input_types = [],
                output_types = [leaf_type.with_inserted_dimension(0, Size::Static(10)).unwrap()],
            }],
        );
    }

    #[test]
    fn test_coordinate_basis_operation_renders_semantic_attributes() {
        let operation = CoordinateBasisOperation::new(
            ArrayType::new(F32, Shape::new(vec![Size::Static(2), Size::Static(3)])),
            4,
            10,
        );

        assert_eq!(operation.to_string(), "coordinate_basis [leaf_type=f32[2, 3], coordinate_offset=4, basis_size=10]",);
    }

    #[test]
    fn test_coordinate_basis_operation_rejects_non_finite_or_out_of_range_coordinates() {
        for data_type in [Boolean, I32] {
            let leaf_type = ArrayType::scalar(data_type);
            assert_eq!(
                CoordinateBasisOperation::new(leaf_type.clone(), 0, 1)
                    .infer_output_types(&[], &[])
                    .unwrap_err()
                    .message,
                format!("coordinate basis requires a differentiable leaf type but got {leaf_type}"),
            );
        }

        let e8m0_type = ArrayType::scalar(F8E8M0FNU);
        assert_eq!(
            CoordinateBasisOperation::new(e8m0_type.clone(), 0, 1)
                .infer_output_types(&[], &[])
                .unwrap_err()
                .message,
            format!(
                "coordinate basis values of type {e8m0_type} cannot represent their own cotangents; use {} instead",
                e8m0_type.clone().with_data_type(F32),
            ),
        );

        let dynamic_type = ArrayType::new(F32, Shape::new(vec![Size::Dynamic(None)]));
        assert_eq!(
            CoordinateBasisOperation::new(dynamic_type, 0, 1).infer_output_types(&[], &[]).unwrap_err().message,
            "coordinate basis requires a fully static leaf type but got f32[*]",
        );

        let leaf_type = ArrayType::new(F32, Shape::new(vec![Size::Static(3)]));
        assert_eq!(
            CoordinateBasisOperation::new(leaf_type, 2, 4).infer_output_types(&[], &[]).unwrap_err().message,
            "coordinate range [2, 5) exceeds basis size 4",
        );

        let overflowing_type = ArrayType::new(F32, Shape::new(vec![Size::Static(usize::MAX), Size::Static(2)]));
        assert_eq!(
            CoordinateBasisOperation::new(overflowing_type, 0, usize::MAX)
                .infer_output_types(&[], &[])
                .unwrap_err()
                .message,
            format!("coordinate count overflows usize for leaf type f32[{}, 2]", usize::MAX),
        );

        let zero_coordinate_type =
            ArrayType::new(F32, Shape::new(vec![Size::Static(usize::MAX), Size::Static(2), Size::Static(0)]));
        assert_eq!(
            CoordinateBasisOperation::new(zero_coordinate_type.clone(), usize::MAX, usize::MAX)
                .infer_output_types(&[], &[])
                .unwrap(),
            vec![zero_coordinate_type.with_inserted_dimension(0, Size::Static(usize::MAX)).unwrap()],
        );
    }

    #[test]
    fn test_coordinate_basis_operation_interprets_zero_sized_leaf_without_stride_overflow() {
        let leaf_type =
            ArrayType::new(F32, Shape::new(vec![Size::Static(0), Size::Static(usize::MAX), Size::Static(2)]));
        let operation = CoordinateBasisOperation::new(leaf_type.clone(), 0, 0);
        let context = EagerContext::<Array, CoordinateBasisOperation<ArrayType>>::new();
        assert_eq!(
            operation.interpret(&context, &EmptyRegionDriver, &[]).unwrap(),
            vec![Array::from_f64s(leaf_type.with_inserted_dimension(0, Size::Static(0)).unwrap(), Vec::new(),)],
        );
    }
}
