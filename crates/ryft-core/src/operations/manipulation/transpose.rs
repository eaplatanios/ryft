use std::fmt::{Debug, Display};
use std::ops::Deref;

use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::{Operation, OperationFormatter};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{ProgramError, Value};
use crate::regions::RegionInterface;
use crate::sharding::{Sharding, ShardingError};
use crate::types::{ArrayType, Shape, TypeError, Typed};

/// Canonical operation name for [`TransposeOperation`].
pub const TRANSPOSE_OPERATION_NAME: &str = "transpose";

/// An axis permutation used by [`TransposeOperation`] and the [`Transpose`] capability. For each output axis `i`,
/// `permutation[i]` is the input axis routed to it. [`Permutation`] is a thin wrapper over the axis vector. It
/// [`Deref`]s to `[usize]` and implements `AsRef<[usize]>`, and so it composes with everything that already accepts
/// an axis slice. Also, it supports [`From`] conversion from an owned `Vec<usize>` or a borrowed `&[usize]`. Validity
/// (i.e., being a bijection of `0..len`) is not enforced at construction time. It is validated against a concrete input
/// rank by the type-level [`Transpose`] rule.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Permutation(Vec<usize>);

impl Permutation {
    /// Returns the permutation axes as a slice, where element `i` is the input axis routed to output axis `i`.
    #[inline]
    pub fn as_slice(&self) -> &[usize] {
        self.0.as_slice()
    }

    /// Returns the inverse [`Permutation`] of this one (i.e., the one that _undoes_ it). Transposing by a permutation
    /// and then by its inverse restores the original axis order.
    #[inline]
    pub fn inverse(&self) -> Permutation {
        let mut inverse = vec![0usize; self.0.len()];
        for (position, axis) in self.0.iter().enumerate() {
            inverse[*axis] = position;
        }
        Permutation(inverse)
    }
}

impl Deref for Permutation {
    type Target = [usize];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<[usize]> for Permutation {
    #[inline]
    fn as_ref(&self) -> &[usize] {
        &self.0
    }
}

impl From<Vec<usize>> for Permutation {
    #[inline]
    fn from(axes: Vec<usize>) -> Self {
        Self(axes)
    }
}

impl From<&[usize]> for Permutation {
    #[inline]
    fn from(axes: &[usize]) -> Self {
        Self(axes.to_vec())
    }
}

/// [`Operation`] that reorders the axes of its input array according to a static permutation. The output shape is the
/// input shape with its axes permuted. Output dimension `i` is set to input dimension `permutation[i]`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct TransposeOperation {
    /// Axis [`Permutation`] of this [`TransposeOperation`].
    permutation: Permutation,
}

impl TransposeOperation {
    /// Creates a new [`TransposeOperation`] with the provided axis permutation.
    #[inline]
    pub fn new<P: Into<Permutation>>(permutation: P) -> Self {
        Self { permutation: permutation.into() }
    }

    /// Returns the axis [`Permutation`] of this [`TransposeOperation`].
    #[inline]
    pub fn permutation(&self) -> &Permutation {
        &self.permutation
    }
}

impl Display for TransposeOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation<ArrayType> for TransposeOperation {
    #[inline]
    fn name(&self) -> &'static str {
        TRANSPOSE_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        match (&input_types[0]).transpose(&self.permutation) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError { message: error.to_string() }),
        }
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("permutation", format_args!("{:?}", self.permutation.as_slice())))
    }
}

impl<C: Domain<Type = ArrayType, Value: Transpose>> InterpretableOperation<C> for TransposeOperation {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].transpose(&self.permutation)?])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType, Operation: From<TransposeOperation>>> PartiallyEvaluatableOperation<C>
    for TransposeOperation
{
}

/// Represents the ability to transpose the axes of an array. [`Transpose`] fills the same role for
/// [`TransposeOperation`] that [`std::ops::Add`] and [`std::ops::Neg`] fill for their corresponding arithmetic
/// [`Operation`]s.
pub trait Transpose: Sized {
    /// Reorders the axes of `self` according to the provided permutation, validating that the permutation is a
    /// bijection of the input axes. The permutation is accepted as any `AsRef<[usize]>` (for example, an owned
    /// `Vec<usize>` or a borrowed `&[usize]`), so callers can transpose without allocating a fresh permutation.
    fn transpose<P: AsRef<[usize]>>(&self, permutation: P) -> Result<Self, ProgramError>;

    /// Moves axis `from` to position `to`, shifting the other axes to preserve their relative order. This is the
    /// analogue of NumPy's [`moveaxis`](https://numpy.org/doc/stable/reference/generated/numpy.moveaxis.html).
    /// Returns `self` unchanged when `from == to`.
    #[inline]
    fn move_axis(&self, from: usize, to: usize) -> Result<Self, ProgramError>
    where
        Self: Typed<Type = ArrayType>,
    {
        let rank = self.r#type().rank();
        self.transpose(
            (0..rank)
                .filter(move |&axis| axis != from)
                .take(to)
                .chain([from])
                .chain((0..rank).filter(move |&axis| axis != from).skip(to))
                .collect::<Vec<_>>(),
        )
    }

    /// Swaps axes `i` and `j`, leaving every other axis in place. This is the analogue of NumPy's
    /// [`swapaxes`](https://numpy.org/doc/stable/reference/generated/numpy.swapaxes.html).
    /// Returns `self` unchanged when `i == j`. An out-of-bounds axis yields a [`TypeError`] rather than a panic.
    #[inline]
    fn swap_axes(&self, i: usize, j: usize) -> Result<Self, ProgramError>
    where
        Self: Typed<Type = ArrayType>,
    {
        let rank = self.r#type().rank();
        for axis in [i, j] {
            if axis >= rank {
                return Err(TypeError {
                    message: format!("'{TRANSPOSE_OPERATION_NAME}' swap axis {axis} is out of bounds"),
                }
                .into());
            }
        }
        let mut permutation = (0..rank).collect::<Vec<_>>();
        permutation.swap(i, j);
        self.transpose(permutation)
    }
}

impl Transpose for Sharding {
    /// Reorders the per-dimension [`ShardingDimension`](crate::ShardingDimension) entries so that output dimension `i`
    /// carries the entry of input dimension `permutation[i]`, while the reduction-state and manual-axis sets are left
    /// unchanged. This is the sharding-level analogue of an array axis permutation. `permutation` must be a permutation
    /// of `0..rank` matching this sharding's rank; otherwise a type error describing the offending dimension is
    /// returned.
    fn transpose<P: AsRef<[usize]>>(&self, permutation: P) -> Result<Self, ProgramError> {
        let permutation = permutation.as_ref();
        let permute_dimensions = || -> Result<Self, ShardingError> {
            if permutation.len() != self.dimensions().len() {
                return Err(ShardingError::DimensionOutOfBounds { dimension: permutation.len(), rank: self.rank() });
            }
            let mut dimensions = Vec::with_capacity(self.dimensions().len());
            for axis in permutation {
                let dimension = self
                    .dimensions()
                    .get(*axis)
                    .ok_or(ShardingError::DimensionOutOfBounds { dimension: *axis, rank: self.rank() })?;
                dimensions.push(dimension.clone());
            }
            Sharding::with_manual_axes(
                self.mesh().clone(),
                dimensions,
                self.unreduced_axes().clone(),
                self.reduced_axes().clone(),
                self.varying_manual_axes().clone(),
            )
        };
        permute_dimensions().map_err(|error| TypeError { message: error.to_string() }.into())
    }
}

impl Transpose for ArrayType {
    /// Type-level transpose: validates that `permutation` has length equal to the input rank and is a permutation of
    /// `0..rank` (every axis in range, no duplicates), then permutes the shape and the output sharding. Output axis
    /// `i` carries input axis `permutation[i]`.
    fn transpose<P: AsRef<[usize]>>(&self, permutation: P) -> Result<Self, ProgramError> {
        let permutation = permutation.as_ref();
        let input = self;
        let rank = input.rank();
        if permutation.len() != rank {
            return Err(TypeError {
                message: format!(
                    "'{TRANSPOSE_OPERATION_NAME}' permutation has length {} but input has rank {rank}",
                    permutation.len(),
                ),
            }
            .into());
        }
        let mut seen = vec![false; rank];
        for axis in permutation {
            if *axis >= rank {
                return Err(TypeError {
                    message: format!("'{TRANSPOSE_OPERATION_NAME}' permutation axis {axis} is out of bounds"),
                }
                .into());
            }
            if seen[*axis] {
                return Err(TypeError {
                    message: format!("'{TRANSPOSE_OPERATION_NAME}' permutation contains duplicate axis {axis}"),
                }
                .into());
            }
            seen[*axis] = true;
        }
        let permuted = permutation.iter().map(|axis| input.dimension(*axis as isize)).collect::<Vec<_>>();

        // The output sharding permutes its dimension entries the same way as the array axes: the reduction-state and
        // manual-axis sets are unchanged. This mirrors JAX's `_transpose_sharding_rule` and is correct for every mesh
        // axis type (it is a pure reordering, not explicit-mode reasoning).
        let sharding = input.sharding().map(|sharding| sharding.transpose(permutation)).transpose()?;

        ArrayType::new(input.data_type(), Shape::new(permuted))
            .with_sharding(sharding)
            .map_err(|error| TypeError { message: error.to_string() }.into())
    }
}

impl<V: Value<Type = ArrayType, DispatchDomain: Context<Type = ArrayType, Operation: From<TransposeOperation>>>>
    Transpose for V
{
    #[inline]
    fn transpose<P: AsRef<[usize]>>(&self, permutation: P) -> Result<Self, ProgramError> {
        let permutation = permutation.as_ref();
        if permutation.iter().enumerate().all(|(index, axis)| index == *axis) {
            return Ok(self.clone());
        }
        Ok(self
            .dispatch_domain()
            .bind(TransposeOperation::new(permutation.to_vec()), Vec::new(), &[self.clone()])?
            .remove(0))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
    use crate::regions::EmptyRegionDriver;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tests::TestArray;
    use crate::types::{DataType, Size, Typed};

    use super::*;

    #[test]
    fn test_permutation() {
        // Empty and identity permutations are their own inverses.
        assert_eq!(Permutation::from(vec![]).inverse(), Permutation::from(vec![]));
        assert_eq!(Permutation::from(vec![0, 1, 2]).inverse(), Permutation::from(vec![0, 1, 2]));

        // A swap is its own inverse, while a cycle inverts to the reverse cycle.
        assert_eq!(Permutation::from(vec![1, 0]).inverse(), Permutation::from(vec![1, 0]));
        assert_eq!(Permutation::from(vec![2, 0, 1]).inverse(), Permutation::from(vec![1, 2, 0]));

        // Inverting twice recovers the original permutation, and applying the inverse after the permutation restores
        // the identity ordering.
        let permutation = Permutation::from(vec![3, 0, 2, 1]);
        let inverse = permutation.inverse();
        assert_eq!(inverse.inverse(), permutation);
        let composed = inverse.iter().map(|axis| permutation[*axis]).collect::<Vec<_>>();
        assert_eq!(composed, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_transpose() {
        let operation = TransposeOperation::new(vec![1, 0]);

        // Operation identity and accessors.
        assert_eq!(operation.name(), TRANSPOSE_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "TransposeOperation { permutation: Permutation([1, 0]) }");
        assert_eq!(format!("{operation}"), "transpose [permutation=[1, 0]]");
        assert_eq!(operation.permutation().as_slice(), &[1, 0]);

        // Type inference permutes the input shape, including dynamic dimension sizes.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3), Size::Static(2)]));
        assert_eq!(operation.infer_output_types(std::slice::from_ref(&input_type), &[]), Ok(vec![output_type.clone()]));
        assert_eq!(
            operation.infer_output_types(
                &[ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None), Size::Dynamic(Some(4))]))],
                &[],
            ),
            Ok(vec![ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(Some(4)), Size::Dynamic(None)]))]),
        );

        // Interpretation reorders the row-major payload.
        let input = TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let output = operation
            .interpret(&EagerContext::<TestArray>::new(), &EmptyRegionDriver, std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(*output[0].r#type(), output_type);
        assert_eq!(output[0].values, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            operation.infer_output_types(&[], &[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)]))], &[]),
            Err(TypeError { message: "'transpose' permutation has length 2 but input has rank 1".to_string() }),
        );
        assert_eq!(
            TransposeOperation::new(vec![0, 2]).infer_output_types(std::slice::from_ref(&input_type), &[]),
            Err(TypeError { message: "'transpose' permutation axis 2 is out of bounds".to_string() }),
        );
        assert_eq!(
            TransposeOperation::new(vec![0, 0]).infer_output_types(std::slice::from_ref(&input_type), &[]),
            Err(TypeError { message: "'transpose' permutation contains duplicate axis 0".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<TestArray>>::interpret(
                &operation,
                &EagerContext::<TestArray>::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );

        // Program rendering uses the canonical operation name and includes the captured permutation.
        let mut builder = ProgramBuilder::<TestArray, TransposeOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_output = builder.add_instruction(operation, vec![program_input], Vec::new()).unwrap()[0];
        let program = builder.build::<TestArray, TestArray>(vec![program_output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2, 3] .
                let %1:f64[3, 2] = transpose [permutation=[1, 0]] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_transpose_permutes_sharding_dimensions() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("r", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        // Input dimensions are sharded over `x` and `y`; the reduced manual axis `r` rides along untouched.
        let input_sharding = Sharding::with_manual_axes(
            mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"]), ShardingDimension::replicated()],
            Vec::<&str>::new(),
            ["r"],
            Vec::<&str>::new(),
        )
        .unwrap();
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3), Size::Static(4)]))
                .with_sharding(input_sharding)
                .unwrap();

        // Permutation [2, 0, 1] makes output dimension i carry input dimension permutation[i].
        let operation = TransposeOperation::new(vec![2, 0, 1]);
        let expected =
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(4), Size::Static(2), Size::Static(3)]))
                .with_sharding(
                    Sharding::with_manual_axes(
                        mesh,
                        vec![
                            ShardingDimension::replicated(),
                            ShardingDimension::sharded(["x"]),
                            ShardingDimension::sharded(["y"]),
                        ],
                        Vec::<&str>::new(),
                        ["r"],
                        Vec::<&str>::new(),
                    )
                    .unwrap(),
                )
                .unwrap();
        assert_eq!(operation.infer_output_types(std::slice::from_ref(&input_type), &[]), Ok(vec![expected]));

        // An input without a sharding yields an output without one.
        let unsharded =
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3), Size::Static(4)]));
        assert_eq!(operation.infer_output_types(std::slice::from_ref(&unsharded), &[]).unwrap()[0].sharding(), None);
    }

    #[test]
    fn test_transpose_test_array() {
        // Rank-2 swap of a row-major 2x3 payload.
        let output = TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).transpose(vec![1, 0]).unwrap();
        assert_eq!(output.r#type, ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3), Size::Static(2)])));
        assert_eq!(output.values, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        // Rank-3 permutation moving the last axis to the front.
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3), Size::Static(4)]));
        let values = (0..24).map(|value| value as f64).collect::<Vec<_>>();
        let output = TestArray::new(input_type, values).transpose(vec![2, 0, 1]).unwrap();
        assert_eq!(
            output.r#type,
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(4), Size::Static(2), Size::Static(3)])),
        );
        assert_eq!(
            output.values,
            vec![
                0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 1.0, 5.0, 9.0, 13.0, 17.0, 21.0, 2.0, 6.0, 10.0, 14.0, 18.0, 22.0,
                3.0, 7.0, 11.0, 15.0, 19.0, 23.0,
            ],
        );

        // Rank-0 and empty payloads pass through unchanged.
        let output = TestArray::scalar(42.0).transpose(vec![]).unwrap();
        assert_eq!(output.r#type, ArrayType::scalar(DataType::F64));
        assert_eq!(output.values, vec![42.0]);
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(0), Size::Static(2)]));
        let output = TestArray::new(input_type, Vec::new()).transpose(vec![1, 0]).unwrap();
        assert_eq!(output.r#type, ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(0)])));
        assert_eq!(output.values, Vec::<f64>::new());

        // An invalid permutation is a clean error rather than an out-of-bounds panic, since the value-level transpose
        // validates the permutation through the type-level rule before indexing.
        let matrix = || TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(matrix().transpose(vec![1]).is_err());
        assert!(matrix().transpose(vec![0, 2]).is_err());
        assert!(matrix().transpose(vec![0, 0]).is_err());
    }

    #[test]
    fn test_move_axis() {
        // On a matrix, moving axis 0 to position 1 is a plain transpose: the [2, 3] payload becomes [3, 2].
        let matrix = TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let output = matrix.move_axis(0, 1).unwrap();
        assert_eq!(output.r#type, ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3), Size::Static(2)])));
        assert_eq!(output.values, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        // On a rank-3 array, moving axis 0 to the last position shifts the other axes left to preserve their relative
        // order, so [2, 3, 4] becomes [3, 4, 2] (the permutation [1, 2, 0]).
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3), Size::Static(4)]));
        let values = (0..24).map(|value| value as f64).collect::<Vec<_>>();
        let output = TestArray::new(input_type, values).move_axis(0, 2).unwrap();
        assert_eq!(
            output.r#type,
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3), Size::Static(4), Size::Static(2)])),
        );

        // `from == to` leaves the array unchanged.
        assert_eq!(matrix.move_axis(1, 1).unwrap(), matrix);

        // An out-of-bounds source axis is a clean error rather than an out-of-bounds panic, since the built
        // permutation is validated by the type-level transpose.
        assert!(matrix.move_axis(2, 0).is_err());
    }

    #[test]
    fn test_swap_axes() {
        // Swapping axes 0 and 1 of a matrix is a plain transpose: the [2, 3] payload becomes [3, 2].
        let matrix = TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let swapped = matrix.swap_axes(0, 1).unwrap();
        assert_eq!(swapped.r#type, ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3), Size::Static(2)])));
        assert_eq!(swapped.values, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        // Swapping is symmetric in its axis arguments.
        assert_eq!(matrix.swap_axes(1, 0).unwrap(), swapped);

        // Swapping the outer two axes of a rank-3 array leaves the untouched trailing axis in place, so [2, 3, 4]
        // becomes [3, 2, 4] (the permutation [1, 0, 2]).
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3), Size::Static(4)]));
        let values = (0..24).map(|value| value as f64).collect::<Vec<_>>();
        let output = TestArray::new(input_type, values).swap_axes(0, 1).unwrap();
        assert_eq!(
            output.r#type,
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3), Size::Static(2), Size::Static(4)])),
        );

        // `i == j` leaves the array unchanged.
        assert_eq!(matrix.swap_axes(1, 1).unwrap(), matrix);

        // An out-of-bounds axis is a clean error rather than an out-of-bounds panic.
        assert!(matrix.swap_axes(2, 0).is_err());
    }
}
