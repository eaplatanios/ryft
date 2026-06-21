use std::fmt::Display;

use crate::contexts::StagingContext;
use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::programs::{ProgramError, Value};
use crate::tracing::Tracer;
use crate::types::{ArrayType, Shape, Size, TypeError};

use super::slicing::resized_output_sharding;

// TODO(eaplatanios): Review from here onwards.

/// Canonical operation name for [`PadOperation`].
pub const PAD_OPERATION_NAME: &'static str = "pad";

/// [`Operation`] that expands its first operand by adding edge and interior padding filled with its second (scalar)
/// operand. Refer to the documentation of [`Pad`] for more information.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PadOperation {
    /// Padding added before the first element of each input axis.
    edge_padding_low: Vec<usize>,

    /// Padding added after the last element of each input axis.
    edge_padding_high: Vec<usize>,

    /// Padding added between any two adjacent elements of each input axis.
    interior_padding: Vec<usize>,
}

impl PadOperation {
    /// Creates a new [`PadOperation`] with the provided edge and interior padding amounts. The three vectors must
    /// share one length (one entry per input axis); whether that shared length matches the input rank is validated
    /// during type inference, once an input type is known.
    pub fn new(
        edge_padding_low: Vec<usize>,
        edge_padding_high: Vec<usize>,
        interior_padding: Vec<usize>,
    ) -> Result<Self, ProgramError> {
        if edge_padding_low.len() != edge_padding_high.len() || edge_padding_low.len() != interior_padding.len() {
            return Err(TypeError {
                message: format!(
                    "pad expects edge_padding_low, edge_padding_high, and interior_padding to share one length but \
                    got lengths {}, {}, and {}",
                    edge_padding_low.len(),
                    edge_padding_high.len(),
                    interior_padding.len(),
                ),
            }
            .into());
        }
        Ok(Self { edge_padding_low, edge_padding_high, interior_padding })
    }

    /// Returns the padding added before the first element of each input axis.
    #[inline]
    pub fn edge_padding_low(&self) -> &[usize] {
        self.edge_padding_low.as_slice()
    }

    /// Returns the padding added after the last element of each input axis.
    #[inline]
    pub fn edge_padding_high(&self) -> &[usize] {
        self.edge_padding_high.as_slice()
    }

    /// Returns the padding added between any two adjacent elements of each input axis.
    #[inline]
    pub fn interior_padding(&self) -> &[usize] {
        self.interior_padding.as_slice()
    }
}

impl Display for PadOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation<ArrayType> for PadOperation {
    #[inline]
    fn name(&self) -> &'static str {
        PAD_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        match input_types[0].pad(
            &input_types[1],
            self.edge_padding_low.as_slice(),
            self.edge_padding_high.as_slice(),
            self.interior_padding.as_slice(),
        ) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError { message: error.to_string() }),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("edge_padding_low", format_args!("{:?}", self.edge_padding_low))?;
            operation.field("edge_padding_high", format_args!("{:?}", self.edge_padding_high))?;
            operation.field("interior_padding", format_args!("{:?}", self.interior_padding))
        })
    }
}

impl<V: Value<ArrayType> + Pad> InterpretableOperation<ArrayType, V> for PadOperation {
    fn interpret(
        &self,
        _context: &<V as Value<ArrayType>>::InterpretationContext,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        Ok(vec![inputs[0].pad(
            &inputs[1],
            self.edge_padding_low.as_slice(),
            self.edge_padding_high.as_slice(),
            self.interior_padding.as_slice(),
        )?])
    }
}

/// Represents the ability to expand an array by adding edge and interior padding filled with a scalar padding value.
/// This is the direct analogue of the StableHLO [`pad`](https://openxla.org/stablehlo/spec#pad) operation and JAX's
/// [`lax.pad`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.pad.html), restricted to non-negative padding
/// amounts (StableHLO also allows negative edge padding, which trims elements instead; that form is not supported).
///
/// `t.pad(padding_value, edge_padding_low, edge_padding_high, interior_padding)` returns an array that holds the
/// input element with index `i` at output index `edge_padding_low + i * (interior_padding + 1)` along each axis and
/// `padding_value` everywhere else. The output dimension along an axis whose input dimension is `d` is:
///
///   - `edge_padding_low + edge_padding_high` when `d == 0` (there are no elements, so no interior padding is
///     inserted and the output holds only the edge padding), and
///   - `edge_padding_low + (d - 1) * (interior_padding + 1) + 1 + edge_padding_high` otherwise (`d` elements with
///     `interior_padding` padding elements between each adjacent pair).
///
/// All three padding slices must have length equal to the input rank, and the padding value must be a rank-0 scalar
/// with the input's data type. Padding requires static input extents: inputs with dynamic dimensions are rejected
/// because the padded extent cannot be computed from an unknown extent.
///
/// [`Pad`] is the transpose dual of strided [`Slice`](crate::operations::manipulation::Slice): slicing with stride
/// `s` keeps every `s`-th element, while padding with `interior_padding = s - 1` puts elements back at every `s`-th
/// position.
///
/// # Example
///
/// The following example shows how to use [`Pad`] in practice:
///
/// ```rust
/// # use ryft_core::operations::manipulation::Pad;
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::tests::{TestArray as Array};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Pad [1, 2, 3] with one leading zero, two trailing zeros, and one zero between adjacent elements. With
/// // d = 3, low = 1, high = 2, and interior = 1, the output dimension is 1 + (3 - 1) * 2 + 1 + 2 = 8 and the
/// // input elements land at output positions 1, 3, and 5.
/// let x = Array::vector(vec![1.0, 2.0, 3.0]);
/// let y = x.pad(&Array::scalar(0.0), &[1], &[2], &[1])?;
/// assert_eq!(y.values, vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0]);
/// # Ok(())
/// # }
/// ```
pub trait Pad: Sized {
    /// Pads `self` with `padding_value` using the provided edge and interior padding amounts. Refer to the
    /// documentation of this trait for more information on what this operation does.
    ///
    /// # Parameters
    ///
    ///   - `padding_value`: Rank-0 scalar with the input's data type, written into every padding position.
    ///   - `edge_padding_low`: Padding added before the first element of each input axis.
    ///   - `edge_padding_high`: Padding added after the last element of each input axis.
    ///   - `interior_padding`: Padding added between any two adjacent elements of each input axis.
    fn pad(
        &self,
        padding_value: &Self,
        edge_padding_low: &[usize],
        edge_padding_high: &[usize],
        interior_padding: &[usize],
    ) -> Result<Self, ProgramError>;
}

impl Pad for ArrayType {
    fn pad(
        &self,
        padding_value: &Self,
        edge_padding_low: &[usize],
        edge_padding_high: &[usize],
        interior_padding: &[usize],
    ) -> Result<Self, ProgramError> {
        if self.data_type() != padding_value.data_type() {
            return Err(TypeError {
                message: format!(
                    "pad input data type {} does not match padding value data type {}",
                    self.data_type(),
                    padding_value.data_type(),
                ),
            }
            .into());
        }
        if padding_value.rank() != 0 {
            return Err(TypeError {
                message: format!("pad padding value must be a scalar but has type {padding_value}"),
            }
            .into());
        }
        let rank = self.rank();
        for (name, padding) in [
            ("edge_padding_low", edge_padding_low),
            ("edge_padding_high", edge_padding_high),
            ("interior_padding", interior_padding),
        ] {
            if padding.len() != rank {
                return Err(TypeError {
                    message: format!("pad {name} has length {} but input has rank {rank}", padding.len()),
                }
                .into());
            }
        }
        let mut output_dimensions = Vec::with_capacity(rank);
        for axis in 0..rank {
            let dimension = self.dimension(axis as isize);
            let Size::Static(size) = dimension else {
                return Err(TypeError {
                    message: format!(
                        "pad does not support dynamic input axis {axis} with size {dimension}; the padded extent \
                        cannot be computed from an unknown extent",
                    ),
                }
                .into());
            };
            let interior = if size == 0 { 0 } else { (size - 1) * (interior_padding[axis] + 1) + 1 };
            output_dimensions.push(Size::Static(edge_padding_low[axis] + interior + edge_padding_high[axis]));
        }
        // Padding resizes dimensions in place, so the operand sharding (placement and reduction state) carries
        // through, with the same divisibility check on padded sharded dimensions that `slice` applies (JAX's
        // `_pad_sharding_rule` reuses the shared `_get_sharding_for_varying_out_shape`). The scalar padding value's
        // sharding does not affect the output.
        let sharding = resized_output_sharding(self, &output_dimensions, PAD_OPERATION_NAME)?;
        ArrayType::new(self.data_type(), Shape::new(output_dimensions))
            .with_sharding(sharding)
            .map_err(|error| TypeError { message: error.to_string() }.into())
    }
}

impl<C: StagingContext<Type = ArrayType, Operation: From<PadOperation>>> Pad for Tracer<C> {
    fn pad(
        &self,
        padding_value: &Self,
        edge_padding_low: &[usize],
        edge_padding_high: &[usize],
        interior_padding: &[usize],
    ) -> Result<Self, ProgramError> {
        let mut outputs = self.context().stage_operation(
            PadOperation::new(edge_padding_low.to_vec(), edge_padding_high.to_vec(), interior_padding.to_vec())?,
            &[self, padding_value],
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
    use crate::tests::TestArray;
    use crate::types::{DataType, Typed};

    use super::*;

    #[test]
    fn test_pad() {
        let operation = PadOperation::new(vec![1], vec![2], vec![1]).unwrap();

        // Operation identity and accessors.
        assert_eq!(operation.name(), PAD_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "pad [edge_padding_low=[1], edge_padding_high=[2], interior_padding=[1]]");
        assert_eq!(operation.edge_padding_low(), &[1]);
        assert_eq!(operation.edge_padding_high(), &[2]);
        assert_eq!(operation.interior_padding(), &[1]);

        // Type inference validates the padding geometry and returns the padded type, and the type-level (abstract)
        // capability backs it without consuming the borrowed input type. With d = 3, low = 1, high = 2, and
        // interior = 1, the output dimension is 1 + (3 - 1) * 2 + 1 + 2 = 8.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        let padding_value_type = ArrayType::scalar(DataType::F64);
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(8)]));
        assert_eq!(
            operation.infer_output_types(&[input_type.clone(), padding_value_type.clone()]),
            Ok(vec![output_type.clone()]),
        );
        assert_eq!(input_type.pad(&padding_value_type, &[1], &[2], &[1]), Ok(output_type.clone()));

        // Interpretation writes the input elements at `low + i * (interior + 1)` (positions 1, 3, and 5) and fills
        // every other position with the padding value.
        let input = TestArray::vector(vec![1.0, 2.0, 3.0]);
        let output = operation.interpret(&crate::EagerContext::new(), &[input, TestArray::scalar(9.0)]).unwrap();
        assert_eq!(*output[0].r#type(), output_type);
        assert_eq!(output[0].values, vec![9.0, 1.0, 9.0, 2.0, 9.0, 3.0, 9.0, 9.0]);

        // Empty input axes hold only the edge padding (the `d == 0` case skips interior padding entirely) and
        // rank-0 inputs pass through unchanged.
        let empty_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(0)]));
        assert_eq!(
            (&empty_type).pad(&padding_value_type, &[1], &[2], &[1]),
            Ok(ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]))),
        );
        let empty = TestArray::new(empty_type, vec![]).pad(&TestArray::scalar(7.0), &[1], &[2], &[1]).unwrap();
        assert_eq!(empty.values, vec![7.0, 7.0, 7.0]);
        let scalar = TestArray::scalar(42.0).pad(&TestArray::scalar(7.0), &[], &[], &[]).unwrap();
        assert_eq!(scalar.values, vec![42.0]);

        // Invalid construction and inputs report precise operation and interpreter errors.
        assert_eq!(
            PadOperation::new(vec![1], vec![2, 0], vec![1]),
            Err(ProgramError::Type(TypeError {
                message: "pad expects edge_padding_low, edge_padding_high, and interior_padding to share one length \
                    but got lengths 1, 2, and 1"
                    .to_string(),
            })),
        );
        assert_eq!(
            operation.infer_output_types(&[input_type.clone()]),
            Err(TypeError { message: "expected 2 inputs but got 1".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[input_type.clone(), ArrayType::scalar(DataType::F32)]),
            Err(TypeError {
                message: "pad input data type f64 does not match padding value data type f32".to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(&[input_type.clone(), input_type.clone()]),
            Err(TypeError { message: "pad padding value must be a scalar but has type f64[3]".to_string() }),
        );
        assert_eq!(
            PadOperation::new(vec![1, 0], vec![2, 0], vec![1, 0])
                .unwrap()
                .infer_output_types(&[input_type.clone(), padding_value_type.clone()]),
            Err(TypeError { message: "pad edge_padding_low has length 2 but input has rank 1".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None)])),
                padding_value_type.clone(),
            ]),
            Err(TypeError {
                message: "pad does not support dynamic input axis 0 with size *; the padded extent cannot be \
                    computed from an unknown extent"
                    .to_string(),
            }),
        );
        assert_eq!(
            InterpretableOperation::<ArrayType, TestArray>::interpret(&operation, &crate::EagerContext::new(), &[]),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        );

        // Program rendering uses the canonical operation name and includes all three padding vectors.
        let mut builder = ProgramBuilder::<ArrayType, TestArray, PadOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_padding_value = builder.add_input(padding_value_type);
        let program_output = builder.add_instruction(operation, vec![program_input, program_padding_value]).unwrap()[0];
        let program = builder
            .build::<Vec<TestArray>, TestArray>(vec![program_output], vec![Placeholder, Placeholder], Placeholder)
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[] .
                let %2:f64[8] = pad [edge_padding_low=[1], edge_padding_high=[2], interior_padding=[1]] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_pad_test_array_kernel() {
        // A rank-2 pad exercises the odometer across axes with different padding amounts: rows gain one interior
        // row and columns gain asymmetric edge padding.
        let input = TestArray::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let output = input.pad(&TestArray::scalar(0.0), &[0, 1], &[1, 0], &[1, 0]).unwrap();
        assert_eq!(*output.r#type(), ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(4), Size::Static(3)])),);
        assert_eq!(output.values, vec![0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0],);

        // The kernel validates the padding value shape eagerly.
        assert_eq!(
            TestArray::vector(vec![1.0, 2.0]).pad(&TestArray::vector(vec![0.0]), &[0], &[0], &[0]),
            Err(ProgramError::Type(TypeError {
                message: "pad padding value must be a scalar but has type f64[1]".to_string(),
            })),
        );
    }

    #[test]
    fn test_pad_propagates_sharding() {
        use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};

        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        // [4] sharded over `x` and unreduced over the manual axis `m`.
        let sharding = Sharding::with_manual_axes(
            mesh.clone(),
            vec![ShardingDimension::sharded(["x"])],
            ["m"],
            Vec::<&str>::new(),
            Vec::<&str>::new(),
        )
        .unwrap();
        let input = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]))
            .with_sharding(sharding.clone())
            .unwrap();
        let pad_value = ArrayType::scalar(DataType::F32);

        // Padding to an evenly divisible size keeps the operand sharding (including the unreduced manual axis): with
        // low = 0, interior = 0, and high = 4 the output is 0 + 4 + 4 = 8, divisible by the `x` mesh-axis size (2).
        assert_eq!(input.pad(&pad_value, &[0], &[4], &[0]).unwrap().sharding(), Some(&sharding));
        // Padding to a size not divisible by the explicit mesh-axis size (output 0 + 4 + 1 = 5) is rejected.
        assert!(input.pad(&pad_value, &[0], &[1], &[0]).is_err());
    }
}
