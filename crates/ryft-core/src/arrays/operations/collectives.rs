//! Array IR instantiations of the collectives operation family contracts.
//!
//! Collective operations are homogeneous over ordinary array data, so the composite array IR ordinarily lifts them
//! into its array member family. Operations with explicit composite boundaries, including shape-changing collectives
//! with first-class dimensions and ragged all-to-all's direct six-array carrier, are declared directly by
//! [`ArrayIrOperation`] instead.

use crate::arrays::addressing::ArrayAddressing;
use crate::arrays::arrays::Array;
use crate::arrays::encoding::{i1, i2, i4, u1, u2, u4};
use crate::arrays::macros::dispatch_on_array_element_type;
use crate::arrays::operations::{ArrayIrOperation, ArrayOperation};
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::dimensions::{Dimension, Shape};
use crate::operations::collectives::ragged_all_to_all::{RaggedAllToAllEvaluation, RaggedAllToAllUpdateKind};
use crate::operations::collectives::{
    ParallelPermuteOperation, RAGGED_ALL_TO_ALL_OPERATION_NAME, RaggedAllToAllOperation,
};
use crate::operations::math::add::Add;
use crate::programs::{ProgramError, Typed, Value};

// TODO(eaplatanios): Review this.

impl<A: Value<Type = ArrayType>> From<ParallelPermuteOperation> for ArrayIrOperation<A> {
    #[inline]
    fn from(operation: ParallelPermuteOperation) -> Self {
        Self::Array(ArrayOperation::ParallelPermute(operation))
    }
}

/// Conversion into a host integer used to validate ragged collective metadata without narrowing unsigned values.
trait MetadataInteger {
    /// Converts this integer exactly into the common host representation.
    fn to_i128(self) -> i128;
}

// Implements exact host widening for the native integer element types accepted as collective metadata.
macro_rules! impl_metadata_integer {
    ($($type:ty),* $(,)?) => {
        $(
            impl MetadataInteger for $type {
                #[inline]
                fn to_i128(self) -> i128 {
                    self as i128
                }
            }
        )*
    };
}

impl_metadata_integer!(i8, i16, i32, i64, u8, u16, u32, u64);

// Implements exact host widening for Ryft's checked sub-byte integer element wrappers.
macro_rules! impl_sub_byte_metadata_integer {
    ($($type:ty),* $(,)?) => {
        $(
            impl MetadataInteger for $type {
                #[inline]
                fn to_i128(self) -> i128 {
                    self.value() as i128
                }
            }
        )*
    };
}

impl_sub_byte_metadata_integer!(i1, i2, i4, u1, u2, u4);

/// Decodes one statically typed integer metadata array and rejects negative or host-unrepresentable entries.
pub(crate) fn decode_nonnegative_integer_metadata(
    metadata: &Array,
    operation_name: &str,
    name: &str,
) -> Result<Vec<usize>, ProgramError> {
    dispatch_on_array_element_type!(@integer metadata.r#type().data_type(), |Element| {
        metadata
            .elements::<Element>()?
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                let value = value.to_i128();
                if value < 0 {
                    return Err(ProgramError::InvalidArgument {
                        message: format!(
                            "`{operation_name}` `{name}[{index}]` must be nonnegative but got \
                             {value}",
                        ),
                    });
                }
                usize::try_from(value).map_err(|_| ProgramError::InvalidArgument {
                    message: format!(
                        "`{operation_name}` `{name}[{index}]` value {value} does not fit in `usize`",
                    ),
                })
            })
            .collect()
    })
}

impl RaggedAllToAllEvaluation for Array {
    fn evaluate_ragged_all_to_all(
        operation: &RaggedAllToAllOperation,
        operand: &Self,
        output: &Self,
        input_offsets: &Self,
        send_sizes: &Self,
        output_offsets: &Self,
        receive_sizes: &Self,
    ) -> Result<Self, ProgramError> {
        let batched = operation.is_physical();
        let input_offsets =
            decode_nonnegative_integer_metadata(input_offsets, RAGGED_ALL_TO_ALL_OPERATION_NAME, "input_offsets")?;
        let send_sizes =
            decode_nonnegative_integer_metadata(send_sizes, RAGGED_ALL_TO_ALL_OPERATION_NAME, "send_sizes")?;
        let output_offsets =
            decode_nonnegative_integer_metadata(output_offsets, RAGGED_ALL_TO_ALL_OPERATION_NAME, "output_offsets")?;
        let receive_sizes =
            decode_nonnegative_integer_metadata(receive_sizes, RAGGED_ALL_TO_ALL_OPERATION_NAME, "receive_sizes")?;
        let participant_count = if batched { operation.axis_size() } else { 1 };
        let metadata_length = input_offsets.len() / participant_count;
        let input_extent = operand.r#type().shape().dimensions()[usize::from(batched)].value().unwrap();
        let output_extent = output.r#type().shape().dimensions()[usize::from(batched)].value().unwrap();
        let groups = if batched {
            operation
                .axis_index_groups()
                .map_or_else(|| vec![(0..participant_count).collect()], |groups| groups.to_vec())
        } else {
            vec![vec![0]]
        };
        let trailing_start = usize::from(batched) + 1;
        let row_element_count = operand.r#type().shape().dimensions()[trailing_start..]
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(dimension.value().unwrap()))
            .ok_or_else(|| ProgramError::InvalidArgument {
                message: format!("`{RAGGED_ALL_TO_ALL_OPERATION_NAME}` trailing row size does not fit in `usize`"),
            })?;
        let row_byte_count =
            row_element_count
                .checked_mul(ArrayAddressing::new(operand.r#type().into_owned())?.element_byte_width())
                .ok_or_else(|| ProgramError::InvalidArgument {
                    message: format!(
                        "`{RAGGED_ALL_TO_ALL_OPERATION_NAME}` trailing row byte size does not fit in `usize`",
                    ),
                })?;

        // Validate the complete exchange before copying anything. `output_offsets` are sender-owned metadata in the
        // receiver coordinate frame, while `receive_sizes` are indexed receiver-first and sender-second.
        let overwrite = operation.update_kind() == RaggedAllToAllUpdateKind::Overwrite;
        let mut received_regions = overwrite.then(|| vec![Vec::new(); participant_count]);
        let mut transfers = Vec::new();
        for group in &groups {
            let slices_per_peer = metadata_length / group.len();
            for (sender_position, &sender) in group.iter().enumerate() {
                for (receiver_position, &receiver) in group.iter().enumerate() {
                    for slice in 0..slices_per_peer {
                        let send_index = receiver_position * slices_per_peer + slice;
                        let receive_index = sender_position * slices_per_peer + slice;
                        let sender_metadata_index = sender * metadata_length + send_index;
                        let receiver_metadata_index = receiver * metadata_length + receive_index;
                        let input_offset = input_offsets[sender_metadata_index];
                        let send_size = send_sizes[sender_metadata_index];
                        let output_offset = output_offsets[sender_metadata_index];
                        let receive_size = receive_sizes[receiver_metadata_index];
                        let input_end =
                            input_offset.checked_add(send_size).ok_or_else(|| ProgramError::InvalidArgument {
                                message: format!(
                                    "`{RAGGED_ALL_TO_ALL_OPERATION_NAME}` input region for participant {sender} at \
                                     metadata index {send_index} overflows `usize`",
                                ),
                            })?;
                        if input_end > input_extent {
                            return Err(ProgramError::InvalidArgument {
                                message: format!(
                                    "`{RAGGED_ALL_TO_ALL_OPERATION_NAME}` input region [{input_offset}, {input_end}) \
                                     for participant {sender} exceeds input extent {input_extent}",
                                ),
                            });
                        }
                        if send_size != receive_size {
                            return Err(ProgramError::InvalidArgument {
                                message: format!(
                                    "`{RAGGED_ALL_TO_ALL_OPERATION_NAME}` send size {send_size} from participant \
                                     {sender} to participant {receiver} does not match receive size {receive_size}",
                                ),
                            });
                        }
                        let output_end =
                            output_offset.checked_add(receive_size).ok_or_else(|| ProgramError::InvalidArgument {
                                message: format!(
                                    "`{RAGGED_ALL_TO_ALL_OPERATION_NAME}` output region for participant {receiver} \
                                     from participant {sender} overflows `usize`",
                                ),
                            })?;
                        if output_end > output_extent {
                            return Err(ProgramError::InvalidArgument {
                                message: format!(
                                    "`{RAGGED_ALL_TO_ALL_OPERATION_NAME}` output region [{output_offset}, \
                                     {output_end}) for participant {receiver} exceeds output extent {output_extent}",
                                ),
                            });
                        }
                        if receive_size != 0
                            && let Some(received_regions) = &mut received_regions
                        {
                            received_regions[receiver].push((output_offset, output_end));
                        }
                        if send_size != 0 && row_byte_count != 0 {
                            let source_row = sender
                                .checked_mul(input_extent)
                                .and_then(|offset| offset.checked_add(input_offset))
                                .ok_or_else(|| ProgramError::InvalidArgument {
                                    message: format!(
                                        "`{RAGGED_ALL_TO_ALL_OPERATION_NAME}` source byte offset for participant \
                                         {sender} does not fit in `usize`",
                                    ),
                                })?;
                            let destination_row = receiver
                                .checked_mul(output_extent)
                                .and_then(|offset| offset.checked_add(output_offset))
                                .ok_or_else(|| ProgramError::InvalidArgument {
                                    message: format!(
                                        "`{RAGGED_ALL_TO_ALL_OPERATION_NAME}` destination byte offset for \
                                         participant {receiver} does not fit in `usize`",
                                    ),
                                })?;
                            let source_start = source_row.checked_mul(row_byte_count).ok_or_else(|| {
                                ProgramError::InvalidArgument {
                                    message: format!(
                                        "`{RAGGED_ALL_TO_ALL_OPERATION_NAME}` source byte offset for participant \
                                         {sender} does not fit in `usize`",
                                    ),
                                }
                            })?;
                            let destination_start = destination_row.checked_mul(row_byte_count).ok_or_else(|| {
                                ProgramError::InvalidArgument {
                                    message: format!(
                                        "`{RAGGED_ALL_TO_ALL_OPERATION_NAME}` destination byte offset for \
                                         participant {receiver} does not fit in `usize`",
                                    ),
                                }
                            })?;
                            let byte_count =
                                send_size.checked_mul(row_byte_count).ok_or_else(|| ProgramError::InvalidArgument {
                                    message: format!(
                                        "`{RAGGED_ALL_TO_ALL_OPERATION_NAME}` transfer byte size does not fit in \
                                         `usize`",
                                    ),
                                })?;
                            transfers.push((source_start, destination_start, byte_count, send_size));
                        }
                    }
                }
            }
        }
        if let Some(mut received_regions) = received_regions {
            for (receiver, regions) in received_regions.iter_mut().enumerate() {
                regions.sort_unstable();
                for regions in regions.windows(2) {
                    if regions[1].0 < regions[0].1 {
                        return Err(ProgramError::InvalidArgument {
                            message: format!(
                                "`{RAGGED_ALL_TO_ALL_OPERATION_NAME}` received output regions [{}, {}) and [{}, {}) \
                                 overlap for participant {receiver}",
                                regions[0].0, regions[0].1, regions[1].0, regions[1].1,
                            ),
                        });
                    }
                }
            }
        }

        let operand_bytes = operand.logical_bytes();
        let mut result_bytes = output.logical_bytes();
        for (source_start, destination_start, byte_count, row_count) in transfers {
            let source = &operand_bytes[source_start..source_start + byte_count];
            let destination = &mut result_bytes[destination_start..destination_start + byte_count];
            if overwrite {
                destination.copy_from_slice(source);
            } else {
                let mut dimensions = vec![Dimension::Static(row_count)];
                dimensions.extend(operand.r#type().shape().dimensions()[trailing_start..].iter().cloned());
                let segment_type = ArrayType::new(operand.r#type().data_type(), Shape::new(dimensions));
                let source = Array::from_logical_bytes(segment_type.clone(), source)?;
                let destination_array = Array::from_logical_bytes(segment_type, destination)?;
                destination.copy_from_slice(destination_array.add(&source)?.logical_bytes().as_slice());
            }
        }
        Array::from_logical_bytes(output.r#type().into_owned(), result_bytes.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::arrays::Array;
    use crate::arrays::batching::ArrayIrBatching;
    use crate::arrays::dimensions::DimensionValue;
    use crate::arrays::ir::ArrayIrValue;
    use crate::arrays::operations::{ArrayIrOperation, ArrayOperation, DimensionOperation};
    use crate::arrays::types::arrays::ArrayType;
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::dimensions::{Dimension, DimensionBounds, DimensionType, DimensionVariable, Shape};
    use crate::arrays::types::ir::ArrayIrType;
    use crate::axes::NamedAxis;
    use crate::batching::{BatchAxis, BatchAxisSpecification, BatchingTracer, batch};
    use crate::contexts::{Context, EagerContext};
    use crate::differentiation::DifferentiationError;
    use crate::macros::check_operation_partial_evaluation;
    use crate::operations::collectives::{
        AllGather, AllGatherOperation, AllGatherOutputVariance, AllToAllOperation, CollectiveOptions,
        ParallelSumScatter, ParallelSumScatterOperation, RaggedAllToAll, RaggedAllToAllOperation,
    };
    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError, TypeError, Typed};
    use crate::tracing::TracingContext;

    // Executes a degenerate single-participant ragged exchange with i64 metadata.
    fn interpret_single_participant_ragged_all_to_all(
        input_offsets: Vec<i64>,
        send_sizes: Vec<i64>,
        output_offsets: Vec<i64>,
        receive_sizes: Vec<i64>,
    ) -> Result<Array, ProgramError> {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let mut outputs = context.bind(
            RaggedAllToAllOperation::new("x".to_string(), 1),
            Vec::new(),
            &[
                Array::vector(vec![1.0_f32, 2.0, 3.0]),
                Array::vector(vec![9.0_f32, 9.0, 9.0, 9.0]),
                Array::vector(input_offsets),
                Array::vector(send_sizes),
                Array::vector(output_offsets),
                Array::vector(receive_sizes),
            ],
        )?;
        Ok(outputs.remove(0))
    }

    // Applies an explicit list of logical row transfers without deriving routing from the operation metadata.
    fn reference_ragged_transfers(
        operand: &[i32],
        output: &[i32],
        input_extent: usize,
        output_extent: usize,
        row_width: usize,
        transfers: &[(usize, usize, usize, usize, usize)],
    ) -> Vec<i32> {
        let mut result = output.to_vec();
        for &(sender, input_offset, receiver, output_offset, size) in transfers {
            for row in 0..size {
                let source = (sender * input_extent + input_offset + row) * row_width;
                let destination = (receiver * output_extent + output_offset + row) * row_width;
                result[destination..destination + row_width].copy_from_slice(&operand[source..source + row_width]);
            }
        }
        result
    }

    #[test]
    fn test_array_ragged_all_to_all_eager_metadata_validation() {
        assert_eq!(
            interpret_single_participant_ragged_all_to_all(vec![1], vec![2], vec![0], vec![2]),
            Ok(Array::vector(vec![2.0_f32, 3.0, 9.0, 9.0])),
        );
        assert_eq!(
            interpret_single_participant_ragged_all_to_all(vec![-1], vec![1], vec![0], vec![1]).unwrap_err(),
            ProgramError::InvalidArgument {
                message: "`ragged_all_to_all` `input_offsets[0]` must be nonnegative but got -1".to_string(),
            },
        );
        assert_eq!(
            interpret_single_participant_ragged_all_to_all(vec![2], vec![2], vec![0], vec![2]).unwrap_err(),
            ProgramError::InvalidArgument {
                message: "`ragged_all_to_all` input region [2, 4) for participant 0 exceeds input extent 3".to_string(),
            },
        );
        assert_eq!(
            interpret_single_participant_ragged_all_to_all(vec![0], vec![2], vec![0], vec![1]).unwrap_err(),
            ProgramError::InvalidArgument {
                message: "`ragged_all_to_all` send size 2 from participant 0 to participant 0 does not match receive \
                          size 1"
                    .to_string(),
            },
        );
        assert_eq!(
            interpret_single_participant_ragged_all_to_all(vec![0], vec![2], vec![3], vec![2]).unwrap_err(),
            ProgramError::InvalidArgument {
                message: "`ragged_all_to_all` output region [3, 5) for participant 0 exceeds output extent 4"
                    .to_string(),
            },
        );
        assert_eq!(
            interpret_single_participant_ragged_all_to_all(vec![0, 1], vec![1, 1], vec![0, 0], vec![1, 1],)
                .unwrap_err(),
            ProgramError::InvalidArgument {
                message: "`ragged_all_to_all` received output regions [0, 1) and [0, 1) overlap for participant 0"
                    .to_string(),
            },
        );
        assert_eq!(
            interpret_single_participant_ragged_all_to_all(vec![0, 0], vec![1, 1], vec![0, 1], vec![1, 1]),
            Ok(Array::vector(vec![1.0_f32, 1.0, 9.0, 9.0])),
        );

        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        assert_eq!(
            context
                .bind(
                    RaggedAllToAllOperation::new("x".to_string(), 1),
                    Vec::new(),
                    &[
                        Array::vector(vec![1.0_f32]),
                        Array::vector(vec![0.0_f32]),
                        Array::vector(vec![u64::MAX]),
                        Array::vector(vec![1_u64]),
                        Array::vector(vec![0_u64]),
                        Array::vector(vec![1_u64]),
                    ],
                )
                .unwrap_err(),
            ProgramError::InvalidArgument {
                message: "`ragged_all_to_all` input region for participant 0 at metadata index 0 overflows `usize`"
                    .to_string(),
            },
        );
    }

    #[test]
    fn test_array_ragged_all_to_all_matches_documented_and_grouped_reference_exchanges() {
        type TestContext = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        type TestTracer = BatchingTracer<TestContext, ArrayIrBatching>;

        let operand = vec![1_i32, 2, 2, 3, 4, 0];
        let output_seed = vec![0_i32; 8];
        let input_offsets = vec![0_usize, 1, 0, 1];
        let send_sizes = vec![1_usize, 2, 1, 1];
        let output_offsets = vec![0_usize, 0, 1, 2];
        let receive_sizes = vec![1_i32, 1, 2, 1];
        let output: ArrayIrValue<Array> = batch(
            |inputs: Vec<TestTracer>| {
                inputs[0].ragged_all_to_all("x", &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5])
            },
            vec![
                ArrayIrValue::Array(Array::matrix(2, 3, operand.clone())),
                ArrayIrValue::Array(Array::matrix(2, 4, output_seed.clone())),
                ArrayIrValue::Array(Array::matrix(2, 2, input_offsets.iter().map(|value| *value as i32).collect())),
                ArrayIrValue::Array(Array::matrix(2, 2, send_sizes.iter().map(|value| *value as i32).collect())),
                ArrayIrValue::Array(Array::matrix(2, 2, output_offsets.iter().map(|value| *value as i32).collect())),
                ArrayIrValue::Array(Array::matrix(2, 2, receive_sizes)),
            ],
            vec![BatchAxis::new(0); 6],
            BatchAxis::new(0),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();
        let expected = reference_ragged_transfers(
            operand.as_slice(),
            output_seed.as_slice(),
            3,
            4,
            1,
            &[(0, 0, 0, 0, 1), (0, 1, 1, 0, 2), (1, 0, 0, 1, 1), (1, 1, 1, 2, 1)],
        );
        assert_eq!(output, ArrayIrValue::Array(Array::matrix(2, 4, expected)));

        // Reversed noncontiguous groups, two slices per peer, and width-two rows exercise every routing index and
        // prove that the byte kernel preserves trailing dimensions.
        let groups = vec![vec![3, 1], vec![2, 0]];
        let operand = (0..4)
            .flat_map(|participant| {
                (0..4).flat_map(move |row| [participant * 100 + row * 10, participant * 100 + row * 10 + 1])
            })
            .collect::<Vec<i32>>();
        let output_seed = vec![-1_i32; 4 * 5 * 2];
        let input_offsets = [0_usize, 1, 2, 3].repeat(4);
        let send_sizes = vec![1_usize; 16];
        let output_offsets = vec![2, 3, 2, 3, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1];
        let receive_sizes = vec![1_i32; 16];
        let output: ArrayIrValue<Array> = batch(
            |inputs: Vec<TestTracer>| {
                inputs[0].ragged_all_to_all_with_axis_index_groups(
                    "x",
                    &inputs[1],
                    &inputs[2],
                    &inputs[3],
                    &inputs[4],
                    &inputs[5],
                    groups.clone(),
                )
            },
            vec![
                ArrayIrValue::Array(
                    Array::from_elements(ArrayType::new_static(DataType::I32, [4, 4, 2]), operand.as_slice()).unwrap(),
                ),
                ArrayIrValue::Array(
                    Array::from_elements(ArrayType::new_static(DataType::I32, [4, 5, 2]), output_seed.as_slice())
                        .unwrap(),
                ),
                ArrayIrValue::Array(Array::matrix(4, 4, input_offsets.iter().map(|value| *value as i32).collect())),
                ArrayIrValue::Array(Array::matrix(4, 4, send_sizes.iter().map(|value| *value as i32).collect())),
                ArrayIrValue::Array(Array::matrix(4, 4, output_offsets.iter().map(|value| *value as i32).collect())),
                ArrayIrValue::Array(Array::matrix(4, 4, receive_sizes)),
            ],
            vec![BatchAxis::new(0); 6],
            BatchAxis::new(0),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();
        let expected = reference_ragged_transfers(
            operand.as_slice(),
            output_seed.as_slice(),
            4,
            5,
            2,
            &[
                (3, 0, 3, 0, 1),
                (3, 1, 3, 1, 1),
                (3, 2, 1, 0, 1),
                (3, 3, 1, 1, 1),
                (1, 0, 3, 2, 1),
                (1, 1, 3, 3, 1),
                (1, 2, 1, 2, 1),
                (1, 3, 1, 3, 1),
                (2, 0, 2, 0, 1),
                (2, 1, 2, 1, 1),
                (2, 2, 0, 0, 1),
                (2, 3, 0, 1, 1),
                (0, 0, 2, 2, 1),
                (0, 1, 2, 3, 1),
                (0, 2, 0, 2, 1),
                (0, 3, 0, 3, 1),
            ],
        );
        assert_eq!(
            output,
            ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::I32, [4, 5, 2]), expected.as_slice()).unwrap(),
            ),
        );
    }

    #[test]
    fn test_array_ir_explicit_collective_eager_contracts() {
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]));
        let extent = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());

        assert_eq!(
            context.bind(
                AllGatherOperation::new(
                    "x".to_string(),
                    1,
                    0,
                    CollectiveOptions::tiled(),
                    AllGatherOutputVariance::Varying
                ),
                Vec::new(),
                &[input.clone(), extent.clone()],
            ),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            context.bind(
                ParallelSumScatterOperation::new("x".to_string(), 1, 0, CollectiveOptions::tiled()),
                Vec::new(),
                &[input.clone(), extent.clone()],
            ),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            context.bind(
                AllToAllOperation::new("x".to_string(), 1, 0, 0, CollectiveOptions::tiled()),
                Vec::new(),
                &[input.clone(), extent.clone()],
            ),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            context.bind(
                AllToAllOperation::new("x".to_string(), 1, 0, 1, CollectiveOptions::tiled()),
                Vec::new(),
                &[
                    ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],)),
                    ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
                    ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()),
                ],
            ),
            Ok(vec![ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],))]),
        );

        assert_eq!(
            context
                .bind(
                    AllGatherOperation::new(
                        "x".to_string(),
                        1,
                        0,
                        CollectiveOptions::tiled(),
                        AllGatherOutputVariance::Varying
                    ),
                    Vec::new(),
                    &[input.clone(), ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap()),],
                )
                .unwrap_err()
                .to_string(),
            "`all_gather` output axis 0 extent must equal observed result extent 3 but got 4",
        );
        assert_eq!(
            context
                .bind(
                    AllGatherOperation::new(
                        "x".to_string(),
                        2,
                        0,
                        CollectiveOptions::tiled(),
                        AllGatherOutputVariance::Varying
                    ),
                    Vec::new(),
                    &[input.clone(), ArrayIrValue::Dimension(DimensionValue::constant(6).unwrap()),],
                )
                .unwrap_err(),
            ProgramError::UnsupportedOperation {
                message: "cannot interpret `all_gather` over axis `x` of size 2 without an enclosing binder"
                    .to_string(),
            },
        );
        assert_eq!(
            context
                .bind(
                    ParallelSumScatterOperation::new("empty".to_string(), 0, 0, CollectiveOptions::tiled()),
                    Vec::new(),
                    &[input.clone(), extent.clone()],
                )
                .unwrap_err(),
            ProgramError::Type(TypeError::invalid("`parallel_sum_scatter` axis size must be greater than zero")),
        );

        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = AllGatherOperation::new("x".to_string(), 1, 0, CollectiveOptions::tiled(), AllGatherOutputVariance::Varying),
            cases = [
                {
                    inputs = [(@known, input.clone()), (@known, extent.clone())],
                    outputs = [(@known, input.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(type = input.r#type().into_owned(), replay = input.clone())),
                        (@known, extent.clone()),
                    ],
                    outputs = [(@residual, input.clone())],
                    residual_instructions = 1,
                },
            ],
        );

        let variable = DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap());
        let dimension_type = DimensionType::new(variable.clone());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = builder.add_input(array_type.into());
        let result_extent = builder.add_input(dimension_type.clone().into());
        let output = builder
            .add_instruction(
                AllGatherOperation::new(
                    "x".to_string(),
                    1,
                    0,
                    CollectiveOptions::tiled(),
                    AllGatherOutputVariance::Varying,
                ),
                Vec::new(),
                vec![array, result_extent],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let primal = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]));
        let tangent = ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]));
        let result_extent = ArrayIrValue::Dimension(DimensionValue::new(dimension_type.clone(), 3).unwrap());
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.interpret(vec![primal.clone(), result_extent.clone(), tangent.clone(),]),
            Ok(vec![primal, tangent]),
        );
        assert!(
            jvp.instructions()
                .iter()
                .any(|instruction| { matches!(instruction.operation(), ArrayIrOperation::LinearCall(_)) })
        );
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0])), result_extent])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        let cotangent = ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]));
        let mut pullback_inputs = vec![cotangent.clone()];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![cotangent]));
        let zero_extent = ArrayIrValue::Dimension(DimensionValue::new(dimension_type, 0).unwrap());
        let zero_array = || {
            ArrayIrValue::Array(Array::from_f64s(
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(0)])),
                Vec::new(),
            ))
        };
        let mut primal_outputs = linearization.primal().interpret(vec![zero_array(), zero_extent]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let zero_cotangent = zero_array();
        let mut pullback_inputs = vec![zero_cotangent.clone()];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![zero_cotangent]));
        assert!(matches!(
            program.transpose_with_respect_to(&[0]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "direct `all_gather` transposition with runtime-dependent type metadata requires \
                    linearization so that the relevant primal information can be retained as residuals",
        ));
    }

    #[test]
    fn test_array_ir_invariant_all_gather_linearization() {
        let variable = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let dimension_type = DimensionType::new(variable.clone());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = builder.add_input(array_type.into());
        let result_extent = builder.add_input(dimension_type.clone().into());
        let output = builder
            .add_instruction(
                AllGatherOperation::new(
                    "x".to_string(),
                    1,
                    0,
                    CollectiveOptions::tiled(),
                    AllGatherOutputVariance::Invariant,
                ),
                Vec::new(),
                vec![array, result_extent],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        let rendered_tangent = linearization.tangent().to_string();
        assert!(rendered_tangent.contains("dynamic_shape_slice"));
        assert!(rendered_tangent.contains("reshape"));
        let input = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]));
        let extent = ArrayIrValue::Dimension(DimensionValue::new(dimension_type, 3).unwrap());
        let mut primal_outputs = linearization.primal().interpret(vec![input, extent]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let cotangent = ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]));
        let mut pullback_inputs = vec![cotangent];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]))]),
        );

        // A nondegenerate untiled invariant gather selects the current participant's size-one slice and reshapes
        // away the ranked participant axis.
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = builder.add_input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])).into());
        let participant_extent = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));
        let input_extent = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()));
        let output = builder
            .add_instruction(
                AllGatherOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    CollectiveOptions::default(),
                    AllGatherOutputVariance::Invariant,
                ),
                Vec::new(),
                vec![array, participant_extent, input_extent],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        // The mixed boundary delegates its array contribution to the homogeneous all-gather rule, so the invariant
        // guard that rule owns is what rejects direct transposition here.
        assert!(matches!(
            program.transpose_with_respect_to(&[0]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "direct transposition of invariant `all_gather` cannot represent the participant-indexed \
                    slice; linearize so that the current participant can select its gathered chunk",
        ));
        let pullback = program.linearize().unwrap().pullback().unwrap().to_string();
        assert!(pullback.contains("axis_index [axis_name=\"x\"]"));
        assert!(pullback.contains("dimension_from_scalar"));
        assert!(pullback.contains("dimension_mul"));
        assert!(pullback.contains("dynamic_shape_slice"));
    }

    #[test]
    fn test_array_ir_shape_changing_collective_linearization() {
        let variable = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let dimension_type = DimensionType::new(variable.clone());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = builder.add_input(array_type.into());
        let result_extent = builder.add_input(dimension_type.clone().into());
        let output = builder
            .add_instruction(
                ParallelSumScatterOperation::new("x".to_string(), 1, 0, CollectiveOptions::tiled()),
                Vec::new(),
                vec![array, result_extent],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        assert!(linearization.tangent().to_string().contains("linear_call [residual_count=1]"));
        let input = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]));
        let extent = ArrayIrValue::Dimension(DimensionValue::new(dimension_type, 3).unwrap());
        let mut primal_outputs = linearization.primal().interpret(vec![input, extent]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let cotangent = ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]));
        let mut pullback_inputs = vec![cotangent.clone()];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![cotangent]));

        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = builder.add_input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])).into());
        let extent = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()));
        let output = builder
            .add_instruction(
                AllToAllOperation::new("x".to_string(), 1, 0, 0, CollectiveOptions::tiled()),
                Vec::new(),
                vec![array, extent],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();
        assert!(linearization.tangent().to_string().contains("linear_call"));
        let input = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]));
        let mut primal_outputs = linearization.primal().interpret(vec![input]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let cotangent = ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]));
        let mut pullback_inputs = vec![cotangent.clone()];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![cotangent]));
    }

    #[test]
    fn test_array_ir_explicit_collective_tracing_import_and_rendering() {
        type TestContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let bounds = DimensionBounds::new(1, Some(5)).unwrap();
        let input_variable = DimensionVariable::new("items", bounds);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(input_variable.clone())]));
        let (_, program) = TestContext::trace_with_named_axes(
            |input| input.all_gather_tiled("devices", 0),
            ArrayIrType::Array(input_type),
            vec![("devices".to_string(), NamedAxis::Mesh { axis: 0, size: 2 })],
        )
        .unwrap();

        let [dimension_size, multiplied_extent, all_gather] = program.instructions() else {
            panic!("expected dimension observation, multiplication, and all-gather");
        };
        assert!(matches!(dimension_size.operation(), ArrayIrOperation::DimensionSize(_)));
        assert!(matches!(multiplied_extent.operation(), ArrayIrOperation::Dimension(DimensionOperation::Mul(_)),));
        assert!(matches!(all_gather.operation(), ArrayIrOperation::AllGather(_)));
        assert_eq!(multiplied_extent.inputs()[0], dimension_size.outputs()[0]);
        assert_eq!(all_gather.inputs(), &[program.input_ids()[0], multiplied_extent.outputs()[0]]);
        let rendered = program.to_string();
        assert!(rendered.contains("dimension_size"));
        assert!(rendered.contains("dimension_mul"));
        assert!(rendered.contains("all_gather ["));
        assert!(rendered.contains("axis_name=\"devices\""));
        assert!(rendered.contains("options=Tiled"));

        let target_variable = DimensionVariable::new("target", bounds);
        let target_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(target_variable)]));
        let instantiated = program
            .with_instantiated_type_identities(&[ArrayIrType::Array(target_type.clone())])
            .unwrap()
            .into_owned();
        let mut destination = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let imported_input = destination.add_input(target_type.into());
        let imported_outputs = destination.splice_program(&instantiated, &[imported_input]).unwrap();
        let [imported_dimension_size, imported_multiplied_extent, imported_all_gather] = destination.instructions()
        else {
            panic!("expected the imported explicit collective graph");
        };
        assert_eq!(imported_dimension_size.inputs(), &[imported_input]);
        assert_eq!(imported_all_gather.inputs(), &[imported_input, imported_multiplied_extent.outputs()[0]]);
        assert_eq!(imported_all_gather.outputs(), imported_outputs.as_slice());
    }

    #[test]
    fn test_array_ir_untiled_collective_retains_dynamic_extent_requirement() {
        type TestContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let input_variable = DimensionVariable::new("items", DimensionBounds::new(1, Some(5)).unwrap());
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(input_variable)]));
        let (_, program) = TestContext::trace_with_named_axes(
            |input| input.parallel_sum_scatter("devices", 0),
            ArrayIrType::Array(input_type),
            vec![("devices".to_string(), NamedAxis::Mesh { axis: 0, size: 2 })],
        )
        .unwrap();

        let [dimension_size, requirement, parallel_sum_scatter] = program.instructions() else {
            panic!("expected dimension observation, equality requirement, and sum-scatter");
        };
        assert!(matches!(dimension_size.operation(), ArrayIrOperation::DimensionSize(_)));
        assert!(matches!(requirement.operation(), ArrayIrOperation::Dimension(DimensionOperation::Requirement(_)),));
        assert!(matches!(parallel_sum_scatter.operation(), ArrayIrOperation::ParallelSumScatter(_)));
        assert_eq!(requirement.inputs()[0], dimension_size.outputs()[0]);
        assert_eq!(parallel_sum_scatter.inputs(), &[program.input_ids()[0]]);
        assert_eq!(program.output_types(), &[ArrayIrType::Array(ArrayType::scalar(DataType::F32))],);
    }
}
