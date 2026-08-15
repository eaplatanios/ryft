//! Reference [`Array`] kernels for the random-number operation family contracts.
//!
//! Bit generation decodes the counter-based generator state, produces the requested number of words through the
//! shared algorithm implementations, and re-encodes both the advanced state and the generated bits into arrays whose
//! declared physical layouts are preserved.

use crate::arrays::arrays::Array;
use crate::arrays::ir::ArrayIrValue;
use crate::arrays::operations::ArrayIrOperation;
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::data::DataType;
use crate::arrays::types::dimensions::{Dimension, DimensionType, Shape};
use crate::arrays::types::ir::ArrayIrType;
use crate::contexts::EagerContext;
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::operations::dimensions::dimension_size::DimensionSize;
use crate::operations::random::{
    RNG_BIT_GENERATOR_OPERATION_NAME, RandomAlgorithm, RngBitGenerator, RngBitGeneratorOperation, philox_u32_words,
    philox_u64_words, threefry_u32_words, threefry_u64_words,
};
use crate::programs::{Operation, ProgramError, TypeError, Typed, Value, ValueProjection};

// TODO(eaplatanios): Review this.

impl<A: DimensionSize<usize> + RngBitGenerator + Value<Type = ArrayType>>
    InterpretableOperation<EagerContext<ArrayIrValue<A>, ArrayIrOperation<A>>>
    for RngBitGeneratorOperation<ArrayIrType>
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
        let state = <ArrayIrValue<A> as ValueProjection<ArrayType>>::projected(&inputs[0])?;
        let mut output_extents = inputs[1..].iter();
        let concrete_output_dimensions = self
            .output_type()
            .shape()
            .dimensions()
            .iter()
            .map(|dimension| match dimension {
                Dimension::Static(extent) => Ok(Dimension::Static(*extent)),
                Dimension::Dynamic(_) => {
                    let extent = output_extents.next().unwrap();
                    Ok(Dimension::Static(
                        <ArrayIrValue<A> as ValueProjection<DimensionType>>::projected(extent)?.extent(),
                    ))
                }
            })
            .collect::<Result<Vec<_>, TypeError>>()?;
        let concrete_output_type = self.output_type().clone().with_shape(Shape::new(concrete_output_dimensions));
        let (advanced_state, bits) = state.rng_bit_generator(self.algorithm(), &concrete_output_type)?;
        for (axis, dimension) in self.output_type().shape().dimensions().iter().enumerate() {
            if matches!(dimension, Dimension::Dynamic(_)) {
                let expected_extent =
                    concrete_output_type.shape().dimensions()[axis].value().expect("the concrete output is static");
                let actual_extent = bits.dimension_size(axis)?;
                if actual_extent != expected_extent {
                    return Err(ProgramError::InvalidArgument {
                        message: format!(
                            "`{RNG_BIT_GENERATOR_OPERATION_NAME}` bits output axis {axis} has extent {actual_extent}, \
                             but its explicit extent operand is {expected_extent}",
                        ),
                    });
                }
            }
        }
        Ok(vec![ArrayIrValue::Array(advanced_state), ArrayIrValue::Array(bits)])
    }
}

impl RngBitGenerator for Array {
    fn rng_bit_generator(
        &self,
        algorithm: RandomAlgorithm,
        output_type: &ArrayType,
    ) -> Result<(Self, Self), ProgramError> {
        let Some(output_shape) = output_type.static_shape() else {
            return Err(TypeError::invalid(format!(
                "`{RNG_BIT_GENERATOR_OPERATION_NAME}` does not support dynamically shaped outputs"
            ))
            .into());
        };
        let count = output_shape.dimensions().iter().product::<usize>();
        let data_type = output_type.data_type();
        if !matches!(data_type, DataType::U8 | DataType::U16 | DataType::U32 | DataType::U64) {
            return Err(TypeError::invalid(format!(
                "`{RNG_BIT_GENERATOR_OPERATION_NAME}` does not support output data type {data_type}",
            ))
            .into());
        }
        let expected_state_type = algorithm.state_type();
        if self.r#type().data_type() != expected_state_type.data_type()
            || self.r#type().shape() != expected_state_type.shape()
        {
            return Err(TypeError::invalid(format!(
                "`{}` with the {} algorithm needs a {} state but got {}",
                RNG_BIT_GENERATOR_OPERATION_NAME,
                algorithm,
                expected_state_type,
                self.r#type().as_ref(),
            ))
            .into());
        }
        // Narrower-than-32-bit outputs retain the low bits of each generated `u32` word.
        let bits_from_u32_words = |words: Vec<u32>| match data_type {
            DataType::U32 => Array::from_elements(output_type.clone(), &words),
            DataType::U16 => Array::from_fn_elements(output_type.clone(), |index| Ok(words[index] as u16)),
            DataType::U8 => Array::from_fn_elements(output_type.clone(), |index| Ok(words[index] as u8)),
            _ => unreachable!(),
        };
        match algorithm {
            RandomAlgorithm::ThreeFry => {
                // The state-type check above guarantees exactly two decoded `u64` elements.
                let [key, counter]: [u64; 2] = self.elements::<u64>()?.try_into().unwrap();
                let (new_counter, bits) = if data_type == DataType::U64 {
                    let (words, new_counter) = threefry_u64_words(key, counter, count);
                    (new_counter, Array::from_elements(output_type.clone(), &words)?)
                } else {
                    let (words, new_counter) = threefry_u32_words(key, counter, count);
                    (new_counter, bits_from_u32_words(words)?)
                };
                Ok((Array::from_elements(self.r#type().into_owned(), &[key, new_counter])?, bits))
            }
            RandomAlgorithm::Philox => {
                // The state-type check above guarantees exactly three decoded `u64` elements.
                let [key, counter_low, counter_high]: [u64; 3] = self.elements::<u64>()?.try_into().unwrap();
                let counter = u128::from(counter_low) | (u128::from(counter_high) << 64);
                let (new_counter, bits) = if data_type == DataType::U64 {
                    let (words, new_counter) = philox_u64_words(key, counter, count);
                    (new_counter, Array::from_elements(output_type.clone(), &words)?)
                } else {
                    let (words, new_counter) = philox_u32_words(key, counter, count);
                    (new_counter, bits_from_u32_words(words)?)
                };
                let advanced_state = [key, new_counter as u64, (new_counter >> 64) as u64];
                Ok((Array::from_elements(self.r#type().into_owned(), &advanced_state)?, bits))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::arrays::array_type;
    use crate::arrays::types::layouts::{Layout, StridedLayout};
    use crate::programs::Typed;

    use super::*;

    #[test]
    fn test_array_random_bit_generation() {
        // State and result layouts remain physical storage contracts while the generator consumes and produces values
        // in logical order.
        let state_type =
            RandomAlgorithm::ThreeFry.state_type().with_layout(Layout::Strided(StridedLayout::new(vec![-8])));
        let state = Array::from_elements(state_type.clone(), &[42u64, 7]).unwrap();
        let output_type = array_type(DataType::U16, &[5]).with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        let (advanced_state, bits) = state.rng_bit_generator(RandomAlgorithm::ThreeFry, &output_type).unwrap();
        let (expected_words, expected_counter) = threefry_u32_words(42, 7, 5);
        let expected_words = expected_words.into_iter().map(|word| word as u16).collect::<Vec<_>>();
        assert_eq!(advanced_state.r#type().as_ref(), &state_type);
        assert_eq!(advanced_state.elements::<u64>(), Ok(vec![42, expected_counter]));
        assert_eq!(bits.r#type().as_ref(), &output_type);
        assert_eq!(bits.elements::<u16>(), Ok(expected_words.clone()));
        let mut expected_storage = vec![0; 18];
        for (index, word) in expected_words.into_iter().enumerate() {
            expected_storage[index * 4..index * 4 + 2].copy_from_slice(&word.to_le_bytes());
        }
        assert_eq!(bits.storage_bytes(), expected_storage);

        // Eight-bit outputs retain the low byte of each generated `u32` word.
        let output_type = array_type(DataType::U8, &[5]);
        let (_, bits) = state.rng_bit_generator(RandomAlgorithm::ThreeFry, &output_type).unwrap();
        let (expected_words, _) = threefry_u32_words(42, 7, 5);
        assert_eq!(bits.elements::<u8>(), Ok(expected_words.into_iter().map(|word| word as u8).collect()));

        // Direct backend calls enforce the same state contract as operation type inference.
        let invalid_state = Array::vector(vec![42u64, 7, 9]);
        assert!(matches!(
            invalid_state.rng_bit_generator(RandomAlgorithm::ThreeFry, &output_type),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`rng_bit_generator` with the three_fry algorithm needs a u64[2] state but got u64[3]",
        ));
    }
}
