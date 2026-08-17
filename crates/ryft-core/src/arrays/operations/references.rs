use crate::arrays::ir::ArrayIrValue;
use crate::arrays::types::arrays::ArrayType;
use crate::operations::{NewReference, ReferenceRead};
use crate::programs::{ProgramError, Reference, ReferenceType, Value, ValueProjection};

// TODO(eaplatanios): Review this module.

impl<A: Value<Type = ArrayType>> NewReference for ArrayIrValue<A> {
    fn new_reference(&self) -> Result<Self, ProgramError> {
        let value = <Self as ValueProjection<ArrayType>>::projected(self)?.clone();
        Ok(Self::Reference(Reference::new(value)))
    }
}

impl<A: Value<Type = ArrayType>> ReferenceRead for ArrayIrValue<A> {
    fn read(&self) -> Result<Self, ProgramError> {
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        Ok(Self::Array(reference.read().map_err(ProgramError::custom)?))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::arrays::Array;
    use crate::programs::TypeError;

    use super::*;

    #[test]
    fn test_eager_reference_allocation_and_read_roundtrip() {
        let initial = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0]));
        let reference = initial.new_reference().unwrap();
        assert!(matches!(reference, ArrayIrValue::Reference(_)));
        assert_eq!(ReferenceRead::read(&reference).unwrap(), initial);
    }

    #[test]
    fn test_eager_reference_operations_reject_mismatched_member_kinds() {
        let array = ArrayIrValue::<Array>::Array(Array::scalar(1.0_f32));
        assert_eq!(
            ReferenceRead::read(&array),
            Err(TypeError::invalid("expected reference type but got array type").into()),
        );
        let reference = array.new_reference().unwrap();
        assert_eq!(
            reference.new_reference(),
            Err(TypeError::invalid("expected array type but got reference type").into()),
        );
    }
}
