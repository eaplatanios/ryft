//! Reference [`Array`] kernels for the complex-number operation family contracts.
//!
//! Complex construction pairs two identically typed real arrays into the complex element data type with the parts'
//! precision, and the part extractors invert that mapping. Conjugation preserves the complex element data type.

use num_complex::Complex;

use crate::arrays::arrays::Array;
use crate::arrays::types::data::DataType;
use crate::operations::complex::{Conjugate, Imaginary, Real};
use crate::programs::{ProgramError, TypeError, Typed};

// TODO(eaplatanios): Review this.

impl crate::operations::complex::Complex for Array {
    fn complex(&self, imaginary: &Self) -> Result<Self, ProgramError> {
        // Mirrors the `ComplexOperation` type-inference contract: the two part arrays must have identical types, and
        // the element data type maps to the complex data type with the parts' precision.
        if self.r#type() != imaginary.r#type() {
            return Err(TypeError::invalid(format!(
                "'complex' requires identical part types but got {} and {}",
                self.r#type(),
                imaginary.r#type(),
            ))
            .into());
        }
        let data_type = match self.r#type().data_type() {
            DataType::F32 => DataType::C64,
            DataType::F64 => DataType::C128,
            other => {
                return Err(TypeError::invalid(format!(
                    "cannot construct a complex value from parts of data type {other}",
                ))
                .into());
            }
        };
        let output_type = self.r#type().into_owned().with_data_type(data_type);
        if data_type == DataType::C64 {
            self.binary_elements::<f32, Complex<f32>>(imaginary, output_type, |real, imaginary| {
                Ok(Complex::new(real, imaginary))
            })
        } else {
            self.binary_elements::<f64, Complex<f64>>(imaginary, output_type, |real, imaginary| {
                Ok(Complex::new(real, imaginary))
            })
        }
    }
}

impl Conjugate for Array {
    fn conjugate(&self) -> Result<Self, ProgramError> {
        match self.r#type().data_type() {
            DataType::C64 => {
                self.map_elements::<Complex<f32>, Complex<f32>>(self.r#type().into_owned(), |value| Ok(value.conj()))
            }
            DataType::C128 => {
                self.map_elements::<Complex<f64>, Complex<f64>>(self.r#type().into_owned(), |value| Ok(value.conj()))
            }
            other => Err(TypeError::invalid(format!("cannot conjugate a scalar of data type {other}")).into()),
        }
    }
}

impl Real for Array {
    fn real(&self) -> Result<Self, ProgramError> {
        // The real part of a complex array has the parts' real data type, mirroring the `RealOperation`
        // type-inference contract, which requires a complex operand.
        match self.r#type().data_type() {
            DataType::C64 => self
                .map_elements::<Complex<f32>, f32>(self.r#type().into_owned().with_data_type(DataType::F32), |value| {
                    Ok(value.re)
                }),
            DataType::C128 => self
                .map_elements::<Complex<f64>, f64>(self.r#type().into_owned().with_data_type(DataType::F64), |value| {
                    Ok(value.re)
                }),
            other => {
                Err(TypeError::invalid(format!("cannot extract the real part of a scalar of data type {other}")).into())
            }
        }
    }
}

impl Imaginary for Array {
    fn imaginary(&self) -> Result<Self, ProgramError> {
        // The imaginary part of a complex array has the parts' real data type, mirroring the `ImaginaryOperation`
        // type-inference contract, which requires a complex operand.
        match self.r#type().data_type() {
            DataType::C64 => self
                .map_elements::<Complex<f32>, f32>(self.r#type().into_owned().with_data_type(DataType::F32), |value| {
                    Ok(value.im)
                }),
            DataType::C128 => self
                .map_elements::<Complex<f64>, f64>(self.r#type().into_owned().with_data_type(DataType::F64), |value| {
                    Ok(value.im)
                }),
            other => {
                Err(TypeError::invalid(format!("cannot extract the imaginary part of a scalar of data type {other}",))
                    .into())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::arrays::array_type;
    use crate::operations::Neg;
    use crate::operations::complex::Complex;
    use crate::programs::Typed;

    use super::*;

    #[test]
    fn test_array_complex_parts() {
        let real = Array::vector(vec![1.0, 2.0]);
        let imaginary = Array::vector(vec![3.0, -4.0]);
        let complex = real.complex(&imaginary).unwrap();
        assert_eq!(complex.r#type().into_owned(), array_type(DataType::C128, &[2]));
        assert_eq!(
            complex.elements::<ComplexNumber<f64>>(),
            Ok(vec![ComplexNumber::new(1.0, 3.0), ComplexNumber::new(2.0, -4.0)]),
        );
        assert_eq!(complex.real().unwrap(), real);
        assert_eq!(complex.imaginary().unwrap(), imaginary);
        let conjugate = complex.conjugate().unwrap();
        assert_eq!(conjugate.imaginary().unwrap(), imaginary.neg().unwrap());
        // Complex construction requires identical part types.
        assert!(real.complex(&Array::vector(vec![1.0f32])).is_err());
        assert!(Array::vector(vec![1i32]).complex(&Array::vector(vec![2i32])).is_err());
    }
}
