//! Reference [`Array`] implementations of quantization capabilities.

use crate::arrays::arrays::Array;
use crate::arrays::types::data::DataType;
use crate::operations::dot::DotDimensionNumbers;
use crate::operations::quantization::{ScaledDot, ScaledDotOperation, scaled_dot_composition};
use crate::programs::{ProgramError, Typed};

impl ScaledDot for Array {
    fn scaled_dot(
        &self,
        rhs: &Self,
        lhs_scale: Option<&Self>,
        rhs_scale: Option<&Self>,
        dimensions: Option<&DotDimensionNumbers>,
        preferred_element_type: Option<DataType>,
    ) -> Result<Self, ProgramError> {
        let dimensions = dimensions
            .cloned()
            .map(Ok)
            .unwrap_or_else(|| ScaledDotOperation::default_dimensions(self.r#type().rank()))?;
        scaled_dot_composition(
            self,
            rhs,
            lhs_scale,
            rhs_scale,
            &dimensions,
            preferred_element_type.unwrap_or(DataType::BF16),
        )
    }
}
