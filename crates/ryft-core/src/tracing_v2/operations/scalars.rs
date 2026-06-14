use crate::operations::constants::ConstantOperation;
use crate::operations::scalars::LinearScalarOperation;
use crate::programs::{ProgramError, Value};
use crate::tracing_v2::differentiation::FactorParameterizedOperation;
use crate::types::DataType;

impl<C: Value<DataType>, F: Value<DataType>> FactorParameterizedOperation<DataType, F> for LinearScalarOperation<C, F> {
    type WithFactor<MappedFactor: Value<DataType>> = LinearScalarOperation<C, MappedFactor>;

    fn try_map_factors<MappedFactor: Value<DataType>, MapFactorFn>(
        &self,
        map_factor: &mut MapFactorFn,
    ) -> Result<Self::WithFactor<MappedFactor>, ProgramError>
    where
        MapFactorFn: FnMut(&F) -> Result<MappedFactor, ProgramError>,
    {
        match self {
            Self::Zero(zero) => Ok(LinearScalarOperation::Zero(zero.clone())),
            Self::ZeroLike => Ok(LinearScalarOperation::ZeroLike),
            Self::One(one) => Ok(LinearScalarOperation::One(one.clone())),
            Self::OneLike => Ok(LinearScalarOperation::OneLike),
            Self::Constant(constant) => {
                Ok(LinearScalarOperation::Constant(ConstantOperation::new(map_factor(constant.value())?)))
            }
            Self::Neg => Ok(LinearScalarOperation::Neg),
            Self::Add => Ok(LinearScalarOperation::Add),
            Self::Sub => Ok(LinearScalarOperation::Sub),
            Self::Scale { factor } => Ok(LinearScalarOperation::Scale { factor: map_factor(factor)? }),
            Self::Select { condition } => Ok(LinearScalarOperation::Select { condition: map_factor(condition)? }),
            Self::CustomVjpCall(call) => {
                Ok(LinearScalarOperation::CustomVjpCall(Box::new(call.map_factors(map_factor)?)))
            }
        }
    }
}
