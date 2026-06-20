use crate::operations::arithmetic::ScaleOperation;
use crate::operations::constants::ConstantOperation;
use crate::operations::scalars::LinearScalarOperation;
use crate::programs::{ProgramError, Value};
use crate::tracing_v2::differentiation::FactorParameterizedOperation;
use crate::tracing_v2::operations::select::LinearSelectOperation;
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
            Self::Zero(operation) => Ok(operation.clone().into()),
            Self::ZeroLike(operation) => Ok(operation.clone().into()),
            Self::One(operation) => Ok(operation.clone().into()),
            Self::OneLike(operation) => Ok(operation.clone().into()),
            Self::Constant(constant) => Ok(ConstantOperation::new(map_factor(constant.value())?).into()),
            Self::Neg(operation) => Ok(operation.clone().into()),
            Self::Add(operation) => Ok(operation.clone().into()),
            Self::Sub(operation) => Ok(operation.clone().into()),
            Self::Scale(operation) => Ok(ScaleOperation::new(map_factor(operation.factor())?).into()),
            Self::Select(operation) => Ok(LinearSelectOperation::new(map_factor(operation.condition())?).into()),
            Self::CustomVjpCall(call) => {
                Ok(LinearScalarOperation::CustomVjpCall(Box::new(call.map_factors(map_factor)?)))
            }
        }
    }
}
