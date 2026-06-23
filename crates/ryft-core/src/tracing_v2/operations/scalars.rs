use crate::operations::arithmetic::ScaleOperation;
use crate::operations::scalars::LinearScalarOperation;
use crate::payloads::Input;
use crate::programs::{ProgramError, Value};
use crate::tracing_v2::differentiation::FactorParameterizedOperation;
use crate::tracing_v2::operations::select::LinearSelectOperation;
use crate::types::DataType;

impl<V: Value<DataType>, C: Value<DataType>, F: Value<DataType>> FactorParameterizedOperation<DataType, F>
    for LinearScalarOperation<V, C, F>
{
    type WithFactor<MappedFactor: Value<DataType>> = LinearScalarOperation<V, C, MappedFactor>;

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
            Self::Constant(operation) => Ok(operation.clone().into()),
            Self::Neg(operation) => Ok(operation.clone().into()),
            Self::Add(operation) => Ok(operation.clone().into()),
            Self::Sub(operation) => Ok(operation.clone().into()),
            Self::Scale(operation) => {
                Ok(ScaleOperation::<DataType, MappedFactor, Input>::new(map_factor(operation.factor())?).into())
            }
            Self::Select(operation) => Ok(LinearSelectOperation::new(map_factor(operation.condition())?).into()),
            Self::CustomVjpCall(call) => {
                Ok(LinearScalarOperation::CustomVjpCall(Box::new(call.map_factors(map_factor)?)))
            }
        }
    }
}
