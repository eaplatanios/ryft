use std::fmt::Display;

use ryft_macros::Parameterized;

use crate::arrays::batching::{
    DynamicArrayBatchingPolicy, broadcast_array, dimension_constant, folded_array_dimension,
};
use crate::arrays::{
    ArrayBatch, ArrayBatching, ArrayBatchingPolicy, ArrayIrType, ArrayType, DataType, Dimension, DimensionType,
    DimensionValue, Shape, Sharding, StaticArrayBatchingPolicy,
};
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain, ProjectedContext};
use crate::differentiation::DifferentiableType;
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::compare::{Compare, ComparisonDirection};
use crate::operations::constants::constant::ConstantOperation;
use crate::operations::constants::fill::Fill;
use crate::operations::constants::iota::Iota;
use crate::operations::control_flow::select::Select;
use crate::operations::dimensions::dimension_mul::DimensionMulOperation;
use crate::operations::dimensions::dimension_size::DimensionSizeOperation;
use crate::operations::logical::and::And;
use crate::operations::manipulation::broadcasting::{Broadcast, DynamicBroadcastOperation};
use crate::operations::manipulation::conversion::ConvertElementType;
use crate::operations::manipulation::reshaping::{DynamicReshapeOperation, Reshape};
use crate::operations::manipulation::transposition::Transpose;
use crate::operations::math::add::Add;
use crate::operations::math::div::Div;
use crate::operations::math::dot::{Dot, DotDimensionNumbers};
use crate::operations::math::exp::Exp;
use crate::operations::math::log::Log;
use crate::operations::math::mul::Mul;
use crate::operations::math::reduce::{Reduce, ReduceOperation, ReductionKind};
use crate::operations::math::sub::Sub;
use crate::parameters::Parameter;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    Operation, OperationFormatter, OperationProjection, ProgramError, RegionInterface, TypeError, Typed, Value,
    ValueProjection,
};

mod batching;
mod capabilities;
mod composition;
mod configuration;
mod differentiation;
mod inference;
mod operations;

use inference::*;

pub use capabilities::DotProductAttention;
pub(crate) use capabilities::DotProductAttentionBackward;
pub use composition::{dot_product_attention_backward_ir_composition, dot_product_attention_ir_composition};
pub use configuration::{AttentionConfiguration, AttentionImplementation, AttentionInputs, AttentionOperandSignature};
pub use differentiation::differentiable_dot_product_attention;
pub use operations::{
    DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME, DOT_PRODUCT_ATTENTION_OPERATION_NAME,
    DotProductAttentionBackwardOperation, DotProductAttentionOperation,
};

#[cfg(test)]
mod tests;
