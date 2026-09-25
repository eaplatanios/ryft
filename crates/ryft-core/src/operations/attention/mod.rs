use std::fmt::Display;

use ryft_macros::Parameterized;

use crate::arrays::batching::DynamicArrayExtentBatchingPolicy;
use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrType, ArrayIrValue, ArrayType, DataType,
    Dimension, DimensionType, DimensionValue, Shape, StaticArrayExtentBatchingPolicy,
};
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain, ProjectedContext};
use crate::differentiation::DifferentiableType;
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::arithmetic::{Add, Div, Mul, Sub};
use crate::operations::collectives::parallel_vary::ManualVariationAlignment;
use crate::operations::comparisons::{Compare, ComparisonDirection};
use crate::operations::constants::constant::{ConstantOperation, DimensionConstant};
use crate::operations::constants::fill::Fill;
use crate::operations::constants::iota::Iota;
use crate::operations::control_flow::select::Select;
use crate::operations::dimensions::dimension_mul::DimensionMulOperation;
use crate::operations::dimensions::dimension_size::{DimensionSize, DimensionSizeOperation};
use crate::operations::dot::{Dot, DotDimensionNumbers};
use crate::operations::exponential::{Exp, Log};
use crate::operations::logical::And;
use crate::operations::manipulation::broadcasting::{Broadcast, DynamicBroadcastOperation};
use crate::operations::manipulation::conversions::ConvertElementType;
use crate::operations::manipulation::reshaping::{DynamicReshapeOperation, Reshape};
use crate::operations::manipulation::transposition::Transpose;
use crate::operations::reductions::{Reduce, ReduceOperation, ReductionKind};
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
pub use differentiation::{DifferentiableDotProductAttention, differentiable_dot_product_attention};
pub use operations::{
    DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME, DOT_PRODUCT_ATTENTION_OPERATION_NAME,
    DotProductAttentionBackwardOperation, DotProductAttentionOperation,
};

#[cfg(test)]
mod tests;
