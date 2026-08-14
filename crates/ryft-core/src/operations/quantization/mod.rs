//! Quantization operations and value-level capabilities.

use std::fmt::{Debug, Display};

use crate::arrays::{
    ArrayBatch, ArrayBatching, ArrayBatchingPolicy, ArrayIrType, ArrayIrValue, ArrayType, DataType, DimensionType,
    DimensionValue,
};
use crate::axes::Axis;
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    DifferentiableOperation, DifferentiationDriver, DifferentiationDual, DifferentiationError,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::dimensions::dimension_requirement::DimensionRequirement;
use crate::operations::dimensions::dimension_size::DimensionSize;
use crate::operations::manipulation::broadcasting::{Broadcast, DynamicBroadcast};
use crate::operations::manipulation::conversion::ConvertElementType;
use crate::operations::manipulation::reshaping::{DynamicReshape, Reshape};
use crate::operations::math::div::Div;
use crate::operations::math::dot::{Dot, DotDimensionNumbers, dot_abstract, lift_dot_dimensions};
use crate::operations::math::mul::Mul;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    Operation, OperationFormatter, ProgramError, RegionInterface, TypeError, Typed, Value, ValueProjection,
};

mod block;
mod scaled_dot;

pub use block::BlockQuantize;
pub use scaled_dot::{
    SCALED_DOT_OPERATION_NAME, ScaledDot, ScaledDotOperation, scaled_dot_composition, scaled_dot_ir_composition,
};
