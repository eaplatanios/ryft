use std::fmt::Display;

use ryft_macros::Parameterized;

use crate::arrays::{ArrayType, DataType, Dimension, Shape};
use crate::batching::BatchAxis;
use crate::contexts::{Context, Domain};
use crate::differentiation::DifferentiableType;
use crate::macros::check_count;
use crate::operations::arithmetic::{Add, Div, Mul, Sub};
use crate::operations::collectives::parallel_vary::ManualVariationAlignment;
use crate::operations::comparisons::{Compare, ComparisonDirection};
use crate::operations::constants::constant::DimensionConstant;
use crate::operations::constants::fill::Fill;
use crate::operations::constants::iota::Iota;
use crate::operations::control_flow::select::Select;
use crate::operations::dimensions::dimension_mul::DimensionMulOperation;
use crate::operations::dimensions::dimension_size::DimensionSize;
use crate::operations::dot::{Dot, DotDimensionNumbers};
use crate::operations::exponential::{Exp, Log};
use crate::operations::logical::And;
use crate::operations::manipulation::broadcasting::Broadcast;
use crate::operations::manipulation::conversions::ConvertElementType;
use crate::operations::manipulation::reshaping::Reshape;
use crate::operations::manipulation::transposition::Transpose;
use crate::operations::random::{Dropout, RandomAlgorithm};
use crate::operations::reductions::{Reduce, ReductionKind};
use crate::parameters::Parameter;
use crate::programs::{Operation, ProgramError, TypeError, Typed, Value, ValueProjection};

mod composition;
mod configuration;
mod definitions;
mod differentiation;
mod inference;

use inference::*;

pub use composition::{
    AttentionIota, AttentionUniverse, dot_product_attention_backward_ir_composition,
    dot_product_attention_ir_composition,
};
pub use configuration::{AttentionConfiguration, AttentionImplementation, AttentionInputSignature, AttentionInputs};
pub use definitions::{
    AttentionBackwardInputs, AttentionDefinitionKind, AttentionKey, AttentionRules,
    DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME, DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
    DOT_PRODUCT_ATTENTION_COMPOSITE_VERSION, DOT_PRODUCT_ATTENTION_TRAINING_COMPOSITE_NAME, DotProductAttention,
    dot_product_attention,
};
pub use differentiation::{
    AttentionResiduals, AttentionVjp, DifferentiableDotProductAttention, differentiable_dot_product_attention,
};

#[cfg(test)]
mod tests;
