//! Primitive operation traits and scalar primitive implementations for `tracing_v2`.
//!
//! The op set is intentionally open: each primitive is represented by its own type implementing one or more
//! transform-specific traits. This keeps graph representations extensible without requiring central enums.

use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

use crate::tracing_v2::{
    TraceError, TraceValue,
    batch::Batch,
    forward::{JvpTracer, TangentSpace},
    graph::{AtomId, GraphBuilder},
};

/// Core primitive operation interface understood by staged graphs.
pub trait Op<V>: Debug + Display
where
    V: TraceValue,
{
    /// Returns the stable primitive name used in diagnostics and pretty-printing.
    fn name(&self) -> &'static str;

    /// Computes abstract outputs from abstract inputs without executing the operation.
    fn abstract_eval(&self, inputs: &[V::Abstract]) -> Result<Vec<V::Abstract>, TraceError>;

    /// Executes the operation on concrete values.
    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError>;
}

/// Marker trait for operations that may appear in JIT-staged graphs.
pub trait StagedOp<V>: Op<V>
where
    V: TraceValue,
{
}

impl<T, V> StagedOp<V> for T
where
    T: Op<V>,
    V: TraceValue,
{
}

/// Shared reference to a dynamically dispatched staged operation.
pub type StagedOpRef<V> = Arc<dyn StagedOp<V>>;

/// Primitive operation with a forward-mode differentiation rule.
pub trait JvpOp<V>: Op<V>
where
    V: TraceValue,
{
    /// Applies the primitive's forward-mode rule to traced inputs.
    fn jvp<T>(&self, inputs: &[JvpTracer<V, T>]) -> Result<Vec<JvpTracer<V, T>>, TraceError>
    where
        T: TangentSpace<V>;
}

/// Primitive operation with a batching rule used by `vmap`.
pub trait BatchOp<V>: Op<V>
where
    V: TraceValue,
{
    /// Applies the primitive's batching rule to batched inputs.
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError>;
}

/// Primitive operation that may appear in a linearized program and therefore supports transposition.
pub trait LinearOp<V>: Op<V>
where
    V: TraceValue,
{
    /// Applies the primitive's transpose rule to output cotangents.
    fn transpose(
        &self,
        builder: &mut GraphBuilder<LinearOpRef<V>, V>,
        inputs: &[AtomId],
        outputs: &[AtomId],
        output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError>;
}
/// Shared reference to a dynamically dispatched linear operation.
pub type LinearOpRef<V> = Arc<dyn LinearOp<V>>;

impl<T, V> Op<V> for Arc<T>
where
    T: Op<V> + ?Sized,
    V: TraceValue,
{
    #[inline]
    fn name(&self) -> &'static str {
        (**self).name()
    }

    #[inline]
    fn abstract_eval(&self, inputs: &[V::Abstract]) -> Result<Vec<V::Abstract>, TraceError> {
        (**self).abstract_eval(inputs)
    }

    #[inline]
    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        (**self).eval(inputs)
    }
}

impl<T, V> JvpOp<V> for Arc<T>
where
    T: JvpOp<V> + ?Sized,
    V: TraceValue,
{
    #[inline]
    fn jvp<U>(&self, inputs: &[JvpTracer<V, U>]) -> Result<Vec<JvpTracer<V, U>>, TraceError>
    where
        U: TangentSpace<V>,
    {
        (**self).jvp(inputs)
    }
}

impl<T, V> BatchOp<V> for Arc<T>
where
    T: BatchOp<V> + ?Sized,
    V: TraceValue,
{
    #[inline]
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
        (**self).batch(inputs)
    }
}

impl<T, V> LinearOp<V> for Arc<T>
where
    T: LinearOp<V> + ?Sized,
    V: TraceValue,
{
    #[inline]
    fn transpose(
        &self,
        builder: &mut GraphBuilder<LinearOpRef<V>, V>,
        inputs: &[AtomId],
        outputs: &[AtomId],
        output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError> {
        (**self).transpose(builder, inputs, outputs, output_cotangents)
    }
}
fn expect_input_count(inputs: usize, expected: usize) -> Result<(), TraceError> {
    if inputs == expected { Ok(()) } else { Err(TraceError::InvalidInputCount { expected, got: inputs }) }
}

fn expect_batch_sizes_match<V>(left: &Batch<V>, right: &Batch<V>) -> Result<(), TraceError> {
    if left.len() == right.len() { Ok(()) } else { Err(TraceError::MismatchedBatchSize) }
}

fn unary_abstract<V>(_op: &'static str, inputs: &[V::Abstract]) -> Result<V::Abstract, TraceError>
where
    V: TraceValue,
{
    expect_input_count(inputs.len(), 1)?;
    Ok(inputs[0].clone())
}

fn binary_same_abstract<V>(op: &'static str, inputs: &[V::Abstract]) -> Result<V::Abstract, TraceError>
where
    V: TraceValue,
{
    expect_input_count(inputs.len(), 2)?;
    if inputs[0] == inputs[1] { Ok(inputs[0].clone()) } else { Err(TraceError::IncompatibleAbstractValues { op }) }
}

/// Elementwise addition primitive.
#[derive(Clone, Default)]
pub struct AddOp;

impl Debug for AddOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Add")
    }
}

impl Display for AddOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "add")
    }
}

impl<V> Op<V> for AddOp
where
    V: TraceValue,
{
    fn name(&self) -> &'static str {
        "add"
    }

    fn abstract_eval(&self, inputs: &[V::Abstract]) -> Result<Vec<V::Abstract>, TraceError> {
        Ok(vec![binary_same_abstract::<V>("add", inputs)?])
    }

    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 2)?;
        Ok(vec![inputs[0].clone() + inputs[1].clone()])
    }
}

impl<V> JvpOp<V> for AddOp
where
    V: TraceValue,
{
    fn jvp<T>(&self, inputs: &[JvpTracer<V, T>]) -> Result<Vec<JvpTracer<V, T>>, TraceError>
    where
        T: TangentSpace<V>,
    {
        expect_input_count(inputs.len(), 2)?;
        Ok(vec![JvpTracer {
            primal: inputs[0].primal.clone() + inputs[1].primal.clone(),
            tangent: T::add(inputs[0].tangent.clone(), inputs[1].tangent.clone()),
        }])
    }
}

impl<V> BatchOp<V> for AddOp
where
    V: TraceValue,
{
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
        expect_input_count(inputs.len(), 2)?;
        expect_batch_sizes_match(&inputs[0], &inputs[1])?;
        Ok(vec![Batch::new(
            inputs[0]
                .lanes()
                .iter()
                .cloned()
                .zip(inputs[1].lanes().iter().cloned())
                .map(|(left, right)| left + right)
                .collect(),
        )])
    }
}

impl<V> LinearOp<V> for AddOp
where
    V: TraceValue,
{
    fn transpose(
        &self,
        _builder: &mut GraphBuilder<LinearOpRef<V>, V>,
        inputs: &[AtomId],
        outputs: &[AtomId],
        output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError> {
        expect_input_count(inputs.len(), 2)?;
        expect_input_count(outputs.len(), 1)?;
        expect_input_count(output_cotangents.len(), 1)?;
        Ok(vec![Some(output_cotangents[0]), Some(output_cotangents[0])])
    }
}

/// Elementwise multiplication primitive.
#[derive(Clone, Default)]
pub struct MulOp;

impl Debug for MulOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Mul")
    }
}

impl Display for MulOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "mul")
    }
}

impl<V> Op<V> for MulOp
where
    V: TraceValue,
{
    fn name(&self) -> &'static str {
        "mul"
    }

    fn abstract_eval(&self, inputs: &[V::Abstract]) -> Result<Vec<V::Abstract>, TraceError> {
        Ok(vec![binary_same_abstract::<V>("mul", inputs)?])
    }

    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 2)?;
        Ok(vec![inputs[0].clone() * inputs[1].clone()])
    }
}

impl<V> JvpOp<V> for MulOp
where
    V: TraceValue,
{
    fn jvp<T>(&self, inputs: &[JvpTracer<V, T>]) -> Result<Vec<JvpTracer<V, T>>, TraceError>
    where
        T: TangentSpace<V>,
    {
        expect_input_count(inputs.len(), 2)?;
        let left = &inputs[0];
        let right = &inputs[1];
        Ok(vec![JvpTracer {
            primal: left.primal.clone() * right.primal.clone(),
            tangent: T::add(
                T::scale(right.primal.clone(), left.tangent.clone()),
                T::scale(left.primal.clone(), right.tangent.clone()),
            ),
        }])
    }
}

impl<V> BatchOp<V> for MulOp
where
    V: TraceValue,
{
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
        expect_input_count(inputs.len(), 2)?;
        expect_batch_sizes_match(&inputs[0], &inputs[1])?;
        Ok(vec![Batch::new(
            inputs[0]
                .lanes()
                .iter()
                .cloned()
                .zip(inputs[1].lanes().iter().cloned())
                .map(|(left, right)| left * right)
                .collect(),
        )])
    }
}

/// Elementwise negation primitive.
#[derive(Clone, Default)]
pub struct NegOp;

impl Debug for NegOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Neg")
    }
}

impl Display for NegOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "neg")
    }
}

impl<V> Op<V> for NegOp
where
    V: TraceValue,
{
    fn name(&self) -> &'static str {
        "neg"
    }

    fn abstract_eval(&self, inputs: &[V::Abstract]) -> Result<Vec<V::Abstract>, TraceError> {
        Ok(vec![unary_abstract::<V>("neg", inputs)?])
    }

    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![-inputs[0].clone()])
    }
}

impl<V> JvpOp<V> for NegOp
where
    V: TraceValue,
{
    fn jvp<T>(&self, inputs: &[JvpTracer<V, T>]) -> Result<Vec<JvpTracer<V, T>>, TraceError>
    where
        T: TangentSpace<V>,
    {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![JvpTracer { primal: -inputs[0].primal.clone(), tangent: T::neg(inputs[0].tangent.clone()) }])
    }
}

impl<V> BatchOp<V> for NegOp
where
    V: TraceValue,
{
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![Batch::new(inputs[0].lanes().iter().cloned().map(|lane| -lane).collect())])
    }
}

impl<V> LinearOp<V> for NegOp
where
    V: TraceValue,
{
    fn transpose(
        &self,
        builder: &mut GraphBuilder<LinearOpRef<V>, V>,
        inputs: &[AtomId],
        outputs: &[AtomId],
        output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        expect_input_count(outputs.len(), 1)?;
        expect_input_count(output_cotangents.len(), 1)?;
        let contribution = builder.add_equation(Arc::new(NegOp), vec![output_cotangents[0]])?[0];
        Ok(vec![Some(contribution)])
    }
}

/// Elementwise sine primitive.
#[derive(Clone, Default)]
pub struct SinOp;

impl Debug for SinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Sin")
    }
}

impl Display for SinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sin")
    }
}

impl<V> Op<V> for SinOp
where
    V: TraceValue,
{
    fn name(&self) -> &'static str {
        "sin"
    }

    fn abstract_eval(&self, inputs: &[V::Abstract]) -> Result<Vec<V::Abstract>, TraceError> {
        Ok(vec![unary_abstract::<V>("sin", inputs)?])
    }

    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone().sin()])
    }
}

impl<V> JvpOp<V> for SinOp
where
    V: TraceValue,
{
    fn jvp<T>(&self, inputs: &[JvpTracer<V, T>]) -> Result<Vec<JvpTracer<V, T>>, TraceError>
    where
        T: TangentSpace<V>,
    {
        expect_input_count(inputs.len(), 1)?;
        let input = &inputs[0];
        Ok(vec![JvpTracer {
            primal: input.primal.clone().sin(),
            tangent: T::scale(input.primal.clone().cos(), input.tangent.clone()),
        }])
    }
}

impl<V> BatchOp<V> for SinOp
where
    V: TraceValue,
{
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![Batch::new(inputs[0].lanes().iter().cloned().map(|lane| lane.sin()).collect())])
    }
}

/// Elementwise cosine primitive.
#[derive(Clone, Default)]
pub struct CosOp;

impl Debug for CosOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cos")
    }
}

impl Display for CosOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cos")
    }
}

impl<V> Op<V> for CosOp
where
    V: TraceValue,
{
    fn name(&self) -> &'static str {
        "cos"
    }

    fn abstract_eval(&self, inputs: &[V::Abstract]) -> Result<Vec<V::Abstract>, TraceError> {
        Ok(vec![unary_abstract::<V>("cos", inputs)?])
    }

    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone().cos()])
    }
}

impl<V> JvpOp<V> for CosOp
where
    V: TraceValue,
{
    fn jvp<T>(&self, inputs: &[JvpTracer<V, T>]) -> Result<Vec<JvpTracer<V, T>>, TraceError>
    where
        T: TangentSpace<V>,
    {
        expect_input_count(inputs.len(), 1)?;
        let input = &inputs[0];
        Ok(vec![JvpTracer {
            primal: input.primal.clone().cos(),
            tangent: T::neg(T::scale(input.primal.clone().sin(), input.tangent.clone())),
        }])
    }
}

impl<V> BatchOp<V> for CosOp
where
    V: TraceValue,
{
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![Batch::new(inputs[0].lanes().iter().cloned().map(|lane| lane.cos()).collect())])
    }
}

/// Unary linear operation that multiplies its input by a captured factor.
#[derive(Clone)]
pub struct ScaleOp<V>
where
    V: TraceValue,
{
    factor: V,
}

impl<V> ScaleOp<V>
where
    V: TraceValue,
{
    /// Creates a new scale operation capturing the provided factor.
    #[inline]
    pub fn new(factor: V) -> Self {
        Self { factor }
    }

    #[inline]
    pub(crate) fn factor(&self) -> &V {
        &self.factor
    }
}

impl<V> Debug for ScaleOp<V>
where
    V: TraceValue,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Scale")
    }
}

impl<V> Display for ScaleOp<V>
where
    V: TraceValue,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "scale")
    }
}

impl<V> Op<V> for ScaleOp<V>
where
    V: TraceValue,
{
    fn name(&self) -> &'static str {
        "scale"
    }

    fn abstract_eval(&self, inputs: &[V::Abstract]) -> Result<Vec<V::Abstract>, TraceError> {
        Ok(vec![unary_abstract::<V>("scale", inputs)?])
    }

    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![self.factor().clone() * inputs[0].clone()])
    }
}

impl<V> JvpOp<V> for ScaleOp<V>
where
    V: TraceValue,
{
    fn jvp<T>(&self, inputs: &[JvpTracer<V, T>]) -> Result<Vec<JvpTracer<V, T>>, TraceError>
    where
        T: TangentSpace<V>,
    {
        expect_input_count(inputs.len(), 1)?;
        let input = &inputs[0];
        Ok(vec![JvpTracer {
            primal: self.factor().clone() * input.primal.clone(),
            tangent: T::scale(self.factor().clone(), input.tangent.clone()),
        }])
    }
}

impl<V> BatchOp<V> for ScaleOp<V>
where
    V: TraceValue,
{
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![Batch::new(inputs[0].lanes().iter().cloned().map(|lane| self.factor().clone() * lane).collect())])
    }
}

impl<V> LinearOp<V> for ScaleOp<V>
where
    V: TraceValue,
{
    fn transpose(
        &self,
        builder: &mut GraphBuilder<LinearOpRef<V>, V>,
        inputs: &[AtomId],
        outputs: &[AtomId],
        output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        expect_input_count(outputs.len(), 1)?;
        expect_input_count(output_cotangents.len(), 1)?;
        let contribution =
            builder.add_equation(Arc::new(ScaleOp::new(self.factor().clone())), vec![output_cotangents[0]])?[0];
        Ok(vec![Some(contribution)])
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{
        parameters::Placeholder,
        tracing_v2::{GraphBuilder, ScalarAbstract, TraceError},
    };

    use super::*;

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    #[test]
    fn add_abstract_eval_rejects_incompatible_inputs() {
        let error = <AddOp as Op<f64>>::abstract_eval(&AddOp, &[ScalarAbstract::F32, ScalarAbstract::F64]).unwrap_err();
        assert_eq!(error, TraceError::IncompatibleAbstractValues { op: "add" });
    }

    #[test]
    fn mul_jvp_matches_the_product_rule() {
        let output = MulOp
            .jvp::<f64>(&[
                JvpTracer { primal: 2.0f64, tangent: 3.0f64 },
                JvpTracer { primal: 5.0f64, tangent: -1.0f64 },
            ])
            .unwrap()
            .pop()
            .unwrap();

        approx_eq(output.primal, 10.0);
        approx_eq(output.tangent, 13.0);
    }

    #[test]
    fn add_batch_requires_matching_lane_counts() {
        let error = AddOp.batch(&[Batch::new(vec![1.0f64, 2.0f64]), Batch::new(vec![3.0f64])]).unwrap_err();
        assert_eq!(error, TraceError::MismatchedBatchSize);
    }

    #[test]
    fn scale_transpose_scales_output_cotangents() {
        let mut forward_builder = GraphBuilder::<LinearOpRef<f64>, f64>::new();
        let input = forward_builder.add_input(&1.0f64);
        let output = forward_builder.add_equation(Arc::new(ScaleOp::new(3.0f64)), vec![input]).unwrap()[0];

        let mut transpose_builder = GraphBuilder::<LinearOpRef<f64>, f64>::new();
        let output_cotangent = transpose_builder.add_input(&1.0f64);
        let contribution = ScaleOp::new(3.0f64)
            .transpose(&mut transpose_builder, &[input], &[output], &[output_cotangent])
            .unwrap()[0]
            .unwrap();

        let transpose_graph = transpose_builder.build::<f64, f64>(vec![contribution], Placeholder, Placeholder);
        approx_eq(transpose_graph.call(2.0f64).unwrap(), 6.0);
    }
}
