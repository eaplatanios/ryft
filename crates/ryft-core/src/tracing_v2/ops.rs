use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

use crate::tracing_v2::{
    TraceError, TraceValue,
    batch::Batch,
    context::TransposeContext,
    forward::{JvpTracer, TangentSpace},
    graph::AtomId,
};

pub trait Op<V>: Debug + Display
where
    V: TraceValue,
{
    fn name(&self) -> &'static str;

    fn abstract_eval(&self, inputs: &[V::Abstract]) -> Result<Vec<V::Abstract>, TraceError>;

    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError>;
}

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

pub type StagedOpRef<V> = Arc<dyn StagedOp<V>>;

pub trait JvpOp<V>: Op<V>
where
    V: TraceValue,
{
    fn jvp<T>(&self, inputs: &[JvpTracer<V, T>]) -> Result<Vec<JvpTracer<V, T>>, TraceError>
    where
        T: TangentSpace<V>;
}

pub trait BatchOp<V>: Op<V>
where
    V: TraceValue,
{
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError>;
}

pub trait LinearOp<V>: Op<V>
where
    V: TraceValue,
{
    fn transpose(
        &self,
        context: &mut TransposeContext<'_, V>,
        inputs: &[AtomId],
        outputs: &[AtomId],
        output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError>;
}

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
        context: &mut TransposeContext<'_, V>,
        inputs: &[AtomId],
        outputs: &[AtomId],
        output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError> {
        (**self).transpose(context, inputs, outputs, output_cotangents)
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
        _context: &mut TransposeContext<'_, V>,
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
        context: &mut TransposeContext<'_, V>,
        inputs: &[AtomId],
        outputs: &[AtomId],
        output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        expect_input_count(outputs.len(), 1)?;
        expect_input_count(output_cotangents.len(), 1)?;
        let contribution = context.graph_builder().add_equation(Arc::new(NegOp), vec![output_cotangents[0]])?[0];
        Ok(vec![Some(contribution)])
    }
}

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
        context: &mut TransposeContext<'_, V>,
        inputs: &[AtomId],
        outputs: &[AtomId],
        output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        expect_input_count(outputs.len(), 1)?;
        expect_input_count(output_cotangents.len(), 1)?;
        let contribution = context
            .graph_builder()
            .add_equation(Arc::new(ScaleOp::new(self.factor().clone())), vec![output_cotangents[0]])?[0];
        Ok(vec![Some(contribution)])
    }
}
