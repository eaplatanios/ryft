use std::{cell::RefCell, marker::PhantomData, rc::Rc, sync::Arc};

use ryft_macros::Parameter;

use crate::{
    parameters::{Parameter, Parameterized, ParameterizedFamily},
    tracing_v2::{
        OneLike, TraceError, TraceValue,
        context::{JvpContext, TransposeContext},
        forward::{JvpTracer, TangentSpace},
        graph::{AtomId, Graph, GraphBuilder},
        ops::{AddOp, LinearOpRef, NegOp, ScaleOp},
    },
};

#[derive(Clone, Debug, Parameter)]
pub struct LinearTerm<V>
where
    V: TraceValue,
{
    atom: AtomId,
    builder: Rc<RefCell<GraphBuilder<LinearOpRef<V>, V>>>,
}

impl<V> LinearTerm<V>
where
    V: TraceValue,
{
    #[inline]
    fn add(self, rhs: Self) -> Self {
        debug_assert!(Rc::ptr_eq(&self.builder, &rhs.builder));
        let atom = self
            .builder
            .borrow_mut()
            .add_equation(Arc::new(AddOp), vec![self.atom, rhs.atom])
            .expect("staging linear addition should succeed")[0];
        Self { atom, builder: self.builder }
    }

    #[inline]
    fn neg(self) -> Self {
        let atom = self
            .builder
            .borrow_mut()
            .add_equation(Arc::new(NegOp), vec![self.atom])
            .expect("staging linear negation should succeed")[0];
        Self { atom, builder: self.builder }
    }

    #[inline]
    fn scale(self, factor: V) -> Self {
        let atom = self
            .builder
            .borrow_mut()
            .add_equation(Arc::new(ScaleOp::new(factor)), vec![self.atom])
            .expect("staging linear scaling should succeed")[0];
        Self { atom, builder: self.builder }
    }
}

impl<V> TangentSpace<V> for LinearTerm<V>
where
    V: TraceValue,
{
    #[inline]
    fn add(lhs: Self, rhs: Self) -> Self {
        lhs.add(rhs)
    }

    #[inline]
    fn neg(value: Self) -> Self {
        value.neg()
    }

    #[inline]
    fn scale(factor: V, tangent: Self) -> Self {
        tangent.scale(factor)
    }
}

pub type Linearized<V> = JvpTracer<V, LinearTerm<V>>;

pub struct LinearProgram<V, Input, Output>
where
    V: TraceValue,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    graph: Graph<LinearOpRef<V>, V, Input, Output>,
    zero: V,
    marker: PhantomData<fn(Input) -> Output>,
}

impl<V, Input, Output> LinearProgram<V, Input, Output>
where
    V: TraceValue,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    #[inline]
    pub fn graph(&self) -> &Graph<LinearOpRef<V>, V, Input, Output> {
        &self.graph
    }

    pub fn call(&self, input: Input) -> Result<Output, TraceError>
    where
        Input::ParameterStructure: PartialEq,
        Output::ParameterStructure: Clone,
    {
        self.graph.call(input)
    }

    pub fn transpose(&self) -> Result<LinearProgram<V, Output, Input>, TraceError>
    where
        Input::ParameterStructure: Clone,
        Output::ParameterStructure: Clone,
    {
        fn accumulate<V>(
            builder: &mut GraphBuilder<LinearOpRef<V>, V>,
            adjoints: &mut [Option<AtomId>],
            atom: AtomId,
            contribution: AtomId,
        ) where
            V: TraceValue,
        {
            adjoints[atom] = Some(match adjoints[atom] {
                Some(existing) => builder
                    .add_equation(Arc::new(AddOp), vec![existing, contribution])
                    .expect("accumulating cotangents should succeed")[0],
                None => contribution,
            });
        }

        let mut builder = GraphBuilder::<LinearOpRef<V>, V>::new();
        let mut output_cotangent_inputs = Vec::with_capacity(self.graph.outputs().len());
        for output in self.graph.outputs() {
            let abstract_value =
                self.graph.atom(*output).ok_or(TraceError::UnboundAtomId { id: *output })?.abstract_value.clone();
            output_cotangent_inputs.push(builder.add_input_abstract(abstract_value));
        }

        let mut adjoints = vec![None; self.graph.atom_count()];
        for (cotangent, output) in output_cotangent_inputs.into_iter().zip(self.graph.outputs().iter().copied()) {
            accumulate(&mut builder, adjoints.as_mut_slice(), output, cotangent);
        }

        for equation in self.graph.equations().iter().rev() {
            let equation_output_cotangents =
                equation.outputs.iter().map(|output| adjoints[*output]).collect::<Option<Vec<_>>>();
            let Some(equation_output_cotangents) = equation_output_cotangents else {
                continue;
            };
            let input_cotangents = {
                let mut transpose_context = TransposeContext::new(&mut builder);
                equation.op.transpose(
                    &mut transpose_context,
                    equation.inputs.as_slice(),
                    equation.outputs.as_slice(),
                    equation_output_cotangents.as_slice(),
                )?
            };
            for (input, contribution) in equation.inputs.iter().copied().zip(input_cotangents) {
                if let Some(contribution) = contribution {
                    accumulate(&mut builder, adjoints.as_mut_slice(), input, contribution);
                }
            }
        }

        let zero_atom = builder.add_constant(self.zero.clone());
        let outputs = self
            .graph
            .input_atoms()
            .iter()
            .copied()
            .map(|input| adjoints[input].unwrap_or(zero_atom))
            .collect::<Vec<_>>();
        Ok(LinearProgram {
            graph: builder.build::<Output, Input>(
                outputs,
                self.graph.output_structure().clone(),
                self.graph.input_structure().clone(),
            ),
            zero: self.zero.clone(),
            marker: PhantomData,
        })
    }
}

pub fn jvp_program<'context, Context, F, Input, Output, V>(
    context: &'context mut Context,
    function: F,
    primals: Input,
) -> Result<(Output, LinearProgram<V, Input, Output>), TraceError>
where
    V: TraceValue,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<Linearized<V>>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<Linearized<V>>,
    F: FnOnce(&mut JvpContext<'context, Context, V>, Input::To<Linearized<V>>) -> Output::To<Linearized<V>>,
{
    let input_structure = primals.parameter_structure();
    let primal_parameters = primals.into_parameters().collect::<Vec<_>>();
    let zero = primal_parameters
        .first()
        .map(|value| value.zero_like())
        .ok_or(TraceError::EmptyParameterizedValue)?;

    let mut jvp_context = JvpContext::new(context);
    let traced_input = Input::To::<Linearized<V>>::from_parameters(
        input_structure.clone(),
        primal_parameters.into_iter().map(|primal| {
            let builder = jvp_context.linear_builder();
            let atom = builder.borrow_mut().add_input(&primal);
            JvpTracer { primal, tangent: LinearTerm { atom, builder } }
        }),
    )?;

    let (output_structure, primal_outputs, tangent_outputs) = {
        let traced_output = function(&mut jvp_context, traced_input);
        let output_structure = traced_output.parameter_structure();
        let traced_outputs = traced_output.into_parameters().collect::<Vec<_>>();
        let primal_outputs = traced_outputs.iter().map(|output| output.primal.clone()).collect::<Vec<_>>();
        let tangent_outputs = traced_outputs.into_iter().map(|output| output.tangent.atom).collect::<Vec<_>>();
        (output_structure, primal_outputs, tangent_outputs)
    };

    let primal_output = Output::from_parameters(output_structure.clone(), primal_outputs)?;
    let (_context, builder) = jvp_context.finish();
    let builder = match Rc::try_unwrap(builder) {
        Ok(builder) => builder.into_inner(),
        Err(_) => {
            return Err(TraceError::InternalInvariantViolation("linearization builder escaped the tracing scope"));
        }
    };
    Ok((
        primal_output,
        LinearProgram {
            graph: builder.build::<Input, Output>(tangent_outputs, input_structure, output_structure),
            zero,
            marker: PhantomData,
        },
    ))
}

pub fn linearize<'context, Context, F, Input, Output, V>(
    context: &'context mut Context,
    function: F,
    primals: Input,
) -> Result<(Output, LinearProgram<V, Input, Output>), TraceError>
where
    V: TraceValue,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<Linearized<V>>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<Linearized<V>>,
    F: FnOnce(&mut JvpContext<'context, Context, V>, Input::To<Linearized<V>>) -> Output::To<Linearized<V>>,
{
    jvp_program(context, function, primals)
}

pub fn vjp<'context, Context, F, Input, Output, V>(
    context: &'context mut Context,
    function: F,
    primals: Input,
) -> Result<(Output, LinearProgram<V, Output, Input>), TraceError>
where
    V: TraceValue,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<Linearized<V>>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<Linearized<V>>,
    F: FnOnce(&mut JvpContext<'context, Context, V>, Input::To<Linearized<V>>) -> Output::To<Linearized<V>>,
{
    let (output, pushforward) = jvp_program(context, function, primals)?;
    Ok((output, pushforward.transpose()?))
}

pub fn grad<'context, Context, F, Input, V>(
    context: &'context mut Context,
    function: F,
    primals: Input,
) -> Result<Input, TraceError>
where
    V: TraceValue + OneLike,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<Linearized<V>>,
    F: FnOnce(&mut JvpContext<'context, Context, V>, Input::To<Linearized<V>>) -> Linearized<V>,
{
    let (output, pullback): (V, LinearProgram<V, V, Input>) = vjp(context, function, primals)?;
    pullback.call(output.one_like())
}
