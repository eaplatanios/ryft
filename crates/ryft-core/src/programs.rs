use std::{
    borrow::Cow,
    collections::{HashMap, hash_map::Entry},
    fmt::{Debug, Display},
};

use dyn_clone::{DynClone, clone_trait_object};
use thiserror::Error;

use crate::{
    assert_input_count_matches,
    parameters::{Parameter, Parameterized},
    tracing::Tracer,
    types::{Type, Typed, array_structure_type::ArrayStructureTypeBroadcastingError},
};

pub type AtomId = usize;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Constant<V> {
    pub id: AtomId,
    pub value: V,
}

#[derive(Clone, Debug)]
pub enum ConstantExpression<T, V, O> {
    Value { tpe: T, value: V },
    // TODO(eaplatanios): Document that this op is only allowed to have a single output.
    Expression { tpe: T, op: O, inputs: Vec<ConstantExpression<T, V, O>> },
}

impl<T: Clone, V: Typed<T>, O> ConstantExpression<T, V, O> {
    #[inline]
    pub fn new_value(value: V) -> Self {
        let tpe = value.tpe();
        Self::Value { tpe, value }
    }

    #[inline]
    pub fn new_expression(op: O, inputs: Vec<ConstantExpression<T, V, O>>) -> Self
    where
        O: Op<T>,
    {
        let input_types = inputs.iter().map(|input| input.tpe()).collect::<Vec<_>>();
        let input_types = input_types.iter().collect::<Vec<_>>();
        let output_types = op.infer_output_types(input_types.as_slice()).unwrap();
        assert_eq!(output_types.len(), 1);
        let tpe = output_types[0].clone();
        Self::Expression { tpe, op, inputs }
    }
}

impl<T, V: Display, O: Display> Display for ConstantExpression<T, V, O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            ConstantExpression::Value { value, .. } => write!(f, "{value}"),
            ConstantExpression::Expression { op, inputs, .. } => {
                write!(f, "{}({})", op, inputs.iter().map(|input| input.to_string()).collect::<Vec<_>>().join(", "))
            }
        }
    }
}

impl<T: Clone, V: Clone + Typed<T>, O: Clone + InterpretableOp<T, V>> ConstantExpression<T, V, O> {
    #[inline]
    pub fn value(self) -> V {
        match self {
            ConstantExpression::Value { value, .. } => value,
            ConstantExpression::Expression { op, inputs, .. } => {
                let input_values = inputs.clone().into_iter().map(|input| input.value()).collect::<Vec<_>>();
                let input_values = input_values.iter().collect::<Vec<_>>();
                let outputs = op.interpret(input_values.as_slice()).unwrap();
                assert_eq!(outputs.len(), 1);
                outputs[0].clone()
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Variable<T> {
    pub id: AtomId,
    pub tpe: T,
}

impl<T: Clone> Typed<T> for Variable<T> {
    #[inline]
    fn tpe(&self) -> T {
        self.tpe.clone()
    }
}

pub trait Op<T>: Display + DynClone {
    /// Type checks the operation's inputs and outputs.
    /// Returns the expected output shapes given the input shapes.
    fn infer_output_types(&self, input_types: &[&T]) -> Result<Vec<T>, ProgramError>;
}

impl<T> Debug for dyn Op<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

impl<T, O: Op<T> + ?Sized> Op<T> for &O {
    #[inline]
    fn infer_output_types(&self, input_types: &[&T]) -> Result<Vec<T>, ProgramError> {
        (**self).infer_output_types(input_types)
    }
}

impl<T> Op<T> for Box<dyn Op<T>> {
    #[inline]
    fn infer_output_types(&self, input_types: &[&T]) -> Result<Vec<T>, ProgramError> {
        (**self).infer_output_types(input_types)
    }
}

clone_trait_object!(<T> Op<T>);

/// [Op] that is "interpretable" for values of type `V`. Being "interpretable" means that it can take a [Vec]
/// of input values and produce a corresponding [Vec] of output values. [Op]s are mean to represent operations
/// in a [Program] but do not need to define how the operation is performed. That implementation is dependent
/// on the backend that is being used and can vary. [InterpretableOp]s effectively provide such an implementation
/// for values of a specific type (or even more than one types) which enables easier debugging of programs that
/// use them, using [Program::interpret]. Though given that debugging and testing is the main target use case for
/// this kind of program interpretation, typically [InterpretableOp::interpret] will not be optimized or very
/// efficient, but it will be correct.
pub trait InterpretableOp<T, V>: Op<T> {
    /// Interprets this [InterpretableOp] using the provided input values, producing corresponding output values.
    fn interpret(&self, inputs: &[&V]) -> Result<Vec<V>, ProgramError>;
}

impl<T, V> Debug for dyn InterpretableOp<T, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

impl<T, V, O: InterpretableOp<T, V> + ?Sized> InterpretableOp<T, V> for &O {
    #[inline]
    fn interpret(&self, inputs: &[&V]) -> Result<Vec<V>, ProgramError> {
        (**self).interpret(inputs)
    }
}

impl<T, V> Op<T> for Box<dyn InterpretableOp<T, V>> {
    fn infer_output_types(&self, input_types: &[&T]) -> Result<Vec<T>, ProgramError> {
        (**self).infer_output_types(input_types)
    }
}

impl<T, V> InterpretableOp<T, V> for Box<dyn InterpretableOp<T, V>> {
    #[inline]
    fn interpret(&self, inputs: &[&V]) -> Result<Vec<V>, ProgramError> {
        (**self).interpret(inputs)
    }
}

clone_trait_object!(<T, V> InterpretableOp<T, V>);

pub trait LinearOp<T, V>: Op<T> {
    // TODO(eaplatanios): We need to know which argument an op is linear with respect to. For example, for multiplication,
    //  say we have `MulOp`. When we differentiate it, we'll get instances of a `LinearMulOp` where either the left or the
    //  right argument is linear. We cannot be linear with respect to both arguments.
    // TODO(eaplatanios): Something like this but which instead operates over tracers for building the transpose program:
    //  `fn transpose(&self, output_tangents: &[&Tracer<T, V, Self>]) -> Vec<Tracer<T, V, Self>>`,
    //  where the returned [AtomId]s correspond to the input tangents and refer to potentially new atoms that may have
    //  been added to the builder.
    // TODO(eaplatanios): This function should return a [Result] instead.
    fn transpose(
        &self,
        inputs: &[&Tracer<T, V, Box<dyn LinearOp<T, V>>>],
        output_tangents: &[&Tracer<T, V, Box<dyn LinearOp<T, V>>>],
    ) -> Vec<Tracer<T, V, Box<dyn LinearOp<T, V>>>>;
}

impl<T, V> Debug for dyn LinearOp<T, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

impl<T, V, O: LinearOp<T, V> + ?Sized> LinearOp<T, V> for &O {
    fn transpose(
        &self,
        inputs: &[&Tracer<T, V, Box<dyn LinearOp<T, V>>>],
        output_tangents: &[&Tracer<T, V, Box<dyn LinearOp<T, V>>>],
    ) -> Vec<Tracer<T, V, Box<dyn LinearOp<T, V>>>> {
        (**self).transpose(inputs, output_tangents)
    }
}

impl<T, V> Op<T> for Box<dyn LinearOp<T, V>> {
    fn infer_output_types(&self, input_types: &[&T]) -> Result<Vec<T>, ProgramError> {
        (**self).infer_output_types(input_types)
    }
}

impl<T, V> LinearOp<T, V> for Box<dyn LinearOp<T, V>> {
    fn transpose(
        &self,
        inputs: &[&Tracer<T, V, Box<dyn LinearOp<T, V>>>],
        output_tangents: &[&Tracer<T, V, Box<dyn LinearOp<T, V>>>],
    ) -> Vec<Tracer<T, V, Box<dyn LinearOp<T, V>>>> {
        (**self).transpose(inputs, output_tangents)
    }
}

clone_trait_object!(<T, V> LinearOp<T, V>);

// TODO(eaplatanios): Do we need the cross-product of all kinds of ops? That's a little annoying.
pub trait LinearInterpretableOp<T, V>: LinearOp<T, V> + InterpretableOp<T, V> {}

impl<T, V> Debug for dyn LinearInterpretableOp<T, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

impl<T, V, O: LinearInterpretableOp<T, V> + ?Sized> LinearInterpretableOp<T, V> for &O {}

impl<T, V> Op<T> for Box<dyn LinearInterpretableOp<T, V>> {
    fn infer_output_types(&self, input_types: &[&T]) -> Result<Vec<T>, ProgramError> {
        (**self).infer_output_types(input_types)
    }
}

impl<T, V> InterpretableOp<T, V> for Box<dyn LinearInterpretableOp<T, V>> {
    #[inline]
    fn interpret(&self, inputs: &[&V]) -> Result<Vec<V>, ProgramError> {
        (**self).interpret(inputs)
    }
}

impl<T, V> LinearOp<T, V> for Box<dyn LinearInterpretableOp<T, V>> {
    fn transpose(
        &self,
        inputs: &[&Tracer<T, V, Box<dyn LinearOp<T, V>>>],
        output_tangents: &[&Tracer<T, V, Box<dyn LinearOp<T, V>>>],
    ) -> Vec<Tracer<T, V, Box<dyn LinearOp<T, V>>>> {
        (**self).transpose(inputs, output_tangents)
    }
}

impl<T, V> LinearInterpretableOp<T, V> for Box<dyn LinearInterpretableOp<T, V>> {}

clone_trait_object!(<T, V> LinearInterpretableOp<T, V>);

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Expression<O> {
    pub op: O,
    pub inputs: Vec<AtomId>,
    pub outputs: Vec<AtomId>,
}

impl<O: Display> Display for Expression<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // TODO(eaplatanios): Improve this rendering.
        write!(f, "{:?} = {} {:?}", self.outputs, self.op, self.inputs)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Program<T, V, O> {
    pub inputs: Vec<Variable<T>>,
    pub constants: Vec<Constant<V>>,
    pub variables: Vec<Variable<T>>,
    pub expressions: Vec<Expression<O>>,
    pub outputs: Vec<Variable<T>>,
}

impl<T, V, O> Program<T, V, O> {
    /// Returns the number of inputs of this [Program].
    #[inline]
    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }

    /// Returns the number of outputs of this [Program].
    #[inline]
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    /// Performs type checking for this [Program] and returns an error if there are any typing issues.
    pub fn type_check(&self) -> Result<(), ProgramError>
    where
        T: Clone + Type,
        V: Typed<T>,
        O: Op<T>,
    {
        // TODO(eaplatanios): Should this be checking for contiguity of all the atom IDs that other parts of
        //  code rely on (e.g., the [Program] interpretation code). This should also check that the atom IDs
        //  are unique across `inputs`, `constants`, `variables`, and `outputs`.

        // This [HashMap] will be used to store the inferred types of all atoms in this program.
        let mut atom_types = HashMap::new();

        // Collect the types of the inputs.
        for input in &self.inputs {
            match atom_types.entry(input.id) {
                Entry::Occupied(_) => return Err(ProgramError::DuplicateAtomId { id: input.id }),
                Entry::Vacant(entry) => _ = entry.insert(input.tpe()),
            }
        }

        // Collect the types of the constants.
        for constant in &self.constants {
            match atom_types.entry(constant.id) {
                Entry::Occupied(_) => return Err(ProgramError::DuplicateAtomId { id: constant.id }),
                Entry::Vacant(entry) => _ = entry.insert(constant.tpe()),
            }
        }

        // Performing type inference over a [Program] consists of going over the [Expression]s in that [Program]
        // and using the underlying [Op]s to infer the output types of each [Expression]. While doing that, we keep
        // updating `atom_types` with the types we infer such that they can be used for inferring other types that
        // depend on them, downstream in the [Program].
        for expression in &self.expressions {
            // Verify that the input count of the current [Expression] matches that expected by the underlying [Op].
            // if let Some(input_count) = expression.op.input_count() {
            //     if expression.inputs.len() != input_count {
            //         return Err(ProgramError::InvalidInputCount {
            //             expected: input_count,
            //             got: expression.inputs.len(),
            //         });
            //     }
            // }

            // Verify that the output count of the current [Expression] matches that expected by the underlying [Op].
            // if let Some(output_count) = expression.op.output_count() {
            //     if expression.outputs.len() != output_count {
            //         return Err(ProgramError::InvalidOutputCount {
            //             expression: expression.to_string(),
            //             expected: output_count,
            //             got: expression.outputs.len(),
            //         });
            //     }
            // }

            // Collect the input types for this expression.
            let mut input_types = Vec::with_capacity(expression.inputs.len());
            for input_id in &expression.inputs {
                if let Some(input_type) = atom_types.get(input_id) {
                    input_types.push(input_type);
                } else {
                    return Err(ProgramError::UnboundAtomId { id: *input_id });
                }
            }

            // Infer the corresponding output types using the underlying [Op].
            let output_types = expression.op.infer_output_types(input_types.as_slice())?;

            // Verify that the output [AtomId]s of [Expression] are unique and store their inferred types
            // so that they can be used to infer downstream variable types later on.
            for (output_id, output_type) in expression.outputs.iter().zip(output_types) {
                match atom_types.entry(*output_id) {
                    Entry::Occupied(_) => return Err(ProgramError::DuplicateAtomId { id: *output_id }),
                    Entry::Vacant(entry) => _ = entry.insert(output_type),
                }
            }
        }

        // TODO(eaplatanios): Check for uniqueness of output IDs (that e.g., program interpretation relies on).

        // Check that the [Program] output types match the inferred output types.
        for output in &self.outputs {
            if let Some(output_type) = atom_types.get(&output.id) {
                if !output_type.is_subtype_of(&output.tpe) {
                    return Err(ProgramError::InvalidAtomType {
                        id: output.id,
                        expected: output.tpe.to_string(),
                        got: output_type.to_string(),
                    });
                }
            } else {
                return Err(ProgramError::UnboundAtomId { id: output.id });
            }
        }

        Ok(())
    }

    /// Interprets this [Program] using the provided values for the inputs. Interpreting a [Program] is a very
    /// inefficient way of executing it and is mainly targetted at debugging use cases. For more efficient [Program]
    /// execution you should consider using just-in-time (JIT) compilation instead of this function.
    ///
    /// The length of the provided inputs must match the [Program::input_count] of this [Program]. The resulting [Vec]
    /// of outputs will correspondingly have a length matching the [Program::output_count] of this [Program].
    // TODO(eaplatanios): Mention memory (i.e., all temporary variables are kept alive throughout the interpretation).
    pub fn interpret(&self, inputs: &[&V]) -> Result<Vec<V>, ProgramError>
    where
        T: Clone + Type,
        V: ToOwned<Owned = V> + Typed<T>,
        O: InterpretableOp<T, V>,
    {
        // Perform type checking of this [Program] and verify that the number and type of inputs is correct. This is
        // not cheap but it is done in order to ensure that this function is safe. This is important because the target
        // use case for this function is debugging and it is not meant to be used as the primary way of executing
        // [Program]s. Therefore, it is acceptable and perhaps even expected that it is not super efficient.
        self.type_check()?;

        assert_input_count_matches!(inputs.len(), self.inputs.len());

        for (input_variable, input_value) in self.inputs.iter().zip(inputs.iter()) {
            let expected_type = input_variable.tpe();
            let got_type = input_value.tpe();
            if !got_type.is_subtype_of(&expected_type) {
                return Err(ProgramError::InvalidAtomType {
                    id: input_variable.id,
                    expected: expected_type.to_string(),
                    got: got_type.to_string(),
                });
            }
        }

        // The [Atom]s of a valid [Program] have monotonically increasing contiguous IDs starting at 0. Therefore, to
        // keep track of the inputs, constants, and temporary values generated during the interpretation of this
        // [Program] we use a [Vec] with a capacity set to the total number of [Atom]s in the [Program]. Then, to read
        // or write the value of an [Atom] we can index this [Vec] using that [Atom]'s ID. Note that this [Vec] holds
        // [Cow]s instead of owned values. That is done in order to avoid cloning any of the input values and any of
        // the [Program] [Constant]s. Any newly computed values during the interpretation of this [Program] will be
        // stored in this [Vec] as [Cow::Owned] instances.
        let environment_len = self.inputs.len()
            + self.constants.len()
            + self.expressions.iter().map(|expression| expression.outputs.len()).sum::<usize>()
            + self.outputs.len();
        let mut environment: Vec<Option<Cow<'_, V>>> = vec![None; environment_len];

        // Write all input values to the current interpretation environment.
        self.inputs
            .iter()
            .zip(inputs.into_iter())
            .for_each(|(variable, value)| environment[variable.id] = Some(Cow::Borrowed(*value)));

        // Write all constant values to the current interpretation environment.
        self.constants
            .iter()
            .for_each(|constant| environment[constant.id] = Some(Cow::Borrowed(&constant.value)));

        // Go through all [Expression]s in this [Program] in the order in which they appear, interpreting the
        // underlying [Op]s and updating the environment according to their outputs.
        for expression in self.expressions.iter() {
            // The type checking that was performed earlier in this function guarantees that all inputs for this
            // expression must have been initialized by this point (and thus it should be safe to unwrap them).
            let inputs = expression
                .inputs
                .iter()
                .map(|input| environment[*input].as_ref().unwrap().as_ref())
                .collect::<Vec<_>>();
            // if let Some(input_count) = expression.op.input_count() {
            //     if inputs.len() != input_count {
            //         return Err(ProgramError::InvalidInputCount { expected: input_count, got: inputs.len() });
            //     }
            // }
            let outputs = expression.op.interpret(inputs.as_slice())?;
            // if let Some(output_count) = expression.op.output_count() {
            //     if outputs.len() != output_count {
            //         return Err(ProgramError::InvalidOutputCount {
            //             expression: expression.to_string(),
            //             expected: output_count,
            //             got: outputs.len(),
            //         });
            //     }
            // }
            expression
                .outputs
                .iter()
                .zip(outputs)
                .for_each(|(atom_id, value)| environment[*atom_id] = Some(Cow::Owned(value)));
        }

        // Collect the [Program] outputs from the environment and return them. As we pull values out of the environment
        // we replace them with [None] in the environment. This relies on the assumption that the [Program] outputs
        // have unique [AtomId]s. We also check for this uniqueness property and return an error if it is violated.
        // We could avoid returning an error by cloing the corresponding values but this assumption represents a good
        // requirement for [Program]s anyway and so we have decided to keep it. Note that this condition is checked
        // in the type checking that is performed earlier in this function and so we should never get to this point
        // if it is, in fact, violated.
        let mut outputs = Vec::with_capacity(self.outputs.len());
        for output in &self.outputs {
            if let Some(value) = environment.get_mut(output.id) {
                if let Some(value) = value.take() {
                    match value {
                        Cow::Owned(value) => {
                            // Check the output value type before returning it. This should ideally never be wrong,
                            // but if it is, it could indicate a bug in the [Op] type checking or interpretation.
                            let expected_type = output.tpe();
                            let got_type = value.tpe();
                            if !got_type.is_subtype_of(&expected_type) {
                                return Err(ProgramError::InvalidAtomType {
                                    id: output.id,
                                    expected: expected_type.to_string(),
                                    got: got_type.to_string(),
                                });
                            }
                            outputs.push(value);
                        }
                        Cow::Borrowed(_) => {
                            // This means that there is an output whose [AtomId] matches that of an input or constant.
                            return Err(ProgramError::DuplicateAtomId { id: output.id });
                        }
                    }
                } else {
                    // This means that we have multiple outputs with the same [AtomId].
                    return Err(ProgramError::DuplicateAtomId { id: output.id });
                }
            } else {
                // This means that we have an invalid output [AtomId].
                return Err(ProgramError::UnboundAtomId { id: output.id });
            }
        }
        Ok(outputs)
    }
}

impl<T: Display, V, O: Op<T>> Display for Program<T, V, O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inputs = self
            .inputs
            .iter()
            .map(|input| format!("%{}:{}", input.id, input.tpe))
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "lambda {inputs} .\n")?;
        for (index, expression) in self.expressions.iter().enumerate() {
            // TODO(eaplatanios): Include types for the outputs.
            let inputs = expression.inputs.iter().map(|input| format!("%{}", input)).collect::<Vec<_>>().join(" ");
            let outputs = expression.outputs.iter().map(|output| format!("%{}", output)).collect::<Vec<_>>().join(", ");
            let prefix = if index == 0 { "let" } else { "   " };
            write!(f, "{prefix} {outputs} = {} {inputs}\n", expression.op.to_string())?;
        }
        let outputs = self.outputs.iter().map(|output| format!("%{}", output.id)).collect::<Vec<_>>().join(", ");
        write!(f, "in ({outputs})")
        // TODO(eaplatanios): Improve this rendering.
        // { lambda a:float64[] b:float64[] .
        //   let c:float64[] = sin a
        //       d:float64[] = cos a
        //       e:float64[] = mul d b
        //       f:float64[] = neg c
        //       g:float64[] = neg e
        //   in ( f, g ) }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ProgramType<T> {
    pub inputs: Vec<T>,
    pub outputs: Vec<T>,
}

impl<T> ProgramType<T> {
    #[inline]
    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }

    #[inline]
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }
}

// TODO(eaplatanios): Clean this up. Also, in the parameterized program rendering, should there be something
//  about the input and output structures?
pub struct ParameterizedProgram<T, V: Parameter, O, Input: Parameterized<V>, Output: Parameterized<V>> {
    pub program: Program<T, V, O>,
    pub input_structure: Input::ParameterStructure,
    pub output_structure: Output::ParameterStructure,
}

impl<T, V: Parameter, O, Input: Parameterized<V>, Output: Parameterized<V>>
    ParameterizedProgram<T, V, O, Input, Output>
{
    pub fn new(
        program: Program<T, V, O>,
        input_structure: Input::ParameterStructure,
        output_structure: Output::ParameterStructure,
    ) -> Self {
        Self { program, input_structure, output_structure }
    }

    pub fn interpret(&self, input: Input) -> Result<Output, ProgramError>
    where
        T: Clone + Debug + Type,
        V: ToOwned<Owned = V> + Typed<T> + Parameter,
        O: InterpretableOp<T, V> + Debug,
        Output::ParameterStructure: Clone,
    {
        let input = input.parameters().collect::<Vec<_>>();
        let output = self.program.interpret(input.as_slice())?;
        // TODO(eaplatanios): Do not force `unwrap` below.
        Ok(Output::from_parameters(self.output_structure.clone(), output).unwrap())
    }
}

#[derive(Clone)]
pub struct ProgramBuilder<T, V, O> {
    pub constants: Vec<Constant<V>>,
    pub variables: Vec<Variable<T>>,
    pub expressions: Vec<Expression<O>>,
    atom_types: HashMap<AtomId, T>,
    atom_count: usize,
}

impl<T, V, O> ProgramBuilder<T, V, O> {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn atom_type(&self, atom_id: &AtomId) -> Option<&T> {
        self.atom_types.get(atom_id)
    }

    #[inline]
    pub fn add_constant(&mut self, value: V) -> AtomId
    where
        V: Typed<T>,
    {
        let id = self.atom_count;
        self.atom_types.insert(id, value.tpe());
        self.atom_count += 1;
        self.constants.push(Constant { id, value });
        id
    }

    #[inline]
    pub fn add_variable(&mut self, tpe: T) -> AtomId
    where
        T: Clone,
    {
        let id = self.atom_count;
        self.atom_types.insert(id, tpe.clone());
        self.atom_count += 1;
        self.variables.push(Variable { id, tpe });
        id
    }

    #[inline]
    pub fn add_expression(&mut self, op: O, inputs: Vec<AtomId>) -> Result<Vec<AtomId>, ProgramError>
    where
        T: Clone,
        O: Op<T>,
    {
        let input_types = inputs
            .iter()
            .map(|input| self.atom_types.get(input).ok_or(ProgramError::UnboundAtomId { id: *input }))
            .collect::<Result<Vec<_>, ProgramError>>()?;
        let output_types = op.infer_output_types(input_types.as_slice())?;
        let outputs = output_types.into_iter().map(|output_type| self.add_variable(output_type)).collect::<Vec<_>>();
        self.expressions.push(Expression { op, inputs, outputs });
        Ok(self.expressions.last().unwrap().outputs.iter().map(|output| *output).collect())
    }

    #[inline]
    pub fn add_constant_expression(&mut self, expression: ConstantExpression<T, V, O>) -> Result<AtomId, ProgramError>
    where
        T: Clone,
        V: Typed<T>,
        O: Op<T>,
    {
        match expression {
            ConstantExpression::Value { value, .. } => Ok(self.add_constant(value)),
            ConstantExpression::Expression { op, inputs, .. } => {
                let inputs = inputs
                    .into_iter()
                    .map(|input| self.add_constant_expression(input))
                    .collect::<Result<Vec<_>, ProgramError>>()?;
                let outputs = self.add_expression(op, inputs)?;
                // TODO(eaplatanios): This should return an error instead of panicking.
                assert_eq!(outputs.len(), 1);
                Ok(outputs[0])
            }
        }
    }

    pub fn build(self, inputs: Vec<AtomId>, outputs: Vec<AtomId>) -> Result<Program<T, V, O>, ProgramError>
    where
        T: Clone + Debug + Type,
        V: Typed<T>,
        O: Op<T> + Debug,
    {
        // The first thing we need to do is partition `self.variables` into inputs, variables, and outputs, that
        // will go into the resulting [Program]. The IDs of the variables in `self.variables` are already sorted in
        // increasing order by construction. Therefore, in order to partition them into our inputs, variables, and
        // outputs, we just need to traverse them jointly with `inputs` and `outputs` after sorting the latter. This
        // results in this step requiring a single linear pass over `self.variables`. Specifically, since the IDs of
        // the variables are monotonically increasing by design, once we have sorted `inputs` and `outputs`, we can
        // simply keep going over the variables and whenever we hit a match with either the current input or output ID,
        // we increment the corresponding index.
        //
        // Note that we could be doing all this using hash sets for `inputs` and `outputs` but it does not seem worth
        // it given that we expect the size of `inputs` and `outputs` to typically be small for large programs relative
        // to the size of `self.variables`.

        let mut input_ids = inputs;
        let mut output_ids = outputs;

        input_ids.sort_unstable();
        output_ids.sort_unstable();

        let mut inputs = Vec::with_capacity(input_ids.len());
        let mut variables = Vec::with_capacity(self.variables.len() - input_ids.len() - output_ids.len());
        let mut outputs = Vec::with_capacity(output_ids.len());

        let mut input_index = 0;
        let mut output_index = 0;
        for variable in self.variables {
            if input_ids.get(input_index).filter(|&id| id == &variable.id).is_some() {
                inputs.push(variable);
                input_index += 1;
            } else if output_ids.get(output_index).filter(|&id| id == &variable.id).is_some() {
                outputs.push(variable);
                output_index += 1;
            } else {
                variables.push(variable);
            }
        }

        if inputs.len() < input_ids.len() {
            // TODO(eaplatanios): This error is not quite right. The actual thing that went wrong is something else.
            return Err(ProgramError::InvalidInputCount { expected: inputs.len(), got: input_ids.len() });
        }

        if output_index < output_ids.len() {
            // TODO(eaplatanios): This error is not quite right. The actual thing that went wrong is something else.
            return Err(ProgramError::InvalidInputCount { expected: outputs.len(), got: output_ids.len() });
        }

        // Construct the program and perform type checking before returning it.
        let program = Program { inputs, constants: self.constants, variables, expressions: self.expressions, outputs };
        program.type_check()?;
        Ok(program)
    }
}

impl<T, V, O> Default for ProgramBuilder<T, V, O> {
    fn default() -> Self {
        Self {
            constants: Vec::new(),
            variables: Vec::new(),
            expressions: Vec::new(),
            atom_count: 0,
            atom_types: HashMap::new(),
        }
    }
}

#[macro_export]
macro_rules! assert_input_count_matches {
    ($input_count:expr, $expected_input_count:expr) => {
        let input_count = $input_count;
        let expected_input_count = $expected_input_count;
        if input_count != expected_input_count {
            return Err(ProgramError::InvalidInputCount { expected: expected_input_count, got: input_count });
        }
    };
}

#[derive(Error, Debug)]
pub enum ProgramError {
    #[error("invalid number of inputs; {got} but expected {expected}")]
    InvalidInputCount { expected: usize, got: usize },

    #[error("invalid number of outputs for expression: {expression}; got {got} but expected {expected}")]
    InvalidOutputCount { expression: String, expected: usize, got: usize },

    #[error("unbound atom ID: {id}")]
    UnboundAtomId { id: AtomId },

    #[error("duplicate atom ID: {id}")]
    DuplicateAtomId { id: AtomId },

    #[error("expected subtype of {expected} for the type of atom {id} but got {got}")]
    InvalidAtomType { id: AtomId, expected: String, got: String },

    #[error("{0}")]
    ArrayStructureTypeBroadcastingError(#[from] ArrayStructureTypeBroadcastingError),
    //
    // #[error("encountered tracers with mismatched program builders")]
    // MismatchedProgramBuilders,

    // #[error("borrow error")]
    // BorrowError(#[from] BorrowError),

    // #[error("borrow mutable error")]
    // BorrowMutError(#[from] BorrowMutError),
}
