use std::marker::PhantomData;

struct TypeError;

struct ProgramError;

trait Type {}

trait DifferentiableType: Type {}

#[derive(Clone)]
struct ArrayType;

impl Type for ArrayType {}

impl DifferentiableType for ArrayType {}

trait Value<T: Type>: Clone {}

#[derive(Clone)]
struct Factor;

impl Value<ArrayType> for Factor {}

trait Operation<T: Type> {
    fn name(&self) -> &'static str;

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError>;

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        let _ = indentation;
        formatter.write_str(self.name())
    }
}

struct AbstractTracingContext<'context, T: Type, V: Value<T>, O: Operation<T>> {
    marker: PhantomData<(&'context (), T, V, O)>,
}

struct Cotangent<'context, T: Type, V: Value<T>, O: Operation<T>> {
    marker: PhantomData<(&'context (), T, V, O)>,
}

trait TransposableOperation<T: Type, V: Value<T>, O: Operation<T>>: Operation<T> {
    fn transpose<'transpose>(
        &self,
        context: &mut AbstractTracingContext<'transpose, T, V, O>,
        input_types: &[&T],
        output_cotangents: &[Cotangent<'transpose, T, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, O>>, ProgramError>;
}

trait MaybeZeroOperation<T: Type> {}

#[derive(Clone)]
struct ZeroOperation<T: Type> {
    marker: PhantomData<T>,
}

#[derive(Clone)]
struct AddOperation;

trait TransposableProgramOperation<T: DifferentiableType, V: Value<T>>: Operation<T> + Sized {
    fn transpose_program(program: &Program<T, V, Self, Vec<V>, Vec<V>>)
    -> Result<Program<T, V, Self, Vec<V>, Vec<V>>, ProgramError>;
}

struct Program<T: Type, V: Value<T>, O: Operation<T>, Input, Output> {
    marker: PhantomData<(T, V, O, Input, Output)>,
}

impl<
    T: DifferentiableType,
    V: Value<T>,
    O: TransposableOperation<T, V, O> + MaybeZeroOperation<T> + From<ZeroOperation<T>> + From<AddOperation>,
    Input,
    Output,
> Program<T, V, O, Input, Output>
{
    fn transpose(&self) -> Result<Program<T, V, O, Output, Input>, ProgramError> {
        Ok(Program { marker: PhantomData })
    }
}

#[derive(Clone)]
struct ConstantOperation<T: Type, V> {
    marker: PhantomData<(T, V)>,
}

impl<T: Type, V> Operation<T> for ConstantOperation<T, V> {
    fn name(&self) -> &'static str {
        "constant"
    }

    fn infer_output_types(&self, _input_types: &[T]) -> Result<Vec<T>, TypeError> {
        Ok(Vec::new())
    }
}

impl<T: Type, V: Value<T>, O: Operation<T>, C> TransposableOperation<T, V, O> for ConstantOperation<T, C> {
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, T, V, O>,
        _input_types: &[&T],
        _output_cotangents: &[Cotangent<'transpose, T, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, O>>, ProgramError> {
        Ok(Vec::new())
    }
}

trait FutureTranspositionBound {}

#[derive(Clone, ryft::TransposableOperation)]
#[ryft(crate = "crate")]
#[ryft(bounds(transposition(FutureTranspositionBound)))]
enum LinearOperation<V: Value<ArrayType>> {
    Constant(ConstantOperation<ArrayType, V>),
}

impl<V: Value<ArrayType>> Operation<ArrayType> for LinearOperation<V> {
    fn name(&self) -> &'static str {
        "linear"
    }

    fn infer_output_types(&self, _input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        Ok(Vec::new())
    }
}

impl<V: Value<ArrayType>> MaybeZeroOperation<ArrayType> for LinearOperation<V> {}

impl<V: Value<ArrayType>> From<ZeroOperation<ArrayType>> for LinearOperation<V> {
    fn from(_operation: ZeroOperation<ArrayType>) -> Self {
        Self::Constant(ConstantOperation { marker: PhantomData })
    }
}

impl<V: Value<ArrayType>> From<AddOperation> for LinearOperation<V> {
    fn from(_operation: AddOperation) -> Self {
        Self::Constant(ConstantOperation { marker: PhantomData })
    }
}

fn main() {
    let operation = LinearOperation::<Factor>::Constant(ConstantOperation { marker: PhantomData });
    let mut context = AbstractTracingContext::<ArrayType, Factor, LinearOperation<Factor>> { marker: PhantomData };

    let _ = <LinearOperation<Factor> as TransposableOperation<
        ArrayType,
        Factor,
        LinearOperation<Factor>,
    >>::transpose(&operation, &mut context, &[], &[]);
}
