use std::fmt::Debug;
use std::marker::PhantomData;

use half::{bf16, f16};

use crate::contexts::Context;
use crate::domains::Domain;
use crate::operations::InterpretableOperation;
use crate::operations::scalars::{LinearScalarOperation, ScalarOperation};
use crate::programs::{ProgramError, Value};
use crate::types::DataType;

/// Stateless [`Domain`] that uses [`DataType`] for scalar metadata and Rust scalar values such as `f32` for
/// runtime values. [`ScalarDomain`] is the minimal scalar-only backend used throughout tests and examples in
/// `ryft-core`. It demonstrates the intended role of an eager [`Context`] in the smallest possible form: there are no
/// device handles, no mesh states, and no backend registries; just the built-in [`ScalarOperation`] variants plus
/// [`DataType`]-driven construction of scalar values.
#[derive(Copy, Clone, Debug, Default)]
pub struct ScalarDomain<V> {
    /// [`LinearScalarDomain`] to be used by automatic differentiation transforms.
    linear_domain: LinearScalarDomain<V>,
}

impl<V> ScalarDomain<V> {
    /// Creates a new [`ScalarDomain`].
    #[inline]
    pub const fn new() -> Self {
        Self { linear_domain: LinearScalarDomain::new() }
    }

    /// Returns the [`LinearScalarDomain`] associated with this [`ScalarDomain`].
    #[inline]
    pub const fn linear_domain(&self) -> &LinearScalarDomain<V> {
        &self.linear_domain
    }
}

/// Stateless linear [`Domain`] for scalar tangent and cotangent [`Program`](crate::Program)s. This is the linear
/// compliment of [`ScalarDomain`]. They both use the same scalar type (i.e, [`DataType`]) and the same runtime
/// scalar values (i.e., `f32`, `f64`, etc.); they differ only in the operation type selected by [`Domain`]:
///
/// - [`ScalarDomain`] records ordinary scalar programs using [`ScalarOperation`].
/// - [`LinearScalarDomain`] records linear tangent and cotangent programs using [`LinearScalarOperation`].
///
/// This separate domain is needed because [`Domain::Operation`] is an associated type. Once [`ScalarDomain`]
/// says "ordinary scalar traces store [`ScalarOperation`] instructions", the same domain type cannot also say "linear
/// scalar traces store [`LinearScalarOperation`] instructions". Automatic differentiation therefore keeps a tiny
/// companion domain for linear [`Program`](crate::Program)s.
///
/// For example, tracing `f(x) = x * x` with [`ScalarDomain<f64>`] records an ordinary multiplication. Linearizing that
/// program at `x = 3.0` produces a tangent program equivalent to `δx -> 3.0 * δx + 3.0 * δx`; that tangent program is
/// stored with [`LinearScalarOperation`] instructions such as `scale` and `add`. [`LinearScalarDomain`] is what tells
/// the generic tracing machinery to use that linear operation type instead of the standard operation type.
#[derive(Copy, Clone, Debug, Default)]
pub struct LinearScalarDomain<V> {
    /// [`PhantomData`] marker that ties this zero-sized [`LinearScalarDomain`] to its scalar value type.
    marker: PhantomData<fn() -> V>,
}

impl<V> LinearScalarDomain<V> {
    /// Creates a new [`LinearScalarDomain`].
    #[inline]
    pub const fn new() -> Self {
        Self { marker: PhantomData }
    }
}

macro_rules! impl_domain_for_scalar {
    ($ty:ty) => {
        impl Domain for ScalarDomain<$ty> {
            type Type = DataType;
            type Value = $ty;
            type Constant = $ty;
            type Operation = ScalarOperation<$ty>;
        }

        impl Domain for LinearScalarDomain<$ty> {
            type Type = DataType;
            type Value = $ty;
            type Constant = $ty;
            type Operation = LinearScalarOperation<$ty>;
        }
    };
}

impl_domain_for_scalar!(bool);
impl_domain_for_scalar!(i8);
impl_domain_for_scalar!(i16);
impl_domain_for_scalar!(i32);
impl_domain_for_scalar!(i64);
impl_domain_for_scalar!(u8);
impl_domain_for_scalar!(u16);
impl_domain_for_scalar!(u32);
impl_domain_for_scalar!(u64);
impl_domain_for_scalar!(bf16);
impl_domain_for_scalar!(f16);
impl_domain_for_scalar!(f32);
impl_domain_for_scalar!(f64);

// Eager [`Context`] support is provided through the operation's interpretation, so it is available only for the
// floating-point element types whose [`ScalarOperation`]/[`LinearScalarOperation`] enums are fully interpretable. The
// integer and boolean scalar domains intentionally remain trace-only [`Domain`]s because their operation enums include
// variants (such as negation, division, and the trigonometric primitives) that those element types do not support.
// Because [`Domain::Value`] equals [`Domain::Constant`] here, these are eager [`Context`]s whose bind interprets.
impl<V: Clone + Value<DataType>> Context for ScalarDomain<V>
where
    Self: Domain<Value = V, Constant = V, Operation: InterpretableOperation<DataType, V>>,
{
    #[inline]
    fn lift(&self, constant: V) -> Result<V, ProgramError> {
        Ok(constant)
    }

    #[inline]
    fn bind(&self, operation: Self::Operation, inputs: &[Self::Value]) -> Result<Vec<Self::Value>, ProgramError> {
        operation.interpret(inputs)
    }
}

impl<V: Clone + Value<DataType>> Context for LinearScalarDomain<V>
where
    Self: Domain<Value = V, Constant = V, Operation: InterpretableOperation<DataType, V>>,
{
    #[inline]
    fn lift(&self, constant: V) -> Result<V, ProgramError> {
        Ok(constant)
    }

    #[inline]
    fn bind(&self, operation: Self::Operation, inputs: &[Self::Value]) -> Result<Vec<Self::Value>, ProgramError> {
        operation.interpret(inputs)
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::cell::RefCell;
    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::contexts::StagingContext;
    use crate::domains::AbstractDomain;
    use crate::macros::check_count;
    use crate::operations::Operation;
    use crate::operations::constants::{OneLike, SupportsOne, SupportsZero, ZeroLike};
    use crate::parameters::Placeholder;
    use crate::programs::{AtomId, ProgramBuilder};
    use crate::tracing::{Tracer, TracerState, TracingContext};
    use crate::types::{DataType, TypeError, Typed};

    use super::*;

    /// Binds a [`ZeroOperation`](crate::ZeroOperation) for `r#type` through `domain` and extracts its single output,
    /// mirroring the canonical zero-synthesis path used by interpreters and transforms.
    fn domain_zero<D: Context<Operation: SupportsZero<D::Type, D::Value>>>(
        domain: &D,
        r#type: &D::Type,
    ) -> Result<D::Value, ProgramError> {
        let mut outputs = domain.bind(SupportsZero::zero_operation(r#type.clone()), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.pop().expect("zero operation produces exactly one output"))
    }

    /// Binds a [`OneOperation`](crate::OneOperation) for `r#type` through `domain` and extracts its single output,
    /// mirroring the canonical one-synthesis path used by interpreters and transforms.
    fn domain_one<D: Context<Operation: SupportsOne<D::Type, D::Value>>>(
        domain: &D,
        r#type: &D::Type,
    ) -> Result<D::Value, ProgramError> {
        let mut outputs = domain.bind(SupportsOne::one_operation(r#type.clone()), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.pop().expect("one operation produces exactly one output"))
    }

    #[test]
    fn test_domain() {
        // Only the floating-point element types provide eager [`Context`] support, because that capability routes
        // through operation interpretation and the integer/boolean [`ScalarOperation`] enums are not interpretable. The
        // integer and boolean zero/one values themselves are covered by the [`Zero`]/[`One`] tests in
        // `crate::operations::constants`.
        let bf16_type = DataType::BF16;
        let bf16_domain = ScalarDomain::<bf16>::new();
        assert_eq!(domain_zero(&bf16_domain, &bf16_type), Ok(bf16::ZERO));
        assert_eq!(domain_one(&bf16_domain, &bf16_type), Ok(bf16::ONE));

        let f16_type = DataType::F16;
        let f16_domain = ScalarDomain::<f16>::new();
        assert_eq!(domain_zero(&f16_domain, &f16_type), Ok(f16::ZERO));
        assert_eq!(domain_one(&f16_domain, &f16_type), Ok(f16::ONE));

        let f32_type = DataType::F32;
        let f32_domain = ScalarDomain::<f32>::new();
        assert_eq!(domain_zero(&f32_domain, &f32_type), Ok(0.0f32));
        assert_eq!(domain_one(&f32_domain, &f32_type), Ok(1.0f32));

        let f64_type = DataType::F64;
        let f64_domain = ScalarDomain::<f64>::new();
        assert_eq!(domain_zero(&f64_domain, &f64_type), Ok(0.0f64));
        assert_eq!(domain_one(&f64_domain, &f64_type), Ok(1.0f64));
        assert!(matches!(
            domain_zero(&f64_domain, &DataType::F32),
            Err(ProgramError::Type(TypeError { message }))
                if message == "scalar value expected data type f64 but got f32",
        ));
        assert!(matches!(
            domain_one(&f64_domain, &DataType::F32),
            Err(ProgramError::Type(TypeError { message }))
                if message == "scalar value expected data type f64 but got f32",
        ));
    }

    #[test]
    fn test_scalar_domain() {
        // Check that [`ScalarDomain`] is zero-sized.
        assert_eq!(size_of::<ScalarDomain<bool>>(), 0);
        assert_eq!(size_of::<ScalarDomain<i8>>(), 0);
        assert_eq!(size_of::<ScalarDomain<i16>>(), 0);
        assert_eq!(size_of::<ScalarDomain<i32>>(), 0);
        assert_eq!(size_of::<ScalarDomain<i64>>(), 0);
        assert_eq!(size_of::<ScalarDomain<u8>>(), 0);
        assert_eq!(size_of::<ScalarDomain<u16>>(), 0);
        assert_eq!(size_of::<ScalarDomain<u32>>(), 0);
        assert_eq!(size_of::<ScalarDomain<u64>>(), 0);
        assert_eq!(size_of::<ScalarDomain<bf16>>(), 0);
        assert_eq!(size_of::<ScalarDomain<f16>>(), 0);
        assert_eq!(size_of::<ScalarDomain<f32>>(), 0);
        assert_eq!(size_of::<ScalarDomain<f64>>(), 0);

        // Check that `ScalarDomain` is an eager `Context` (binding interprets over concrete values).
        assert_eq!(domain_zero(&ScalarDomain::<f64>::new(), &DataType::F64), Ok(0.0));
        assert_eq!(domain_one(&ScalarDomain::<f64>::default(), &DataType::F64), Ok(1.0));
    }

    #[test]
    fn test_tracer_state_clone_debug_and_equality() {
        let live = TracerState::Live(AtomId::new(3));
        assert_eq!(live.clone(), TracerState::Live(AtomId::new(3)));
        assert_eq!(TracerState::Poison.clone(), TracerState::Poison);
        assert_ne!(live, TracerState::Poison);
        assert_eq!(format!("{live:?}"), "Live(AtomId { index: 3 })");
        assert_eq!(format!("{:?}", TracerState::Poison), "Poison");
    }

    #[test]
    fn test_tracer() {
        let domain = ScalarDomain::<f64>::new();

        // Test handles, atom lookup, cloning, typing, and rendering.
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let atom = builder.borrow_mut().add_input(DataType::F64);
        let tracing_context = TracingContext::new(&domain, builder.clone());
        let tracer = tracing_context.tracer(atom, None);
        let poisoned = Tracer::new(TracerState::Poison, DataType::F64, tracing_context.clone());
        let cloned_tracer = tracer.clone();
        assert!(std::ptr::eq(tracer.domain(), &domain));
        assert!(Rc::ptr_eq(tracer.builder(), &builder));
        assert_eq!(tracer.atom_id(), Ok(atom));
        assert_eq!(poisoned.atom_id(), Err(ProgramError::PoisonedValue));
        assert_eq!(cloned_tracer.state(), tracer.state());
        assert_eq!(cloned_tracer.r#type(), tracer.r#type());
        assert!(Rc::ptr_eq(cloned_tracer.builder(), &builder));
        assert!(matches!(tracer.r#type(), Cow::Borrowed(r#type) if *r#type == DataType::F64));
        assert_eq!(tracer.to_string(), "%0");
        assert_eq!(format!("{tracer:?}"), "Tracer { state: Live(AtomId { index: 0 }), type: F64, .. }");
        assert_eq!(poisoned.to_string(), "<poison:f64>");
        assert_eq!(format!("{poisoned:?}"), "Tracer { state: Poison, type: F64, .. }");

        // Test staging value-level identity helpers through the tracer convenience API.
        let zero = tracer.zero_like();
        let one = tracer.one_like();
        assert_eq!(zero.r#type().into_owned(), DataType::F64);
        assert_eq!(one.r#type().into_owned(), DataType::F64);
        let zero_atom = zero.atom_id().expect("zero_like output should remain live");
        let one_atom = one.atom_id().expect("one_like output should remain live");
        let program = builder
            .borrow()
            .clone()
            .build::<f64, Vec<f64>>(vec![zero_atom, one_atom], Placeholder, vec![Placeholder, Placeholder])
            .unwrap();
        assert_eq!(program.interpret(2.0), Ok(vec![0.0, 1.0]));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = zero_like %0
                    %2:f64 = one_like %0
                in (%1, %2)
            "}
            .trim_end(),
        );

        // Test staging a unary operation through the tracer convenience API.
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let atom = builder.borrow_mut().add_input(DataType::F64);
        let tracer = TracingContext::new(&domain, builder.clone()).tracer(atom, None);
        let output = tracer.unary(ScalarOperation::Neg);
        assert_eq!(output.r#type().into_owned(), DataType::F64);
        let output_atom = output.atom_id().expect("unary output should remain live");
        let program = builder.borrow().clone().build::<f64, f64>(vec![output_atom], Placeholder, Placeholder).unwrap();
        assert_eq!(program.interpret(2.0), Ok(-2.0));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = neg %0
                in (%1)
            "}
            .trim_end(),
        );

        // Test staging a binary operation through the tracer convenience API.
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let lhs_atom = builder.borrow_mut().add_input(DataType::F64);
        let rhs_atom = builder.borrow_mut().add_input(DataType::F64);
        let tracing_context = TracingContext::new(&domain, builder.clone());
        let lhs = tracing_context.tracer(lhs_atom, None);
        let rhs = tracing_context.tracer(rhs_atom, None);
        let output = lhs.binary(rhs, ScalarOperation::Add);
        assert_eq!(output.r#type().into_owned(), DataType::F64);
        let output_atom = output.atom_id().expect("binary output should remain live");
        let program = builder
            .borrow()
            .clone()
            .build::<(f64, f64), f64>(vec![output_atom], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        assert_eq!(program.interpret((2.0, 3.0)), Ok(5.0));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = add %0 %1
                in (%2)
            "}
            .trim_end(),
        );

        // Test that binary operations poison the result when inputs belong to different builders.
        let builder_a = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let builder_b = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let atom_a = builder_a.borrow_mut().add_input(DataType::F64);
        let atom_b = builder_b.borrow_mut().add_input(DataType::F64);
        let tracer_a = TracingContext::new(&domain, builder_a.clone()).tracer(atom_a, None);
        let tracer_b = TracingContext::new(&domain, builder_b).tracer(atom_b, None);
        let output = tracer_a.binary(tracer_b, ScalarOperation::Add);
        assert!(matches!(output.state(), TracerState::Poison));
        assert_eq!(output.r#type().into_owned(), DataType::F64);
        assert_eq!(builder_a.borrow().error().cloned(), Some(ProgramError::MismatchedProgramBuilders));
    }

    #[test]
    fn test_tracer_unary_records_invalid_output_count_and_returns_poisoned_tracer() {
        #[derive(Copy, Clone, Debug)]
        struct NoOutputOperation;

        impl Operation<DataType> for NoOutputOperation {
            #[inline]
            fn name(&self) -> &'static str {
                "no_output"
            }

            fn infer_output_types(&self, _input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
                Ok(Vec::new())
            }
        }

        impl InterpretableOperation<DataType, f64> for NoOutputOperation {
            #[inline]
            fn interpret(&self, _inputs: &[f64]) -> Result<Vec<f64>, ProgramError> {
                Ok(Vec::new())
            }
        }

        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, NoOutputOperation>::new()));
        let input_type = DataType::F64;
        let domain = AbstractDomain::<DataType, f64, NoOutputOperation>::new();
        let tracer = TracingContext::new(&domain, builder.clone()).input(input_type);
        let output = tracer.unary(NoOutputOperation);
        assert!(matches!(output.state(), TracerState::Poison));
        assert_eq!(output.r#type().into_owned(), DataType::F64);
        assert_eq!(builder.borrow().error().cloned(), Some(ProgramError::InvalidOutputCount { expected: 1, got: 0 }),);
    }
}
