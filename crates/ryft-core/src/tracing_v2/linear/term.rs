use super::*;

/// Tangent representation backed by atoms in a staged linear program.
///
/// [`LinearTerm`] is the symbolic tangent/cotangent analogue of [`Tracer`](crate::tracing_v2::Tracer).
/// When a primitive JVP rule is building a reusable linear program instead of computing a concrete
/// tangent immediately, its tangent values are instances of this type. Each term points at one atom
/// in a shared linear-program builder and stages new linear instructions as it is combined with other
/// terms.
#[derive(Clone, Parameter)]
pub struct LinearTerm<
    T: Type + Display,
    V: Traceable<T> + Parameter,
    O: Clone + Operation<T> = LinearPrimitiveOperation<ArrayType, V>,
> {
    /// Atom id representing this symbolic tangent or cotangent inside the shared linear builder.
    pub atom: AtomId,

    /// Shared builder that owns the staged linear program currently being assembled.
    pub builder: Rc<RefCell<ProgramBuilder<T, V, O>>>,
}

impl<T: Type + Display, V: Traceable<T>, O: Clone + Operation<T>> std::fmt::Debug for LinearTerm<T, V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("LinearTerm").field("atom", &self.atom).finish()
    }
}

impl<T: Type + Display, V: Traceable<T>, O: Clone + Operation<T>> LinearTerm<T, V, O> {
    /// Reconstructs a linear term from already-staged parts.
    ///
    /// This is mainly used by the linearization and transpose helpers when they need to hand
    /// primitive rules a symbolic tangent view over existing builder state.
    #[inline]
    pub fn from_staged_parts(atom: AtomId, builder: Rc<RefCell<ProgramBuilder<T, V, O>>>) -> Self {
        Self { atom, builder }
    }

    /// Stages a multi-input operation in the tangent program builder.
    ///
    /// Shape validation is performed via [`Operation::abstract_eval`]. Concrete evaluation is intentionally
    /// skipped because tangent-program outputs remain abstract until the staged linear program is
    /// replayed on concrete tangents.
    pub fn apply_staged_op(inputs: &[Self], op: O, output_count: usize) -> Result<Vec<Self>, TracingError>
    where
        O: Operation<T>,
    {
        if inputs.is_empty() {
            return Err(TracingError::EmptyParameterizedValue);
        }

        let builder = inputs[0].builder.clone();
        if inputs.iter().skip(1).any(|input| !Rc::ptr_eq(&builder, &input.builder)) {
            return Err(TracingError::InternalInvariantViolation(
                "linear tracer inputs for one staged op must share the same builder",
            ));
        }

        let input_atoms = inputs.iter().map(|input| input.atom).collect::<Vec<_>>();
        let mut borrow = builder.borrow_mut();
        let output_abstracts = op.abstract_eval(
            &input_atoms.iter().map(|id| borrow.atoms[id.index].r#type().into_owned()).collect::<Vec<_>>(),
        )?;
        let output_atoms = borrow.add_instruction_prevalidated(op, input_atoms, output_abstracts);
        drop(borrow);
        if output_atoms.len() != output_count {
            return Err(TracingError::InvalidOutputCount { expected: output_count, got: output_atoms.len() });
        }
        Ok(output_atoms.into_iter().map(|atom| Self { atom, builder: builder.clone() }).collect())
    }

    /// Stages a unary linear op in the program builder.
    ///
    /// The output atom reuses the abstract type of the input atom, which is valid for shape-preserving
    /// linear operations in tangent programs.
    #[inline]
    pub fn apply_linear_op(self, op: O) -> Self {
        let mut borrow = self.builder.borrow_mut();
        let input_atom = &borrow.atoms[self.atom.index];
        let abstract_value = input_atom.r#type().into_owned();
        let atom = borrow.add_instruction_prevalidated(op, vec![self.atom], vec![abstract_value])[0];
        drop(borrow);
        Self { atom, builder: self.builder }
    }

    /// Stages an addition of two tangent terms.
    #[inline]
    pub fn add(self, rhs: Self) -> Self
    where
        O: LinearAddOperation<T, V>,
    {
        debug_assert!(Rc::ptr_eq(&self.builder, &rhs.builder));
        let mut borrow = self.builder.borrow_mut();
        let input_atom = &borrow.atoms[self.atom.index];
        let abstract_value = input_atom.r#type().into_owned();
        let atom =
            borrow.add_instruction_prevalidated(O::linear_add_op(), vec![self.atom, rhs.atom], vec![abstract_value])[0];
        drop(borrow);
        Self { atom, builder: self.builder }
    }

    /// Stages a negation of this tangent term.
    #[inline]
    pub fn neg(self) -> Self
    where
        O: LinearNegOperation<T, V>,
    {
        self.apply_linear_op(O::linear_neg_op())
    }

    /// Stages a scaling of this tangent term by a concrete factor.
    #[inline]
    pub fn scale(self, factor: V) -> Self
    where
        O: LinearScaleOperation<T, V>,
    {
        self.apply_linear_op(O::linear_scale_op(factor))
    }
}

impl<
    T: Type + Display,
    V: Traceable<T> + ZeroLike,
    O: LinearAddOperation<T, V> + LinearNegOperation<T, V> + LinearScaleOperation<T, V> + Operation<T>,
> TangentSpace<T, V> for LinearTerm<T, V, O>
{
    #[inline]
    fn add(lhs: Self, rhs: Self) -> Self {
        debug_assert!(Rc::ptr_eq(&lhs.builder, &rhs.builder));
        let mut borrow = lhs.builder.borrow_mut();
        let input_atom = &borrow.atoms[lhs.atom.index];
        let abstract_value = input_atom.r#type().into_owned();
        let atom =
            borrow.add_instruction_prevalidated(O::linear_add_op(), vec![lhs.atom, rhs.atom], vec![abstract_value])[0];
        drop(borrow);
        Self { atom, builder: lhs.builder }
    }

    #[inline]
    fn neg(value: Self) -> Self {
        value.neg()
    }

    #[inline]
    fn scale(factor: V, tangent: Self) -> Self {
        tangent.scale(factor)
    }

    #[inline]
    fn zero_like(primal: &V, tangent: &Self) -> Self {
        let builder = tangent.builder.clone();
        let atom = builder.borrow_mut().add_constant(primal.zero_like());
        Self { atom, builder }
    }
}

/// Standard traced value used while building linear programs.
///
/// This is the default tangent payload fed into primitive JVP rules during linearization: the
/// primal component is an ordinary leaf `V`, while the tangent component is a symbolic
/// [`LinearTerm`] staged into the linear builder.
pub type Linearized<V, O = LinearPrimitiveOperation<ArrayType, V>> = JvpTracer<V, LinearTerm<ArrayType, V, O>>;
