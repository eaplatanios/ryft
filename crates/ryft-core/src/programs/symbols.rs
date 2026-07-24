use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::sync::OnceLock;

use crate::programs::types::{Type, TypeError};

/// Coherent symbolic-metadata vocabulary selected by a [`Type`] family. A [`Symbols`] implementation defines the kinds
/// of binder, expression, substitution, constraint, closed-signature identity, and concrete runtime value that belong
/// to one symbolic algebra. It is not a container or a runtime symbol table. A type family selects the vocabulary
/// through [`Type::Symbols`], after which its types, values, operations, and generic [`Program`](crate::Program)
/// machinery agree on all six associated types. The pieces participate in a program as follows:
///
///   1. Types and operation metadata carry [`Expression`](Self::Expression)s over scoped [`Variable`](Self::Variable)s.
///   2. Closing a [`Region`](crate::Region) discovers where those variables come from: formal input types,
///      compiler-managed witness inputs, or indexed ordinary-data inputs. It also retains the applicable
///      [`Constraint`](Self::Constraint)s.
///   3. Import, replay, and nested transformations use [`Substitution`](Self::Substitution)s to rebind symbolic
///      metadata without capturing unrelated scopes.
///   4. Runtime refinement can associate an authorized expression with a concrete [`Value`](Self::Value).
///   5. The closed [`SymbolSignature`] is converted to a scope-independent [`SignatureKey`](Self::SignatureKey),
///      allowing two region instantiations to be recognized as the same even when their variables have different
///      identities or names (refer to the [`SignatureKey`](Self::SignatureKey) documentation for more information on
///      this kind of _alpha-normalization_).
///
/// Type families without symbolic metadata use the dummy [`NoSymbols`] vocabulary.
///
/// # Example
///
/// [`DimensionSymbols`](crate::DimensionSymbols), the vocabulary selected by [`ArrayType`](crate::ArrayType),
/// instantiates the associated types with the dimension algebra: a [`Variable`](Self::Variable) is a symbolic array
/// extent such as `batch`, an [`Expression`](Self::Expression) is a dimension polynomial such as `2 * batch` (the kind
/// of value an array type like `f32[2 * batch]` embeds), a [`Substitution`](Self::Substitution) rewrites one scope's
/// variables into another's, and a [`Value`](Self::Value) is a concrete extent observed at run time:
///
/// ```
/// # use ryft_core::{Dimension, DimensionBindings, DimensionBounds, DimensionScope, DimensionSubstitution};
/// #
/// # fn main() -> Result<(), ryft_core::DimensionError> {
/// // Variable and Expression: `batch` is a symbolic extent, and `2 * batch` an expression over it.
/// let scope = DimensionScope::new_named(vec![(Some("batch".to_owned()), DimensionBounds::positive(None)?)]);
/// let batch = scope.variable(0)?;
/// let expression = batch.dimension().checked_mul(&Dimension::constant(2))?;
/// assert_eq!(expression.to_string(), "2 * batch");
///
/// // Substitution: Replacing `batch` with a caller's `n` rewrites `2 * batch` into `2 * n`. The result is still
/// // symbolic as nothing has been evaluated yet.
/// let caller_scope = DimensionScope::new_named(vec![(Some("n".to_owned()), DimensionBounds::positive(None)?)]);
/// let mut substitution = DimensionSubstitution::new(&scope);
/// substitution.bind(&batch, caller_scope.dimension(0)?)?;
/// assert_eq!(expression.substitute(&substitution)?.to_string(), "2 * n");
///
/// // Witness: At run time `batch` turns out to be 8 and so the expression `2 * batch` is witnessed by 16.
/// let mut bindings = DimensionBindings::new(&scope);
/// bindings.bind(&batch, 8)?;
/// assert_eq!(bindings.evaluate(&expression)?, 16);
/// # Ok(())
/// # }
/// ```
///
/// The remaining two associated types do not usually appear in user code.
/// A [`Constraint`](Self::Constraint) such as "`batch` is divisible by 4" is declared once on a scope via
/// [`DimensionScope::with_constraints`](crate::DimensionScope::with_constraints) and retained by region closure,
/// and a [`SignatureKey`](Self::SignatureKey) identity is computed internally by consistently renaming variables
/// to canonical placeholders, so that a region declared over `batch` and one declared over `n` (identical except
/// for that name) are recognized as the same instantiation and share one imported copy.
pub trait Symbols: Sized {
    /// Identity of one scoped symbolic binder. A variable identifies a symbolic unknown. It is neither an expression
    /// nor the concrete value eventually associated with that unknown. Equality should preserve whatever scope or
    /// binder identity the vocabulary needs for capture avoidance.
    type Variable: Clone + Debug + Display + PartialEq;

    /// Symbolic metadata expression composed from variables and constants. Expressions are the objects embedded in
    /// types, operation metadata, and symbolic source declarations. Depending on the vocabulary, an expression may
    /// reference no variables, one variable, or several variables.
    type Expression: Clone + Debug + Display + PartialEq;

    /// Capture-avoiding symbolic replacement of variables with expressions. Substitutions rebind metadata across scopes
    /// during program import, replay, and nested transformations. For example, a dimension substitution can replace a
    /// callee's `batch` variable with the caller expression `caller_batch + 1`. This remains symbolic; assigning the
    /// concrete runtime value `caller_batch = 7` belongs to a concrete binding/refinement environment, and not to
    /// `Substitution`.
    type Substitution: Clone + Debug;

    /// Semantic predicate over expressions retained by a closed symbolic signature. Constraints restrict the concrete
    /// assignments admitted by a scope without becoming part of structural expression equality. Dimension examples
    /// include `rows == columns`, `batch <= 64`, and `elements % 8 == 0`. The concrete vocabulary is responsible for
    /// proving constraints statically when possible and validating any residual constraints against concrete bindings.
    type Constraint: Clone + Debug + PartialEq;

    /// Hashable identity of a complete closed symbolic signature, independent of variable identities and names.
    /// The identity is *alpha-normalized*: every variable is consistently renamed to a canonical placeholder (e.g.,
    /// numbered in first-occurrence order) before the identity is computed, so neither what a variable is called
    /// nor which scope object owns it can influence the result. The name comes from the λ-calculus notion of
    /// [alpha equivalence](https://en.wikipedia.org/wiki/Lambda_calculus#%CE%B1-conversion), under which `λx. x`
    /// and `λy. y` denote the same function because they differ only in the name of the bound variable. Concretely,
    /// a signature declaring `rows` and one declaring `n` receive equal keys when, after that renaming, their binder
    /// source classes, bounds, runtime source declarations, and retained constraints also match (the key captures
    /// the complete semantics of the signature, not merely the expressions' rendered text). Program import uses this
    /// identity to decide whether a previously imported symbolic region instantiation can be reused safely.
    type SignatureKey: Clone + Debug + PartialEq + Eq + Hash;

    /// Concrete host representation obtained by evaluating a symbolic expression. A value becomes a [`SymbolWitness`]
    /// only when paired with the exact expression that it evaluates. It is distinct from an arbitrary ordinary program
    /// [`Value`](crate::Value). Symbolic refinement accepts it only through a source authorized by a
    /// [`SymbolSignature`].
    type Value: Copy + Clone + Debug + Display + PartialEq;

    // TODO(eaplatanios): Review from here onwards.

    /// Vocabulary-owned snapshot of mutable constraint state observed while deriving symbol signatures. Generic
    /// [`Program`](crate::Program) machinery treats this value as opaque. Mutable vocabularies merge per-region
    /// snapshots, freeze the referenced owners, and verify that their semantic state did not change between derivation
    /// and sealing. Vocabularies with immutable constraints use `()`.
    type ConstraintSnapshot: Clone + Debug + Default;

    /// Returns the variables referenced by `expression` in deterministic order.
    fn expression_variables(expression: &Self::Expression) -> Vec<Self::Variable>;

    /// Returns the constraints retained by the scopes that own `variables`, with duplicates removed.
    fn variable_constraints(variables: &[Self::Variable]) -> Vec<Self::Constraint>;

    /// Returns the constraints retained by `variables` together with an atomic snapshot of their mutable owners.
    ///
    /// A vocabulary with mutable constraint owners must override this method and read each owner's constraints and
    /// snapshot state atomically.
    #[inline]
    fn variable_constraints_with_snapshot(
        variables: &[Self::Variable],
    ) -> (Vec<Self::Constraint>, Self::ConstraintSnapshot) {
        (Self::variable_constraints(variables), Self::ConstraintSnapshot::default())
    }

    /// Merges per-region constraint snapshots into one arena snapshot.
    #[inline]
    fn merge_constraint_snapshots(
        snapshots: impl IntoIterator<Item = Self::ConstraintSnapshot>,
    ) -> Self::ConstraintSnapshot {
        let _ = snapshots;
        Self::ConstraintSnapshot::default()
    }

    /// Seals every mutable constraint owner referenced by `snapshot`.
    #[inline]
    fn freeze_constraint_snapshot(_snapshot: &Self::ConstraintSnapshot) {}

    /// Returns whether every owner in `snapshot` still has the semantic state observed during derivation.
    #[inline]
    fn constraint_snapshot_is_unchanged(_snapshot: &Self::ConstraintSnapshot) -> bool {
        true
    }

    /// Converts potentially overlapping rebindings into capture-avoiding two-phase substitutions.
    fn stage_rebindings(
        rebindings: &[Self::Substitution],
    ) -> Result<Vec<(Self::Substitution, Self::Substitution)>, TypeError>;

    /// Projects `signature` through `substitution`, preserving residual source classes.
    fn substitute_signature(
        signature: &SymbolSignature<Self>,
        substitution: &Self::Substitution,
    ) -> Result<SymbolSignature<Self>, TypeError>;

    /// Computes the scope-independent identity of `signature` by consistently renaming its variables to canonical
    /// placeholders. Refer to the documentation of [`SignatureKey`](Self::SignatureKey) for more information on this
    /// alpha normalization.
    fn alpha_normalized_key(signature: &SymbolSignature<Self>) -> Result<Self::SignatureKey, TypeError>;
}

/// Uninhabited symbol carrier for type families without symbolic metadata.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum NoSymbol {}

impl Display for NoSymbol {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let _ = formatter;
        match *self {}
    }
}

/// Symbol vocabulary for type families whose types carry no symbolic metadata.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct NoSymbols;

impl Symbols for NoSymbols {
    type Variable = NoSymbol;
    type Expression = NoSymbol;
    type Substitution = NoSymbol;
    type Constraint = NoSymbol;
    type SignatureKey = ();
    type Value = NoSymbol;
    type ConstraintSnapshot = ();

    #[inline]
    fn expression_variables(expression: &Self::Expression) -> Vec<Self::Variable> {
        match *expression {}
    }

    #[inline]
    fn variable_constraints(_variables: &[Self::Variable]) -> Vec<Self::Constraint> {
        Vec::new()
    }

    #[inline]
    fn stage_rebindings(
        _rebindings: &[Self::Substitution],
    ) -> Result<Vec<(Self::Substitution, Self::Substitution)>, TypeError> {
        Ok(Vec::new())
    }

    #[inline]
    fn substitute_signature(
        _signature: &SymbolSignature<Self>,
        substitution: &Self::Substitution,
    ) -> Result<SymbolSignature<Self>, TypeError> {
        match *substitution {}
    }

    #[inline]
    fn alpha_normalized_key(_signature: &SymbolSignature<Self>) -> Result<Self::SignatureKey, TypeError> {
        Ok(())
    }
}

/// Symbol variable used by the type family `T`.
pub type SymbolVariable<T> = <<T as Type>::Symbols as Symbols>::Variable;

/// Symbol expression used by the type family `T`.
pub type SymbolExpression<T> = <<T as Type>::Symbols as Symbols>::Expression;

/// Symbol substitution used by the type family `T`.
pub type SymbolSubstitution<T> = <<T as Type>::Symbols as Symbols>::Substitution;

/// Symbol constraint used by the type family `T`.
pub type SymbolConstraint<T> = <<T as Type>::Symbols as Symbols>::Constraint;

/// Alpha-normalized symbol-signature key used by the type family `T`.
pub type SymbolSignatureKey<T> = <<T as Type>::Symbols as Symbols>::SignatureKey;

/// Concrete symbol value used by the type family `T`.
pub type SymbolValue<T> = <<T as Type>::Symbols as Symbols>::Value;

/// An exact symbolic expression and its optional concrete runtime value.
///
/// A [`SymbolWitness`] associates a symbolic expression with the concrete value that establishes its runtime
/// evaluation. A value of [`None`] means that the authorized source remains staged and must be checked by the executing
/// backend.
pub struct SymbolWitness<T: Type> {
    /// Exact symbolic expression whose runtime evaluation this witness represents.
    expression: SymbolExpression<T>,

    /// Concrete runtime value, or [`None`] while its authorized source remains staged.
    value: Option<SymbolValue<T>>,
}

impl<T: Type> SymbolWitness<T> {
    /// Creates a witness associating `expression` with its optional concrete runtime `value`.
    #[inline]
    pub fn new(expression: SymbolExpression<T>, value: Option<SymbolValue<T>>) -> Self {
        Self { expression, value }
    }

    /// Returns the exact symbolic expression represented by this witness.
    #[inline]
    pub fn expression(&self) -> &SymbolExpression<T> {
        &self.expression
    }

    /// Returns this witness's concrete runtime value, or [`None`] while its source remains staged.
    #[inline]
    pub fn value(&self) -> Option<SymbolValue<T>> {
        self.value
    }
}

impl<T: Type> Clone for SymbolWitness<T> {
    fn clone(&self) -> Self {
        Self { expression: self.expression.clone(), value: self.value }
    }
}

impl<T: Type> Debug for SymbolWitness<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SymbolWitness")
            .field("expression", &self.expression)
            .field("value", &self.value)
            .finish()
    }
}

impl<T: Type> PartialEq for SymbolWitness<T> {
    fn eq(&self, other: &Self) -> bool {
        self.expression == other.expression && self.value == other.value
    }
}

/// Symbol variables and runtime sources available at one closed [`Region`](crate::Region) boundary.
pub struct SymbolSignature<S: Symbols> {
    /// Variables obtained from formal input types in deterministic first-occurrence order.
    input_variables: Vec<S::Variable>,

    /// Variables obtained from compiler-managed whole-input witnesses in deterministic first-occurrence order.
    witness_variables: Vec<S::Variable>,

    /// Variables obtained from indexed input sources in deterministic first-occurrence order.
    data_variables: Vec<S::Variable>,

    /// Exact formal region inputs whose complete values carry compiler-managed witnesses.
    witness_sources: Vec<SignatureWitnessSource<S>>,

    /// Exact formal region input/index pairs that supply indexed symbols.
    data_sources: Vec<SignatureDataSource<S>>,

    /// Semantic constraints retained from the scopes that supplied this signature's variables.
    constraints: Vec<S::Constraint>,

    /// Memoized scope-independent alpha-normalized key. This is derived metadata and does not participate in
    /// structural equality or debug formatting.
    alpha_normalized_key: OnceLock<Result<S::SignatureKey, TypeError>>,
}

impl<S: Symbols> SymbolSignature<S> {
    /// Constructs a signature from already validated source classes and constraints.
    pub(crate) fn from_parts(
        input_variables: Vec<S::Variable>,
        witness_variables: Vec<S::Variable>,
        data_variables: Vec<S::Variable>,
        witness_sources: Vec<SignatureWitnessSource<S>>,
        data_sources: Vec<SignatureDataSource<S>>,
        constraints: Vec<S::Constraint>,
    ) -> Self {
        Self {
            input_variables,
            witness_variables,
            data_variables,
            witness_sources,
            data_sources,
            constraints,
            alpha_normalized_key: OnceLock::new(),
        }
    }

    /// Returns whether this signature requires no symbolic refinement or runtime sources.
    #[inline]
    pub fn is_trivial(&self) -> bool {
        self.input_variables.is_empty()
            && self.witness_variables.is_empty()
            && self.data_variables.is_empty()
            && self.witness_sources.is_empty()
            && self.data_sources.is_empty()
            && self.constraints.is_empty()
    }

    /// Returns this signature projected through `substitution`.
    #[inline]
    pub(crate) fn substitute_symbols(&self, substitution: &S::Substitution) -> Result<Self, TypeError> {
        S::substitute_signature(self, substitution)
    }

    /// Returns variables sourced by formal region input types.
    #[inline]
    pub fn input_variables(&self) -> &[S::Variable] {
        self.input_variables.as_slice()
    }

    /// Returns variables sourced by compiler-managed whole-input witnesses.
    #[inline]
    pub fn witness_variables(&self) -> &[S::Variable] {
        self.witness_variables.as_slice()
    }

    /// Returns variables sourced by indexed formal region inputs.
    #[inline]
    pub fn data_variables(&self) -> &[S::Variable] {
        self.data_variables.as_slice()
    }

    /// Returns exact formal region inputs whose complete values carry compiler-managed witnesses.
    #[inline]
    pub fn witness_sources(&self) -> &[SignatureWitnessSource<S>] {
        self.witness_sources.as_slice()
    }

    /// Returns exact formal region input/index pairs that supply indexed symbols.
    #[inline]
    pub fn data_sources(&self) -> &[SignatureDataSource<S>] {
        self.data_sources.as_slice()
    }

    /// Returns constraints retained from the scopes that supplied this signature.
    #[inline]
    pub(crate) fn constraints(&self) -> &[S::Constraint] {
        self.constraints.as_slice()
    }

    /// Returns all variables in source-class order.
    pub fn variables(&self) -> Vec<S::Variable> {
        self.input_variables
            .iter()
            .chain(&self.witness_variables)
            .chain(&self.data_variables)
            .cloned()
            .collect()
    }

    /// Returns this signature's scope-independent alpha-normalized key.
    #[inline]
    pub fn alpha_normalized_key(&self) -> Result<S::SignatureKey, TypeError> {
        self.alpha_normalized_key.get_or_init(|| S::alpha_normalized_key(self)).clone()
    }
}

impl<S: Symbols> Clone for SymbolSignature<S> {
    fn clone(&self) -> Self {
        let alpha_normalized_key = self.alpha_normalized_key.get().cloned().map(OnceLock::from).unwrap_or_default();
        Self {
            input_variables: self.input_variables.clone(),
            witness_variables: self.witness_variables.clone(),
            data_variables: self.data_variables.clone(),
            witness_sources: self.witness_sources.clone(),
            data_sources: self.data_sources.clone(),
            constraints: self.constraints.clone(),
            alpha_normalized_key,
        }
    }
}

impl<S: Symbols> Debug for SymbolSignature<S> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SymbolSignature")
            .field("input_variables", &self.input_variables)
            .field("witness_variables", &self.witness_variables)
            .field("data_variables", &self.data_variables)
            .field("witness_sources", &self.witness_sources)
            .field("data_sources", &self.data_sources)
            .field("constraints", &self.constraints)
            .finish()
    }
}

impl<S: Symbols> PartialEq for SymbolSignature<S> {
    fn eq(&self, other: &Self) -> bool {
        self.input_variables == other.input_variables
            && self.witness_variables == other.witness_variables
            && self.data_variables == other.data_variables
            && self.witness_sources == other.witness_sources
            && self.data_sources == other.data_sources
            && self.constraints == other.constraints
    }
}

/// One formal region input whose complete value carries a compiler-managed symbolic witness expression.
pub struct SignatureWitnessSource<S: Symbols> {
    /// Index of the whole-input witness in the formal region input boundary.
    input_index: usize,

    /// Symbolic expression materialized by that input.
    expression: S::Expression,
}

impl<S: Symbols> SignatureWitnessSource<S> {
    /// Creates a formal region witness source.
    #[inline]
    pub fn new(input_index: usize, expression: S::Expression) -> Self {
        Self { input_index, expression }
    }

    /// Returns the whole-input witness's index in the formal region input boundary.
    #[inline]
    pub fn input_index(&self) -> usize {
        self.input_index
    }

    /// Returns the symbolic expression materialized by the witness.
    #[inline]
    pub fn expression(&self) -> &S::Expression {
        &self.expression
    }
}

impl<S: Symbols> Clone for SignatureWitnessSource<S> {
    fn clone(&self) -> Self {
        Self { input_index: self.input_index, expression: self.expression.clone() }
    }
}

impl<S: Symbols> Debug for SignatureWitnessSource<S> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SignatureWitnessSource")
            .field("input_index", &self.input_index)
            .field("expression", &self.expression)
            .finish()
    }
}

impl<S: Symbols> PartialEq for SignatureWitnessSource<S> {
    fn eq(&self, other: &Self) -> bool {
        self.input_index == other.input_index && self.expression == other.expression
    }
}

/// One indexed component of a formal region input carrying a symbolic expression.
pub struct SignatureDataSource<S: Symbols> {
    /// Index of the source input in the formal region input boundary.
    input_index: usize,

    /// Index within the source input that supplies the symbol.
    element_index: usize,

    /// Symbolic expression materialized by that element.
    expression: S::Expression,
}

impl<S: Symbols> SignatureDataSource<S> {
    /// Creates a formal ordinary-data symbol source.
    #[inline]
    pub fn new(input_index: usize, element_index: usize, expression: S::Expression) -> Self {
        Self { input_index, element_index, expression }
    }

    /// Returns the source input's index in the formal region input boundary.
    #[inline]
    pub fn input_index(&self) -> usize {
        self.input_index
    }

    /// Returns the index within the source input that supplies the symbol.
    #[inline]
    pub fn element_index(&self) -> usize {
        self.element_index
    }

    /// Returns the symbolic expression materialized by the source element.
    #[inline]
    pub fn expression(&self) -> &S::Expression {
        &self.expression
    }
}

impl<S: Symbols> Clone for SignatureDataSource<S> {
    fn clone(&self) -> Self {
        Self { input_index: self.input_index, element_index: self.element_index, expression: self.expression.clone() }
    }
}

impl<S: Symbols> Debug for SignatureDataSource<S> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SignatureDataSource")
            .field("input_index", &self.input_index)
            .field("element_index", &self.element_index)
            .field("expression", &self.expression)
            .finish()
    }
}

impl<S: Symbols> PartialEq for SignatureDataSource<S> {
    fn eq(&self, other: &Self) -> bool {
        self.input_index == other.input_index
            && self.element_index == other.element_index
            && self.expression == other.expression
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use pretty_assertions::assert_eq;

    use crate::{ArrayType, Dimension, TypeError};

    use super::{SymbolSignature, SymbolWitness, Symbols};

    /// Symbol vocabulary that counts alpha-normalization calls for cache-behavior tests.
    struct CountingSymbols;

    /// Number of calls to [`CountingSymbols::alpha_normalized_key`].
    static ALPHA_NORMALIZED_KEY_CALL_COUNT: AtomicUsize = AtomicUsize::new(0);

    impl Symbols for CountingSymbols {
        type Variable = usize;
        type Expression = usize;
        type Substitution = ();
        type Constraint = usize;
        type SignatureKey = usize;
        type Value = usize;
        type ConstraintSnapshot = ();

        fn expression_variables(expression: &Self::Expression) -> Vec<Self::Variable> {
            vec![*expression]
        }

        fn variable_constraints(variables: &[Self::Variable]) -> Vec<Self::Constraint> {
            variables.to_vec()
        }

        fn stage_rebindings(
            rebindings: &[Self::Substitution],
        ) -> Result<Vec<(Self::Substitution, Self::Substitution)>, TypeError> {
            Ok(vec![((), ()); rebindings.len()])
        }

        fn substitute_signature(
            signature: &SymbolSignature<Self>,
            _substitution: &Self::Substitution,
        ) -> Result<SymbolSignature<Self>, TypeError> {
            Ok(signature.clone())
        }

        fn alpha_normalized_key(signature: &SymbolSignature<Self>) -> Result<Self::SignatureKey, TypeError> {
            ALPHA_NORMALIZED_KEY_CALL_COUNT.fetch_add(1, Ordering::Relaxed);
            let key = signature.input_variables().iter().copied().sum();
            if key == usize::MAX {
                Err(TypeError::Invalid { message: "invalid counting signature".to_string() })
            } else {
                Ok(key)
            }
        }
    }

    #[test]
    fn test_symbol_witness() {
        let expression = Dimension::constant(8);
        let witness = SymbolWitness::<ArrayType>::new(expression.clone(), Some(8));

        assert_eq!(witness.expression(), &expression);
        assert_eq!(witness.value(), Some(8));
        assert_eq!(witness.clone(), witness);
        assert_eq!(format!("{witness:?}"), "SymbolWitness { expression: Dimension(8), value: Some(8) }");

        let staged = SymbolWitness::<ArrayType>::new(expression, None);
        assert_eq!(staged.value(), None);
        assert_ne!(staged, witness);
    }

    #[test]
    fn test_symbol_signature_memoizes_alpha_normalized_key_across_clones_and_errors() {
        ALPHA_NORMALIZED_KEY_CALL_COUNT.store(0, Ordering::Relaxed);
        let signature = SymbolSignature::<CountingSymbols>::from_parts(
            vec![3],
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![3],
        );
        let equivalent = SymbolSignature::<CountingSymbols>::from_parts(
            vec![3],
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![3],
        );
        let debug = format!("{signature:?}");

        assert_eq!(signature.alpha_normalized_key(), Ok(3));
        assert_eq!(signature.alpha_normalized_key(), Ok(3));
        assert_eq!(ALPHA_NORMALIZED_KEY_CALL_COUNT.load(Ordering::Relaxed), 1);
        assert_eq!(format!("{signature:?}"), debug);
        assert_eq!(signature, equivalent);

        let cloned = signature.clone();
        assert_eq!(cloned.alpha_normalized_key(), Ok(3));
        assert_eq!(ALPHA_NORMALIZED_KEY_CALL_COUNT.load(Ordering::Relaxed), 1);
        assert_eq!(equivalent.alpha_normalized_key(), Ok(3));
        assert_eq!(ALPHA_NORMALIZED_KEY_CALL_COUNT.load(Ordering::Relaxed), 2);

        let invalid = SymbolSignature::<CountingSymbols>::from_parts(
            vec![usize::MAX],
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![usize::MAX],
        );
        let expected_error = Err(TypeError::Invalid { message: "invalid counting signature".to_string() });
        assert_eq!(invalid.alpha_normalized_key(), expected_error);
        assert_eq!(
            invalid.alpha_normalized_key(),
            Err(TypeError::Invalid { message: "invalid counting signature".to_string() }),
        );
        assert_eq!(
            invalid.clone().alpha_normalized_key(),
            Err(TypeError::Invalid { message: "invalid counting signature".to_string() }),
        );
        assert_eq!(ALPHA_NORMALIZED_KEY_CALL_COUNT.load(Ordering::Relaxed), 3);
    }
}
