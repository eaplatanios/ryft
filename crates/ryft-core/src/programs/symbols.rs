use std::fmt::{Debug, Display};
use std::hash::Hash;

use crate::programs::regions::RegionSymbolSignature;
use crate::programs::types::{Type, TypeError};

/// Coherent symbolic-metadata vocabulary selected by a [`Type`] family. A [`Symbols`] implementation defines the kinds
/// of binder, expression, substitution, constraint, closed-signature identity, and concrete runtime witness that belong
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
///   4. Runtime refinement can associate an authorized expression with a concrete [`Witness`](Self::Witness).
///   5. The closed [`RegionSymbolSignature`] is converted to a scope-independent [`Signature`](Self::Signature),
///      allowing alpha-equivalent region instantiations to be recognized even when their local binder identities
///      or diagnostic names differ.
///
/// Type families without symbolic metadata use the dummy [`NoSymbols`] vocabulary.
///
/// # Example
///
/// [`DimensionSymbols`](crate::DimensionSymbols), the vocabulary selected by [`ArrayType`](crate::ArrayType),
/// instantiates the associated types with the dimension algebra: a [`Variable`](Self::Variable) is a symbolic array
/// extent such as `batch`, an [`Expression`](Self::Expression) is a dimension polynomial such as `2 * batch` (the kind
/// of value an array type like `f32[2 * batch]` embeds), a [`Substitution`](Self::Substitution) rewrites one scope's
/// variables into another's, and a [`Witness`](Self::Witness) is the concrete extent observed at run time:
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
/// The remaining two associated types do not usually appear in user code. A [`Constraint`](Self::Constraint)
/// such as "`batch` is divisible by 4" is declared once on a scope via
/// [`DimensionScope::with_constraints`](crate::DimensionScope::with_constraints) and retained by region closure,
/// and the alpha-normalized [`Signature`](Self::Signature) is computed internally so that two regions that bind,
/// say, `batch` and `n` in structurally identical ways are recognized as the same instantiation and share one
/// imported copy.
pub trait Symbols: Sized {
    // TODO(eaplatanios): Review from here onwards.
    
    /// Identity of one scoped symbolic binder.
    ///
    /// A variable identifies a symbolic unknown; it is neither an expression nor the concrete value eventually
    /// associated with that unknown. Equality should preserve whatever scope or binder identity the vocabulary needs
    /// for capture avoidance. For [`DimensionSymbols`](crate::DimensionSymbols), this is
    /// [`DimensionVariable`](crate::DimensionVariable): its scope and ordinal are semantic, its optional name is only
    /// diagnostic, and a runtime array extent such as `8usize` is not a variable.
    type Variable: Clone + Debug + Display + PartialEq;

    /// Symbolic metadata expression composed from variables and constants.
    ///
    /// Expressions are the objects embedded in types, operation metadata, and symbolic source declarations. Depending
    /// on the vocabulary, an expression may reference no variables, one variable, or several variables. For
    /// [`DimensionSymbols`](crate::DimensionSymbols), the expression type is [`Dimension`](crate::Dimension), and
    /// `2 * batch + 1` is one expression over the `batch` variable while the static extent `32` is a variable-free
    /// expression.
    type Expression: Clone + Debug + Display + PartialEq;

    /// Capture-avoiding symbolic replacement of variables with expressions.
    ///
    /// Substitutions rebind metadata across scopes during program import, replay, and nested transformations. For
    /// example, a dimension substitution can replace a callee's `batch` variable with the caller expression
    /// `caller_batch + 1`. This remains symbolic; assigning the concrete runtime value `caller_batch = 7` belongs to a
    /// concrete binding/refinement environment, not to `Substitution`.
    type Substitution: Clone + Debug;

    /// Semantic predicate over expressions retained by a closed symbolic signature.
    ///
    /// Constraints restrict the concrete assignments admitted by a scope without becoming part of structural
    /// expression equality. Dimension examples include `rows == columns`, `batch <= 64`, and `elements % 8 == 0`.
    /// The concrete vocabulary is responsible for proving constraints statically when possible and validating any
    /// residual constraints against concrete bindings.
    type Constraint: Clone + Debug + PartialEq;

    /// Hashable, scope-independent alpha-normalized identity of a complete closed symbolic signature.
    ///
    /// Two signatures may use distinct scope objects and names such as `rows` and `n` yet receive equal keys when
    /// renaming those binders makes their complete semantics identical. A key accounts for the vocabulary's relevant
    /// binder source classes, bounds, runtime source declarations, and retained constraints—not merely the expressions'
    /// rendered text. Program import uses this identity to decide whether a previously imported symbolic region
    /// instantiation can be reused safely.
    type Signature: Clone + Debug + PartialEq + Eq + Hash;

    /// Concrete host representation of a compiler-authorized runtime symbolic witness.
    ///
    /// A witness pairs this value with an exact [`Expression`](Self::Expression), allowing
    /// [`Type::refinements_with_witnesses`] to refine or validate symbolic metadata. For dimension symbols the witness
    /// value is a `usize`, so an expression such as `2 * batch + 1` might be witnessed by `17`. The `usize` is the
    /// concrete value; the expression records what it means. A witness is also distinct from an arbitrary ordinary
    /// program data value: it is supplied only through a symbolic source authorized by the closed region signature.
    type Witness: Copy + Clone + Debug + Display + PartialEq;

    /// Returns the variables referenced by `expression` in deterministic order.
    fn expression_variables(expression: &Self::Expression) -> Vec<Self::Variable>;

    /// Returns the constraints retained by the scopes that own `variables`, with duplicates removed.
    fn variable_constraints(variables: &[Self::Variable]) -> Vec<Self::Constraint>;

    /// Seals the semantic state owned by `variables` after signature validation.
    ///
    /// This hook prevents a validated program's symbolic meaning from changing after its retained signatures are
    /// derived. Vocabularies without mutable scope metadata retain the no-op behavior.
    fn freeze_variables(variables: &[Self::Variable]);

    /// Converts potentially overlapping rebindings into capture-avoiding two-phase substitutions.
    fn stage_rebindings(
        rebindings: &[Self::Substitution],
    ) -> Result<Vec<(Self::Substitution, Self::Substitution)>, TypeError>;

    /// Projects `signature` through `substitution`, preserving residual source classes.
    fn substitute_signature(
        signature: &RegionSymbolSignature<Self>,
        substitution: &Self::Substitution,
    ) -> Result<RegionSymbolSignature<Self>, TypeError>;

    /// Computes the scope-independent alpha-normalized identity of `signature`.
    fn alpha_normalized_key(signature: &RegionSymbolSignature<Self>) -> Result<Self::Signature, TypeError>;
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
    type Signature = ();
    type Witness = NoSymbol;

    #[inline]
    fn expression_variables(expression: &Self::Expression) -> Vec<Self::Variable> {
        match *expression {}
    }

    #[inline]
    fn variable_constraints(_variables: &[Self::Variable]) -> Vec<Self::Constraint> {
        Vec::new()
    }

    #[inline]
    fn freeze_variables(_variables: &[Self::Variable]) {}

    #[inline]
    fn stage_rebindings(
        _rebindings: &[Self::Substitution],
    ) -> Result<Vec<(Self::Substitution, Self::Substitution)>, TypeError> {
        Ok(Vec::new())
    }

    #[inline]
    fn substitute_signature(
        _signature: &RegionSymbolSignature<Self>,
        substitution: &Self::Substitution,
    ) -> Result<RegionSymbolSignature<Self>, TypeError> {
        match *substitution {}
    }

    #[inline]
    fn alpha_normalized_key(_signature: &RegionSymbolSignature<Self>) -> Result<Self::Signature, TypeError> {
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

/// Alpha-normalized signature used by the type family `T`.
pub type SymbolSignature<T> = <<T as Type>::Symbols as Symbols>::Signature;

/// Concrete witness used by the type family `T`.
pub type SymbolWitness<T> = <<T as Type>::Symbols as Symbols>::Witness;
