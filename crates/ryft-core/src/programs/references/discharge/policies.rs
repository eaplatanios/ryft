use std::fmt::Debug;

// TODO(eaplatanios): Review this module.

use crate::contexts::Domain;
use crate::parameters::Parameter;
use crate::programs::ProgramError;
use crate::programs::types::Type;

use super::super::types::ReferenceType;

/// Policy naming the types and alias mechanics that one reference universe threads through reference discharge.
///
/// Discharge rewrites a program that mutates references into one that threads immutable state. Everything in that
/// rewrite that varies from one reference universe to another lives here, so the primitive rules, the context, the
/// driver, and the rule trait all name exactly one policy parameter instead of a loose collection of generics, and a
/// non-array universe is a first-class instantiation rather than an afterthought.
///
/// Implementors are zero-sized markers whose single generic implementation covers every destination
/// [`Context`](crate::Context) of their type system, following the [`BatchingPolicy`](crate::BatchingPolicy)
/// precedent. This is deliberately not a destination-context capability: a capability implemented by contexts would
/// need a coherence-foreclosing blanket implementation to achieve the same coverage.
///
/// `C` is the *destination* universe that discharge writes into, and is bounded by [`Domain`] rather than
/// [`Context`](crate::Context) so that naming this policy never obliges a caller to prove an active binding contract.
/// The alias application functions therefore reach their work through value-level capabilities on [`Domain::Value`],
/// which is what lets one implementation serve an eager destination and a staging destination alike: a staged
/// [`Tracer`](crate::Tracer) implements those capabilities by recording instructions. The destination context is
/// still passed to each of them, because a universe whose alias mechanics need context-owned state, or need to bind
/// an operation rather than call a value capability, narrows `C` to [`Context`](crate::Context) on its own
/// implementation and uses it.
///
/// This trait carries only what every reference universe can serve: allocation, reading, write-only replacement, and
/// swapping. Ordered accumulation is a separate contract, [`ReferenceAccumulationPolicy`], because its availability
/// and destination requirements genuinely vary. Splitting it keeps capability requirements at per-access granularity
/// rather than collapsing them into one implementation-level union: a universe that cannot accumulate implements
/// only this trait and still discharges every program that reads and replaces, while a program containing
/// `reference_add_update` fails to discharge for it at compile time, scoped to exactly that operation.
/// Each implementation additionally states whatever its own alias mechanics need on its `impl` block.
///
/// An implementation should leave [`Domain::Value`] generic and constrain it by the capabilities it uses, rather than
/// pinning it to one concrete value type, because a pinned policy serves exactly one destination. Pinning also
/// interacts badly with restating a capability bound: a bound on a concrete type is one Rust rejects outright unless
/// that type really does implement the capability, and a concrete backend value family cannot satisfy such a bound by
/// implementing the capability directly either, because the value-level arithmetic sugar is a blanket implementation
/// whose disjointness a downstream crate cannot prove.
pub trait ReferenceDischargePolicy<C: Domain>: Copy + Clone + Debug {
    /// Referent type system of this universe's references. A discharged reference's immutable state is a
    /// [`Domain::Value`] whose type is this universe's lift of the referent.
    type Referent: Type;

    /// Composed alias metadata carried by one flowing reference handle. This is the complete mapping from the stored
    /// value to the handle's selected coordinates, so a handle's view has exactly one source of truth during discharge.
    /// A reference family with no views uses a unit alias, whose composition and application are the identity.
    type Alias: Clone + Debug + Parameter;

    /// Returns the storage alias for a complete value with the provided referent type.
    ///
    /// Allocation and entry-boundary binding assign this alias to each new complete-value handle.
    ///
    /// This is infallible by design. Validating a referent type is type inference's job, and deriving the identity
    /// alias of an already-valid referent is total.
    fn storage_alias(referent: &Self::Referent) -> Self::Alias;

    /// Lifts a reference type into the destination type universe. This is the direction that types a reference-typed
    /// boundary position or a preserved handle in the destination program.
    fn lift_reference_type(r#type: ReferenceType<Self::Referent>) -> C::Type;

    /// Lifts a referent type into the destination type universe. A discharged reference's immutable state is an ordinary
    /// destination value of exactly this type, so this function types an entry-boundary position whose reference became
    /// state and lets a rule describe that state to the destination.
    fn lift_referent_type(referent: Self::Referent) -> C::Type;

    /// Projects a destination type back onto the reference type it denotes, or returns [`None`] when it denotes an
    /// ordinary value. Together with [`lift_reference_type`](Self::lift_reference_type) this is the conversion seam
    /// that access rules use to type-check their operands, so a non-reference operand is a classification outcome
    /// here and becomes the calling rule's own diagnostic rather than an error raised by the policy.
    fn project_reference_type(r#type: &C::Type) -> Option<ReferenceType<Self::Referent>>;

    /// Applies `alias` to one immutable state value and returns the selected value.
    ///
    /// # Parameters
    ///
    ///   - `context`: Destination context that owns `current` and any work this application performs.
    ///   - `current`: Current immutable state of the complete stored value.
    ///   - `alias`: Composed view chain selecting the coordinates to read.
    fn read(context: &C, current: &C::Value, alias: &Self::Alias) -> Result<C::Value, ProgramError>;

    /// Replaces the coordinates that `alias` selects and returns the successor state of the complete stored value
    /// without observing the previous selection.
    ///
    /// # Parameters
    ///
    ///   - `context`: Destination context that owns `current` and any work this application performs.
    ///   - `current`: Current immutable state of the complete stored value. A view implementation may consult it only to
    ///     preserve coordinates outside the selected logical handle.
    ///   - `replacement`: Value written into the selected coordinates.
    ///   - `alias`: Composed view chain selecting the coordinates to replace.
    fn write(
        context: &C,
        current: &C::Value,
        replacement: C::Value,
        alias: &Self::Alias,
    ) -> Result<C::Value, ProgramError>;

    /// Replaces the coordinates that `alias` selects and returns the previous selection followed by the successor
    /// state of the complete stored value.
    ///
    /// # Parameters
    ///
    ///   - `context`: Destination context that owns `current` and any work this application performs.
    ///   - `current`: Current immutable state of the complete stored value.
    ///   - `replacement`: Value written into the selected coordinates.
    ///   - `alias`: Composed view chain selecting the coordinates to replace.
    fn replace(
        context: &C,
        current: &C::Value,
        replacement: C::Value,
        alias: &Self::Alias,
    ) -> Result<(C::Value, C::Value), ProgramError>;
}

/// Ordered accumulation contract of a reference universe whose references support additive updates.
///
/// Accumulation is the one access mode a reference universe may be unable to serve, and the destination requirement
/// it implies is universe-specific: one universe adds through a value-level capability, another lifts an addition
/// operation into its destination operation family, and a third has no addition at all. A single requirement stated
/// on [`ReferenceDischargePolicy`] could serve none of them, so accumulation lives here instead, and each
/// implementation states its own destination requirement on its `impl` block.
///
/// The two ways accumulation can be unavailable stay distinct. A universe that cannot accumulate does not implement
/// this trait, so a program containing `reference_add_update` fails to discharge for it at compile time while reads
/// and replacements keep working through the base policy. A universe whose destination could add but whose references
/// forbid accumulation implements this trait with an explicit [`ProgramError::UnsupportedOperation`] rejection
/// instead. Closed operation-enum dispatch reintroduces the requirement for any enum whose members include an
/// accumulating operation, exactly as ordinary interpretation already does.
pub trait ReferenceAccumulationPolicy<C: Domain>: ReferenceDischargePolicy<C> {
    /// Accumulates `update` into the coordinates that `alias` selects and returns the successor state of the complete
    /// stored value.
    ///
    /// # Parameters
    ///
    ///   - `context`: Destination context that owns `current` and any work this application performs.
    ///   - `current`: Current immutable state of the complete stored value.
    ///   - `update`: Value added into the selected coordinates.
    ///   - `alias`: Composed view chain selecting the coordinates to accumulate into.
    fn accumulate(
        context: &C,
        current: &C::Value,
        update: C::Value,
        alias: &Self::Alias,
    ) -> Result<C::Value, ProgramError>;
}

#[cfg(test)]
mod tests {

    use pretty_assertions::assert_eq;

    use crate::programs::ProgramError;

    use crate::programs::references::discharge::tests::*;

    use crate::programs::references::types::ReferenceType;

    use super::*;

    #[test]
    fn test_list_reference_discharge_policy_applies_composed_aliases() {
        let destination = ListDestination::new();
        let referent = ListType { length: 4 };

        // The identity alias of an allocation covers its complete referent, and the lift/project pair round-trips a
        // reference type through the destination universe while classifying an ordinary type as not a reference.
        let storage_alias =
            <ListReferenceDischarge as ReferenceDischargePolicy<ListDestination>>::storage_alias(&referent);
        assert_eq!(storage_alias, ListAlias { offset: 0, length: 4 });
        let reference_type = ReferenceType::new(referent.clone());
        let lifted = <ListReferenceDischarge as ReferenceDischargePolicy<ListDestination>>::lift_reference_type(
            reference_type.clone(),
        );
        assert_eq!(lifted, ListIrType::Reference(reference_type.clone()));
        assert_eq!(
            <ListReferenceDischarge as ReferenceDischargePolicy<ListDestination>>::project_reference_type(&lifted),
            Some(reference_type),
        );
        assert_eq!(
            <ListReferenceDischarge as ReferenceDischargePolicy<ListDestination>>::project_reference_type(
                &ListIrType::List(referent),
            ),
            None,
        );

        // A composed alias selects only its own coordinates on every access, and replacement and accumulation return
        // the successor state of the complete stored value rather than of the selection.
        let current = ListIrValue::List(vec![1, 2, 3, 4]);
        let view = ListAlias { offset: 1, length: 2 };
        assert_eq!(ListReferenceDischarge::read(&destination, &current, &view), Ok(ListIrValue::List(vec![2, 3])));
        assert_eq!(
            ListReferenceDischarge::replace(&destination, &current, ListIrValue::List(vec![20, 30]), &view),
            Ok((ListIrValue::List(vec![2, 3]), ListIrValue::List(vec![1, 20, 30, 4]))),
        );
        assert_eq!(
            ListReferenceDischarge::accumulate(&destination, &current, ListIrValue::List(vec![20, 30]), &view),
            Ok(ListIrValue::List(vec![1, 22, 33, 4])),
        );

        // The policy reports the universe's own failures instead of silently widening a selection.
        assert_eq!(
            ListReferenceDischarge::read(&destination, &current, &ListAlias { offset: 3, length: 2 }),
            Err(ProgramError::MalformedProgram("selection [3, 5) does not fit a list of length 4".to_string())),
        );
        assert_eq!(
            ListReferenceDischarge::replace(&destination, &current, ListIrValue::List(vec![20]), &view),
            Ok((ListIrValue::List(vec![2, 3]), ListIrValue::List(vec![1, 20, 3, 4]))),
        );
    }
}
