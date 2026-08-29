use std::fmt::Debug;

use crate::contexts::Domain;
use crate::programs::ProgramError;
use crate::programs::types::Type;

/// Defines how one reference type family reads and updates immutable state during reference discharge.
///
/// Reference discharge replaces mutable references with explicitly threaded immutable values. This policy supplies
/// the family-specific pieces of that rewrite: the referent type, the metadata describing which part of a complete
/// value a reference denotes, and the functions that apply that metadata. Discharge paths that cross a type boundary
/// require the destination universe to embed referent and reference types through [`From`] and recognize reference
/// types through borrowed [`TryFrom`]. Those are the same canonical conversions reference-operation type inference
/// uses; the policy defines no parallel conversion seam.
///
/// `C` is the destination [`Domain`] into which discharge writes. Implementations should normally remain generic over
/// compatible destination domains so the same policy can serve eager and tracing contexts. Ordered accumulation is
/// intentionally separate in [`ReferenceAccumulationPolicy`] because not every reference family supports it.
pub trait ReferenceDischargePolicy<C: Domain> {
    /// Referent type system of this reference family.
    type Referent: Type;

    /// Metadata describing which part of a complete stored value a reference denotes.
    ///
    /// A reference family with no views can use a unit-like alias whose application is the identity.
    type Alias: Clone + Debug;

    /// Returns the storage alias for a complete value with the provided referent type.
    ///
    /// Allocation and entry-boundary binding assign this alias to each new complete-value handle.
    ///
    /// This is infallible by design. Validating a referent type is type inference's job, and deriving the identity
    /// alias of an already-valid referent is total.
    fn storage_alias(referent: &Self::Referent) -> Self::Alias;

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

    /// Replaces the part selected by `alias` and returns both its previous value and the complete successor state.
    ///
    /// The default implementation reads the selected value and then writes the replacement. Policies may override it
    /// when their storage representation can perform both operations more efficiently together.
    ///
    /// # Parameters
    ///
    ///   - `context`: Destination context that owns `current` and any work this application performs.
    ///   - `current`: Current immutable state of the complete stored value.
    ///   - `replacement`: Value written into the selected coordinates.
    ///   - `alias`: Composed view chain selecting the coordinates to replace.
    fn swap(
        context: &C,
        current: &C::Value,
        replacement: C::Value,
        alias: &Self::Alias,
    ) -> Result<(C::Value, C::Value), ProgramError> {
        let previous = Self::read(context, current, alias)?;
        let successor = Self::write(context, current, replacement, alias)?;
        Ok((previous, successor))
    }
}

/// Defines additive updates for reference families that support them.
///
/// This contract is separate because accumulation is optional and its destination requirements are family-specific.
/// A family without this implementation can still discharge reads, writes, and swaps; attempting to discharge a
/// `reference_add_update` operation fails at compile time. An implementation may instead reject selected updates with
/// [`ProgramError::UnsupportedOperation`] when support depends on the particular reference.
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

        // The storage alias of an allocation covers its complete referent, and the destination universe's canonical
        // conversions round-trip a reference type while rejecting an ordinary type as a reference.
        let storage_alias =
            <ListReferenceDischarge as ReferenceDischargePolicy<ListDestination>>::storage_alias(&referent);
        assert_eq!(storage_alias, ListAlias { offset: 0, length: 4 });
        let reference_type = ReferenceType::new(referent.clone());
        let lifted = ListIrType::from(reference_type.clone());
        assert_eq!(lifted, ListIrType::Reference(reference_type.clone()));
        assert_eq!(<&ReferenceType<ListType>>::try_from(&lifted), Ok(&reference_type));
        assert!(<&ReferenceType<ListType>>::try_from(&ListIrType::List(referent)).is_err());

        // A composed alias selects only its own coordinates on every access, and replacement and accumulation return
        // the successor state of the complete stored value rather than of the selection.
        let current = ListIrValue::List(vec![1, 2, 3, 4]);
        let view = ListAlias { offset: 1, length: 2 };
        assert_eq!(ListReferenceDischarge::read(&destination, &current, &view), Ok(ListIrValue::List(vec![2, 3])));
        assert_eq!(
            ListReferenceDischarge::swap(&destination, &current, ListIrValue::List(vec![20, 30]), &view),
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
            ListReferenceDischarge::swap(&destination, &current, ListIrValue::List(vec![20]), &view),
            Ok((ListIrValue::List(vec![2, 3]), ListIrValue::List(vec![1, 20, 3, 4]))),
        );
    }
}
