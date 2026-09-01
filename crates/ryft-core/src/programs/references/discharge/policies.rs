use std::fmt::Debug;

use crate::contexts::Domain;
use crate::programs::ProgramError;
use crate::programs::types::Type;

/// [`Type`] capability selecting the canonical [`ReferenceDischargePolicy`] used by reference discharge entry points.
/// This is a discharge-owned extension of [`Type`] rather than part of the core type contract. It lets generic
/// [`Program`](crate::Program) functions select the reference policy of each program universe without requiring
/// callers to name that policy or relying on overlapping implementations distinguished only by the program's type
/// family.
pub trait ReferenceDischargeableType: Type {
    /// Canonical [`ReferenceDischargePolicy`] of this type universe.
    type Policy: Copy + Clone + Debug;
}

/// Trait that defines how a [`ReferenceType`](crate::ReferenceType) family reads and updates immutable state during
/// reference discharge. Reference discharge replaces mutable references with explicitly threaded immutable values.
/// This policy supplies the family-specific pieces of that rewrite like the referent type, the metadata describing
/// which part of a complete value a reference denotes, and the functions that apply that metadata. Discharge paths
/// that cross a type boundary require the destination universe to embed referent and reference types through [`From`]
/// conversions and recognize reference types through borrowed [`TryFrom`] conversions. Those are the same canonical
/// conversions reference-operation type inference uses; the policy defines no parallel conversion seam.
///
/// `C` is the destination [`Domain`] into which discharge writes. Implementations should normally remain generic over
/// compatible destination domains so that the same policy can serve eager and tracing contexts. Ordered accumulation is
/// intentionally separate in [`ReferenceAccumulationPolicy`] because not every reference family supports it.
pub trait ReferenceDischargePolicy<C: Domain> {
    /// Referent [`Type`] family of this reference family.
    type Referent: Type;

    /// Metadata describing which part of a complete stored value a [`Reference`](crate::Reference) denotes.
    /// A reference family with no views can use a unit-like alias whose application is the identity function.
    type Alias: Clone + Debug;

    /// Returns the storage alias for a complete value with the provided referent [`Type`]. Allocation and
    /// entry-boundary binding assign this alias to each new complete-value handle. This is infallible by design.
    /// Validating a referent type is type inference's job, and constructing the identity alias of an already-valid
    /// referent is total.
    fn storage_alias(referent: &Self::Referent) -> Self::Alias;

    /// Returns the value that a [`Reference`](crate::Reference) with `alias` reads from `current`. If `alias` describes
    /// a view into the stored value (e.g., a slice of an array), this function returns only that view. Otherwise, it
    /// returns the complete stored value.
    ///
    /// # Parameters
    ///
    ///   - `context`: Destination context in which any values or operations needed for the read are created.
    ///   - `current`: Complete value currently stored for the reference.
    ///   - `alias`: Describes whether the reference reads all of `current` or a view into it.
    fn read(context: &C, current: &C::Value, alias: &Self::Alias) -> Result<C::Value, ProgramError>;

    /// Returns the complete value that should be stored after a reference with `alias` writes `replacement`. If `alias`
    /// describes a view into the stored value (e.g., a slice of an array), this function replaces only that view and
    /// preserves the rest of `current`.
    ///
    /// # Parameters
    ///
    ///   - `context`: Destination context in which any values or operations needed for the write are created.
    ///   - `current`: Complete value currently stored for the reference.
    ///   - `replacement`: New value to write through the reference.
    ///   - `alias`: Describes whether the reference writes all of `current` or a view into it.
    fn write(
        context: &C,
        current: &C::Value,
        replacement: C::Value,
        alias: &Self::Alias,
    ) -> Result<C::Value, ProgramError>;

    /// Replaces the value of a reference with `replacement` and returns `(previous, updated)`. `previous` is the value
    /// read through the reference before the swap. `updated` is the complete value that should be stored afterward. If
    /// `alias` describes a view into the stored value (e.g., a slice of an array), `updated` preserves the parts of
    /// `current` outside that view.
    ///
    /// The default implementation calls [`Self::read`] followed by [`Self::write`]. Policies may override it when they
    /// can compute both results more efficiently together.
    ///
    /// # Parameters
    ///
    ///   - `context`: Destination context in which any values or operations needed for the swap are created.
    ///   - `current`: Complete value currently stored for the reference.
    ///   - `replacement`: New value to swap into the reference.
    ///   - `alias`: Describes whether the reference swaps all of `current` or a view into it.
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

/// Trait that defines additive updates for [`ReferenceType`](crate::ReferenceType) families that support them. This
/// contract is separate because accumulation is optional and its destination requirements are family-specific. A family
/// without this implementation can still discharge read, write, and swap operations. However, attempting to discharge a
/// `reference_add_update` operation fails at compile time. An implementation may instead reject selected updates with
/// [`ProgramError::UnsupportedOperation`] when support depends on the particular reference.
pub trait ReferenceAccumulationPolicy<C: Domain>: ReferenceDischargePolicy<C> {
    /// Returns the complete value that should be stored after adding `update` to a reference with `alias`. If `alias`
    /// describes a view into the stored value (e.g., a slice of an array), this function updates only that view and
    /// preserves the rest of `current`.
    ///
    /// # Parameters
    ///
    ///   - `context`: Destination context in which any values or operations needed for the update are created.
    ///   - `current`: Complete value currently stored for the reference.
    ///   - `update`: Value to add through the reference.
    ///   - `alias`: Describes whether the reference updates all of `current` or a view into it.
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
    use crate::programs::references::discharge::tests::{
        ListAlias, ListDestination, ListIrValue, ListReferenceDischarge, ListType,
    };

    use super::*;

    #[test]
    fn test_reference_discharge_policy_storage_alias_covers_the_complete_value() {
        assert_eq!(
            <ListReferenceDischarge as ReferenceDischargePolicy<ListDestination>>::storage_alias(&ListType {
                length: 4,
            }),
            ListAlias { offset: 0, length: 4 },
        );
    }

    #[test]
    fn test_reference_discharge_policy_read_returns_complete_values_and_views() {
        let destination = ListDestination::new();
        let current = ListIrValue::List(vec![1, 2, 3, 4]);

        // A storage alias reads the complete value, whereas a view alias reads only the part it describes.
        assert_eq!(
            ListReferenceDischarge::read(&destination, &current, &ListAlias { offset: 0, length: 4 }),
            Ok(ListIrValue::List(vec![1, 2, 3, 4])),
        );
        assert_eq!(
            ListReferenceDischarge::read(&destination, &current, &ListAlias { offset: 1, length: 2 }),
            Ok(ListIrValue::List(vec![2, 3])),
        );

        // An invalid view preserves the exact error produced by the reference family's selection operation.
        assert_eq!(
            ListReferenceDischarge::read(&destination, &current, &ListAlias { offset: 3, length: 2 }),
            Err(ProgramError::MalformedProgram("selection [3, 5) does not fit a list of length 4".to_string())),
        );
    }

    #[test]
    fn test_reference_discharge_policy_write_replaces_complete_values_and_views() {
        let destination = ListDestination::new();
        let current = ListIrValue::List(vec![1, 2, 3, 4]);

        assert_eq!(
            ListReferenceDischarge::write(
                &destination,
                &current,
                ListIrValue::List(vec![5, 6, 7, 8]),
                &ListAlias { offset: 0, length: 4 },
            ),
            Ok(ListIrValue::List(vec![5, 6, 7, 8])),
        );

        // Writing through a view replaces only that view and preserves the values around it.
        assert_eq!(
            ListReferenceDischarge::write(
                &destination,
                &current,
                ListIrValue::List(vec![20, 30]),
                &ListAlias { offset: 1, length: 2 },
            ),
            Ok(ListIrValue::List(vec![1, 20, 30, 4])),
        );
        assert_eq!(
            ListReferenceDischarge::write(
                &destination,
                &current,
                ListIrValue::List(vec![20, 30]),
                &ListAlias { offset: 3, length: 2 },
            ),
            Err(ProgramError::MalformedProgram("splice [3, 5) does not fit a list of length 4".to_string())),
        );
    }

    #[test]
    fn test_reference_discharge_policy_swap_returns_the_previous_view_and_complete_update() {
        let destination = ListDestination::new();
        let current = ListIrValue::List(vec![1, 2, 3, 4]);
        let view = ListAlias { offset: 1, length: 2 };

        // The default implementation returns the value read through the view first and the complete updated value
        // second.
        assert_eq!(
            ListReferenceDischarge::swap(&destination, &current, ListIrValue::List(vec![20, 30]), &view),
            Ok((ListIrValue::List(vec![2, 3]), ListIrValue::List(vec![1, 20, 30, 4]))),
        );

        // Because the default implementation reads before writing, a read failure is returned directly.
        assert_eq!(
            ListReferenceDischarge::swap(
                &destination,
                &current,
                ListIrValue::List(vec![20, 30]),
                &ListAlias { offset: 3, length: 2 },
            ),
            Err(ProgramError::MalformedProgram("selection [3, 5) does not fit a list of length 4".to_string())),
        );
    }

    #[test]
    fn test_reference_accumulation_policy_accumulate_updates_only_the_selected_view() {
        let destination = ListDestination::new();
        let current = ListIrValue::List(vec![1, 2, 3, 4]);
        let view = ListAlias { offset: 1, length: 2 };

        // Accumulation adds through the view and returns the complete value with everything outside the view intact.
        assert_eq!(
            ListReferenceDischarge::accumulate(&destination, &current, ListIrValue::List(vec![20, 30]), &view),
            Ok(ListIrValue::List(vec![1, 22, 33, 4])),
        );
        assert_eq!(
            ListReferenceDischarge::accumulate(&destination, &current, ListIrValue::List(vec![20]), &view),
            Err(ProgramError::MalformedProgram("cannot add lists of lengths 2 and 1".to_string())),
        );
    }
}
