/// Named class of observable _effects_ that an [`Operation`](crate::Operation) can have. Effect classes exist because
/// [`Program`](crate::Program) transforms and backend lowering can have behavior conditional on those classes. For
/// example, XLA lowering threads one StableHLO token chain per ordered effect class to preserve execution order,
/// mirroring [JAX's side effect sequencing design](https://docs.jax.dev/en/latest/jep/10657-sequencing-effects.html).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Effect {
    /// Observable input/output (e.g., printing) [`Effect`] whose execution order relative to other
    /// [`OrderedIo`](Self::OrderedIo) effects is observable (e.g., interleaved printed output). Operations with this
    /// effect must not be folded away or get eliminated, and their relative execution order must be preserved.
    OrderedIo,

    /// Observable input/output (e.g., printing) [`Effect`] whose execution order relative to other effects is not
    /// observable. Operations with this effect must not be folded away or get eliminated, but independent unordered-I/O
    /// effects may execute in any order.
    UnorderedIo,
}

impl Effect {
    /// All declared [`Effect`] classes, in bit order, backing [`Effects`]'s [`IntoIterator`] implementation.
    const ALL: [Effect; 2] = [Effect::OrderedIo, Effect::UnorderedIo];

    /// Returns the bit representing this [`Effect`] class inside an [`Effects`] set.
    #[inline]
    const fn bit(self) -> u8 {
        match self {
            Effect::OrderedIo => 1 << 0,
            Effect::UnorderedIo => 1 << 1,
        }
    }

    /// Returns `true` if the execution order of this [`Effect`] class relative to other effects of the same class is
    /// observable and must be preserved.
    #[inline]
    pub const fn is_ordered(self) -> bool {
        match self {
            Effect::OrderedIo => true,
            Effect::UnorderedIo => false,
        }
    }
}

/// Set of observable [`Effect`] classes of an [`Operation`](crate::Operation), describing the behaviors the operation
/// has beyond computing outputs from inputs, such as printing. [`Program`](crate::Program) transforms consult this
/// classification, instead of hardcoding operation lists, before folding, eliminating, or reordering
/// [`Instruction`](crate::Instruction)s. For example:
///
///   - Dead-code elimination ([`Program::simplified`](crate::Program::simplified) and
///     [`Program::into_simplified`](crate::Program::into_simplified)) keeps non-[pure](Self::is_pure) instructions
///     alive even when no program output consumes their results.
///   - [Ordered](Self::is_ordered) effect classes additionally promise that the relative execution order of same-class
///     instructions is preserved. Transforms that would interleave or reorder such instructions with respect to each
///     other must keep them on one side of any split they introduce, and XLA lowering threads one StableHLO token
///     chain per ordered class.
///
/// The classification of a whole [`Program`](crate::Program) is the [`union`](Self::union) of its instructions'
/// classifications and can be obtained via [`Program::effects`](crate::Program::effects). This is also what nested
/// program operations report through [`Operation::effects`](crate::Operation::effects), so that effects remain visible
/// through higher-order boundaries such as loop bodies and compiled function callees.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Effects {
    /// Bit set over [`Effect::bit`]s.
    bits: u8,
}

impl Effects {
    /// Empty [`Effects`] set meaning that the operation's outputs are a deterministic function of its inputs and it has
    /// no other observable behavior. Pure operations may be folded, eliminated, duplicated, and reordered freely.
    pub const PURE: Effects = Effects { bits: 0 };

    /// Returns the [`Effects`] set containing only `effect`.
    #[inline]
    pub const fn single(effect: Effect) -> Effects {
        Effects { bits: effect.bit() }
    }

    /// Returns the union of this [`Effects`] set and `other`.
    #[inline]
    pub const fn union(self, other: Effects) -> Effects {
        Effects { bits: self.bits | other.bits }
    }

    /// Returns `true` if this [`Effects`] set is empty (i.e., if it is equal to [`Effects::PURE`]).
    #[inline]
    pub const fn is_pure(self) -> bool {
        self.bits == 0
    }

    /// Returns `true` if this [`Effects`] set contains `effect`.
    #[inline]
    pub const fn contains(self, effect: Effect) -> bool {
        self.bits & effect.bit() != 0
    }

    /// Returns `true` if this [`Effects`] set contains any [`Effect`] class whose execution order is observable.
    #[inline]
    pub const fn is_ordered(self) -> bool {
        let mut index = 0;
        while index < Effect::ALL.len() {
            let effect = Effect::ALL[index];
            if effect.is_ordered() && self.contains(effect) {
                return true;
            }
            index += 1;
        }
        false
    }
}

/// Iterator over the [`Effect`] classes contained in an [`Effects`] set, yielded in declaration order.
pub struct EffectsIterator {
    /// [`Effects`] set whose contained effect classes are being iterated over.
    effects: Effects,

    /// Index into [`Effect::ALL`] of the next candidate effect class to consider.
    index: usize,
}

impl Iterator for EffectsIterator {
    type Item = Effect;

    #[inline]
    fn next(&mut self) -> Option<Effect> {
        while self.index < Effect::ALL.len() {
            let effect = Effect::ALL[self.index];
            self.index += 1;
            if self.effects.contains(effect) {
                return Some(effect);
            }
        }
        None
    }
}

impl IntoIterator for Effects {
    type Item = Effect;
    type IntoIter = EffectsIterator;

    fn into_iter(self) -> EffectsIterator {
        EffectsIterator { effects: self, index: 0 }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_effects() {
        // The empty set is pure, contains nothing, reports no ordering, and iterates over nothing.
        assert!(Effects::PURE.is_pure());
        assert!(!Effects::PURE.contains(Effect::OrderedIo));
        assert!(!Effects::PURE.contains(Effect::UnorderedIo));
        assert!(!Effects::PURE.is_ordered());
        assert_eq!(Effects::PURE.into_iter().collect::<Vec<_>>(), Vec::<Effect>::new());

        // A singleton set contains only its effect, and only the ordered-I/O class reports observable ordering.
        let ordered = Effects::single(Effect::OrderedIo);
        let unordered = Effects::single(Effect::UnorderedIo);
        assert!(!ordered.is_pure());
        assert!(ordered.contains(Effect::OrderedIo));
        assert!(!ordered.contains(Effect::UnorderedIo));
        assert!(ordered.is_ordered());
        assert!(!unordered.is_pure());
        assert!(!unordered.is_ordered());
        assert_eq!(ordered.into_iter().collect::<Vec<_>>(), vec![Effect::OrderedIo]);
        assert_eq!(unordered.into_iter().collect::<Vec<_>>(), vec![Effect::UnorderedIo]);

        // Union is commutative and idempotent, `PURE` is its identity element, and the combined set contains both
        // classes and iterates in declaration order.
        let both = ordered.union(unordered);
        assert_eq!(both, unordered.union(ordered));
        assert_eq!(both.union(both), both);
        assert_eq!(both.union(Effects::PURE), both);
        assert_eq!(Effects::PURE.union(ordered), ordered);
        assert!(!both.is_pure());
        assert!(both.contains(Effect::OrderedIo));
        assert!(both.contains(Effect::UnorderedIo));
        assert!(both.is_ordered());
        assert_eq!(both.into_iter().collect::<Vec<_>>(), vec![Effect::OrderedIo, Effect::UnorderedIo]);

        // Equality distinguishes distinct sets, self-equality holds for rebuilt sets, and hashing supports map lookups.
        assert_eq!(ordered, Effects::single(Effect::OrderedIo));
        assert_ne!(ordered, unordered);
        assert_ne!(ordered, both);
        assert_ne!(both, Effects::PURE);
        let lookup = HashMap::from([(ordered, "ordered"), (both, "both")]);
        assert_eq!(lookup.get(&Effects::single(Effect::OrderedIo)), Some(&"ordered"));
        assert_eq!(lookup.get(&unordered.union(ordered)), Some(&"both"));
        assert_eq!(lookup.get(&unordered), None);
    }
}
