use crate::programs::instructions::InstructionId;

/// Named class of observable _effects_ that an [`Operation`](crate::Operation) can have. Effect classes exist because
/// [`Program`](crate::Program) transforms and backend lowering can have behavior conditional on those classes. For
/// example, XLA lowering threads StableHLO token chains for backend-supported ordered classes to preserve execution
/// order, mirroring [JAX's design](https://docs.jax.dev/en/latest/jep/10657-sequencing-effects.html). Ordered state
/// is instead discharged before ordinary XLA lowering.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Effect {
    /// Observable runtime assertion whose execution order relative to other
    /// [`OrderedAssertion`](Self::OrderedAssertion) effects determines which failing requirement is reported first.
    /// Operations with this effect must not be eliminated without execution unless their requirement has been proven,
    /// and the relative execution order of retained assertions must be preserved.
    OrderedAssertion,

    /// Observable input/output (e.g., printing) [`Effect`] whose execution order relative to other
    /// [`OrderedIo`](Self::OrderedIo) effects is observable (e.g., interleaved printed output). Operations with this
    /// effect must not be folded away or get eliminated, and their relative execution order must be preserved.
    OrderedIo,

    /// Observable input/output (e.g., printing) [`Effect`] whose execution order relative to other effects is not
    /// observable. Operations with this effect must not be folded away or get eliminated, but independent unordered-I/O
    /// effects may execute in any order.
    UnorderedIo,

    /// Observable access to mutable state whose execution order relative to other [`OrderedState`](Self::OrderedState)
    /// effects on the same state is observable and must be preserved. This effect _orders_ and does not gate
    /// transforms. Partial evaluation, linearization, and differentiation place stateful operations by the ordered
    /// effect frontier documented in [`partial`](crate::partial) instead of rejecting them, while dead-code elimination
    /// keeps them alive unless the operation declares itself
    /// [_removable when unused_](crate::Operation::is_removable_when_unused). Stateful operations must still be either
    /// discharged before stateless lowering or handled by a state-aware backend. References are one source of this
    /// effect, but generic consumers must not infer a particular state representation from the effect class. Keeping
    /// state distinct from I/O also prevents generic transforms from treating mutation like an external I/O effect.
    OrderedState,
}

impl Effect {
    /// All declared [`Effect`] classes, in bit order, backing [`Effects`]'s [`IntoIterator`] implementation.
    const ALL: [Effect; 4] = [Effect::OrderedAssertion, Effect::OrderedIo, Effect::UnorderedIo, Effect::OrderedState];

    /// Returns the bit representing this [`Effect`] class inside an [`Effects`] set.
    const fn bit(self) -> u8 {
        match self {
            Effect::OrderedAssertion => 1 << 0,
            Effect::OrderedIo => 1 << 1,
            Effect::UnorderedIo => 1 << 2,
            Effect::OrderedState => 1 << 3,
        }
    }

    /// Returns `true` if the execution order of this [`Effect`] class relative to other effects of the same class is
    /// observable and must be preserved.
    pub const fn is_ordered(self) -> bool {
        match self {
            Effect::OrderedState | Effect::OrderedAssertion | Effect::OrderedIo => true,
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
///     other must keep them on one side of any split they introduce. For example, XLA lowering threads StableHLO token
///     chains for the ordered classes it supports directly and rejects unresolved ordered state.
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
    pub const fn single(effect: Effect) -> Effects {
        Effects { bits: effect.bit() }
    }

    /// Returns the union of this [`Effects`] set and `other`.
    pub const fn union(self, other: Effects) -> Effects {
        Effects { bits: self.bits | other.bits }
    }

    /// Returns `true` if this [`Effects`] set is empty (i.e., if it is equal to [`Effects::PURE`]).
    pub const fn is_pure(self) -> bool {
        self.bits == 0
    }

    /// Returns `true` if this [`Effects`] set contains `effect`.
    pub const fn contains(self, effect: Effect) -> bool {
        self.bits & effect.bit() != 0
    }

    /// Returns `true` if this [`Effects`] set contains any [`Effect`] class whose execution order is observable.
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

/// [`Effect`] that is intrinsically carried by an [`Instruction`](crate::Instruction) in a [`Region`](crate::Region).
pub struct EffectOccurrence<'o, O> {
    /// Location of the [`Instruction`](crate::Instruction) that corresponds to this [`Effect`] occurrence in the source
    /// [`Region`](crate::Region) arena.
    instruction: InstructionId,

    /// [`Effect`]-carrying [`Operation`](crate::Operation).
    operation: &'o O,
}

impl<'o, O> EffectOccurrence<'o, O> {
    /// Creates a new [`EffectOccurrence`].
    #[inline]
    pub(crate) fn new(instruction: InstructionId, operation: &'o O) -> Self {
        Self { instruction, operation }
    }

    /// Returns the location of the [`Instruction`](crate::Instruction) that corresponds to this [`Effect`] occurrence
    /// in the source [`Region`](crate::Region) arena.
    #[inline]
    pub fn instruction(&self) -> InstructionId {
        self.instruction
    }

    /// Returns the [`Effect`]-carrying [`Operation`](crate::Operation).
    #[inline]
    pub fn operation(&self) -> &'o O {
        self.operation
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use pretty_assertions::assert_eq;

    use crate::programs::regions::RegionId;

    use super::*;

    #[test]
    fn test_effects() {
        // The empty set is pure, contains nothing, reports no ordering, and iterates over nothing.
        assert!(Effects::PURE.is_pure());
        assert!(!Effects::PURE.contains(Effect::OrderedAssertion));
        assert!(!Effects::PURE.contains(Effect::OrderedIo));
        assert!(!Effects::PURE.contains(Effect::UnorderedIo));
        assert!(!Effects::PURE.contains(Effect::OrderedState));
        assert!(!Effects::PURE.is_ordered());
        assert_eq!(Effects::PURE.into_iter().collect::<Vec<_>>(), Vec::<Effect>::new());

        // A singleton set contains only its effect, and every ordered class reports observable ordering.
        let assertion = Effects::single(Effect::OrderedAssertion);
        let ordered_io = Effects::single(Effect::OrderedIo);
        let unordered = Effects::single(Effect::UnorderedIo);
        let ordered_state = Effects::single(Effect::OrderedState);
        assert!(!assertion.is_pure());
        assert!(assertion.contains(Effect::OrderedAssertion));
        assert!(!assertion.contains(Effect::OrderedIo));
        assert!(assertion.is_ordered());
        assert!(!ordered_io.is_pure());
        assert!(!ordered_io.contains(Effect::OrderedAssertion));
        assert!(ordered_io.contains(Effect::OrderedIo));
        assert!(!ordered_io.contains(Effect::UnorderedIo));
        assert!(ordered_io.is_ordered());
        assert!(!unordered.is_pure());
        assert!(!unordered.is_ordered());
        assert!(!ordered_state.is_pure());
        assert!(ordered_state.contains(Effect::OrderedState));
        assert!(ordered_state.is_ordered());
        assert_eq!(assertion.into_iter().collect::<Vec<_>>(), vec![Effect::OrderedAssertion]);
        assert_eq!(ordered_io.into_iter().collect::<Vec<_>>(), vec![Effect::OrderedIo]);
        assert_eq!(unordered.into_iter().collect::<Vec<_>>(), vec![Effect::UnorderedIo]);
        assert_eq!(ordered_state.into_iter().collect::<Vec<_>>(), vec![Effect::OrderedState]);

        // Union is commutative and idempotent, `PURE` is its identity element, and the combined set contains every
        // class and iterates in declaration order.
        let all = assertion.union(ordered_io).union(unordered).union(ordered_state);
        assert_eq!(all, unordered.union(ordered_io).union(assertion).union(ordered_state));
        assert_eq!(all.union(all), all);
        assert_eq!(all.union(Effects::PURE), all);
        assert_eq!(Effects::PURE.union(assertion), assertion);
        assert!(!all.is_pure());
        assert!(all.contains(Effect::OrderedAssertion));
        assert!(all.contains(Effect::OrderedIo));
        assert!(all.contains(Effect::UnorderedIo));
        assert!(all.contains(Effect::OrderedState));
        assert!(all.is_ordered());
        assert_eq!(
            all.into_iter().collect::<Vec<_>>(),
            vec![Effect::OrderedAssertion, Effect::OrderedIo, Effect::UnorderedIo, Effect::OrderedState],
        );

        // Equality distinguishes distinct sets, self-equality holds for rebuilt sets, and hashing supports map lookups.
        assert_eq!(assertion, Effects::single(Effect::OrderedAssertion));
        assert_ne!(assertion, ordered_io);
        assert_ne!(ordered_io, unordered);
        assert_ne!(ordered_io, all);
        assert_ne!(all, Effects::PURE);
        let lookup = HashMap::from([(assertion, "assertion"), (ordered_io, "ordered I/O"), (all, "all")]);
        assert_eq!(lookup.get(&Effects::single(Effect::OrderedAssertion)), Some(&"assertion"));
        assert_eq!(lookup.get(&Effects::single(Effect::OrderedIo)), Some(&"ordered I/O"));
        assert_eq!(lookup.get(&unordered.union(ordered_io).union(assertion).union(ordered_state)), Some(&"all"));
        assert_eq!(lookup.get(&unordered), None);
    }

    #[test]
    fn test_effect_occurrence() {
        let operation = "effectful";
        let instruction = InstructionId::new(RegionId::new(2), 3);
        let occurrence = EffectOccurrence::new(instruction, &operation);
        assert_eq!(occurrence.instruction(), instruction);
        assert_eq!(occurrence.operation(), &operation);
    }
}
