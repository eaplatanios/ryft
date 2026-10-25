# Capabilities and Providers

Read this reference when changing value-level capabilities, providers, universe bounds, projections, dispatch,
or backend value integration. The canonical architectural explanations are `Capability` in
`crates/ryft-core/src/operations/mod.rs` and `Value` in `crates/ryft-core/src/programs/values.rs`.

## API Shape and Bounds

- For simple capability/provider impls, keep bounds to the minimum needed for the impl target to be well-formed and for
  the method body to type-check. Do not copy broader bounds from neighboring trait impls unless the provider method
  itself uses the capability.
- In capability-focused modules, prefer inlining small single-use local helper functions back into the owning trait
  impls when that makes the implementation easier to read.

## Value-Level Universes

These universe requirements apply to value-level capabilities. Context constructor capabilities such as `Zero<V>` have a
different role; inspect their existing contract instead of mechanically giving them a value-level receiver.

- Every value-level capability trait, including dynamic-shape, dimension, reference, and alignment capabilities, is
  universe-parameterized (`X<T = <Self as Capability>::Universe>: Capability + Sized`, where the universe is a coherence
  marker that need not be a `Type`, and host types are their own universe), with any other parameters (e.g.,
  `Transform`, `Binding`, or `Output`) before the universe, and every capability bundle has the same universe parameter
  with a blanket implementation.
- Never pin a capability's receiver to one universe through a super-trait such as `Value<Type = ArrayIrType>`: provided
  functions bound a type view instead (e.g., `Self: Value<Type: AsArrayType + AsDimensionType>`), implementations name
  their universe explicitly (e.g., `DynamicPad<ArrayIrType>`), and a blanket implementation that serves several
  universes introduces the universe as its own type parameter (e.g., `impl<U, V: Value<Type = U>> X<U> for V`).
- Capability traits carry the `#[capability]` attribute that checks those conventions, and declare exactly one array IR
  strategy: a projection through `#[capability(projection(ArrayIrType => ArrayType))]` (S1), lifting the operation into
  the array IR family plus an eager `ArrayIrValue` delegation (S2), an existing family-native path (S3), or member-kind
  dispatch through an operation provider (S4).
- Give capability functions default bodies only when the default is correct for every implementor (e.g., by composing
  required functions), and make implementor-dependent functions required instead.
- Only elementwise capabilities lift to `Parameterized` structures, through the `Parameterwise<P, S>` implementations
  that `define_elementwise_capability!` generates (and `impl_parameterwise_operator!` for their `std::ops` operators,
  invoked right after the corresponding `impl_tracer_operator!`), never through derive-generated code.
- In generic bounds, use each capability's default universe (or a bundle such as `ArrayOperations`) and never mix it
  with an explicitly named universe for the same value, and never pin `Typed<Type = T>` next to capability bounds at
  `T`; the trait solver cannot equate `<V as Capability>::Universe` with `<V as Typed>::Type` for generic or projected
  `V`, so mixed bounds fail to unify and pinned equalities make bounds on `V::Type` unusable.
- Never bound a universe parameter by `Type`, including in bundle blanket implementations, because host types are
  universes too, and keep compile-time assertions that host types satisfy the bundles whose members they implement.
- Every universe's implementation of a capability must honor the same documented contract, including edge cases such as
  singleton or empty inputs, so test those edge cases for each universe.
- Document the universe parameter of every trait bounded by `Capability`, including bundles and macro-generated
  capabilities, with a paragraph that starts with "The universe parameter `T` defaults to the `Capability` universe of
  the implementor, so that ..." and names the universes that actually implement it (e.g., `ArrayType` and
  `ArrayIrType`). Place it as the last paragraph before the first section heading of the trait documentation.

## Array IR Projection and Backend State

- Hand-written array IR projections (i.e., capabilities whose irregular signatures the `#[capability]` attribute cannot
  project) bound the projected member by the capability (e.g., `V::Projected: Sort<ArrayType>`) and rebuild results
  through `from_projected`. Never pin `Projected = ProjectedValue<ArrayType, V>`: that serves only array IR tracers and
  forces concrete array IR values (e.g., `ArrayIrValue<A>` or a backend's array IR value) to duplicate the forwarding.
- Backend array IR values that dispatch through a stateful rich domain (e.g., `ryft-xla`'s `XlaValue`) carry that state
  (e.g., the session) in every member and in every projected representation, because `ValueProjection::from_projected`
  is infallible and receives no state. Host-side values enter such a domain through its constants (`Context::lift`),
  never through a state-less constructor, and host-side semantics that do not depend on the state (e.g., predicate and
  reference-handle contracts) delegate to the generic `ArrayIrValue` implementations instead of being reimplemented.

## Domain Dispatch

Values have one `Value::Domain`; `Value::Dispatch` controls whether domain-binding blanket capabilities apply. Binding
blankets, including those that bind through a delegating helper, require `Value<Dispatch = ValueDomainDispatch>` and
bind through `domain()`. Blankets that only compose other capabilities must remain available to `ValueDirectDispatch`
values, as must array IR projection blankets. `ValueDirectDispatch` excludes a value only from domain-binding blankets;
it does not require every capability to be implemented directly. Do not use the removed execution/dispatch-domain split
or obsolete marker spellings from historical plans.

When changing dispatch or generic policy bounds, verify downstream direct implementations and composition as well as
built-in values. When a rule is generic over an extensible transform policy, test an independent custom policy alongside
built-in static/dynamic policies; check both the individual rule and the operation family's generated requirements. Do
not silently narrow an extensible contract to the built-in cases.
