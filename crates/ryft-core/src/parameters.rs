//! Contains machinery for representing and working with parameterized types and type-preserving reparameterization.
//!
//! Ryft APIs accept nested Rust structures (i.e., tuples, arrays, vectors, maps, options, and derived enums and
//! structs) while compiler and transform kernels operate on flat leaf sequences. This module is the lossless bridge
//! between those two views: a parameterized value can be flattened into ordered leaves with stable paths, reduced to a
//! structure-only skeleton, and later rebuilt from that skeleton and a new leaf sequence, with every leaf type replaced
//! while its container family is preserved. The following diagram illustrates some of those relationships:
//!
//! ```text
//! ┌────────────────┐           flatten leaves          ┌────────────────┐
//! │ (x, Vec[y, z]) │ ────────────────────────────────▶ │   [x, y, z]    │
//! └────────┬───────┘                                   └────────────────┘
//!          │ extract structure
//!          ▼
//! ┌────────────────┐   rebuild with leaves [a, b, c]   ┌────────────────┐
//! │ (_, Vec[_, _]) │ ────────────────────────────────▶ │ (a, Vec[b, c]) │
//! └────────────────┘                                   └────────────────┘
//! ```
//!
//! # Entry Points
//!
//! [`Parameterized`] carries the primary API, and generic algorithms normally follow one pattern:
//!
//! 1. Save `value.parameter_structure()`.
//! 2. Consume or iterate the leaves with `into_parameters()` or `parameters()`.
//! 3. Process the flat leaves.
//! 4. Rebuild the corresponding `To<Q>` form with `from_parameters(structure, leaves)`.
//!
//! Prefer [`Parameterized::map_parameters`] when the operation is a simple one-to-one leaf mapping as it preserves
//! the structure and reports cardinality mismatches through the same contract. New leaf types implement or derive
//! [`Parameter`], and new product types derive [`Parameterized`].
//!
//! # Leaves, Containers, and Families
//!
//! [`Parameter`] marks an indivisible leaf. Primitive scalars and many core metadata types implement it directly, and
//! custom leaf types can derive it with `#[derive(Parameter)]`. The marker is what distinguishes one leaf from a
//! container whose elements are leaves without overlapping blanket implementations.
//!
//! [`Parameterized<P>`] describes a concrete structure containing leaves of type `P`. It provides:
//!
//! - ordered leaf iteration and consuming flattening,
//! - stable [`ParameterPath`] iteration for diagnostics and specialization identity,
//! - a structure-only representation using [`Placeholder`],
//! - reconstruction from a structure and a leaf iterator, and
//! - associated `To<Q>` forms that replace every `P` leaf with `Q`.
//!
//! [`ParameterizedFamily<P>`] names the container shape independently of one concrete leaf type. It is the witness that
//! the same family can be instantiated for types, constants, runtime values, or tracers, and most generic Ryft APIs use
//! the family equality carried by `Parameterized` to guarantee that those views have identical structure.
//!
//! # Reparameterization
//!
//! One function signature can appear in several synchronized forms:
//!
//! ```text
//! ┌────────────────────┬─────────────────────┐
//! │ Abstract Signature │ Input::To<Type>     │
//! ├────────────────────┼─────────────────────┤
//! │ Traced Arguments   │ Input::To<Tracer>   │
//! ├────────────────────┼─────────────────────┤
//! │ Stored Constants   │ Input::To<Constant> │
//! ├────────────────────┼─────────────────────┤
//! │ Runtime Arguments  │ Input::To<Value>    │
//! └────────────────────┴─────────────────────┘
//! ```
//!
//! Tracing replaces abstract type leaves with [`Tracer`](crate::Tracer)s. [`Program`](crate::Program)s retain input and
//! output parameter structures beside flat [`AtomId`](crate::AtomId)s. Interpretation and compiled calls validate flat
//! leaves and rebuild the original output shape, and batching, differentiation, and partial evaluation replace leaves
//! with transform-specific wrappers while preserving the caller-visible container.
//!
//! # Paths and Structures
//!
//! [`ParameterPath`] is a sequence of [`ParameterPathSegment`] values describing fields, variants, sequence indices,
//! tuple positions, or map keys from the root to a leaf. Paths make flattening observable and stable enough for error
//! messages and Just-In-Time (JIT) compilation specialization keys. They describe _location_. They do not own or
//! identify the leaf value itself.
//!
//! [`Placeholder`] represents one leaf in a structure-only value. A parameter structure preserves all container choices
//! and cardinalities needed by [`Parameterized::from_parameters`] while discarding leaf data. Reconstruction rejects
//! missing, unused, ambiguous, or structurally incompatible parameters with [`ParameterError`].
//!
//! # Extending Parameterization
//!
//! For a new indivisible leaf, implement or derive [`Parameter`]. For a new container or product type, implement
//! [`Parameterized`] together with its [`ParameterizedFamily`] witness, defining deterministic leaf order, matching
//! paths, a structure representation, reparameterized `To<Q>` forms, and exact reconstruction. Prefer the derived
//! implementations for ordinary structs and enums when available.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;

use convert_case::{Case, Casing};
use half::{bf16, f16};
use paste::paste;
use thiserror::Error;

use crate::programs::ProgramError;

/// Represents [`Parameter`]-related errors.
#[derive(Error, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ParameterError {
    #[error(
        "{}",
        match paths.as_deref() {
            None => "got more parameters than expected".to_string(),
            Some(paths) => format!(
                "unused parameter paths: {}",
                paths.iter().map(|path| format!("`{path}`")).collect::<Vec<_>>().join(", "),
            ),
        }
    )]
    UnusedParameters { paths: Option<Vec<String>> },

    #[error(
        "{}",
        match paths.as_deref() {
            None => format!("got fewer parameters than expected; expected at least {expected_count}"),
            Some(paths) => format!(
                "missing parameter paths: {}",
                paths.iter().map(|path| format!("`{path}`")).collect::<Vec<_>>().join(", "),
            ),
        }
    )]
    MissingParameters { expected_count: usize, paths: Option<Vec<String>> },

    #[error(
        "got ambiguous parameter values while combining parameterized values: {}",
        values.iter().map(|value| format!("`{value}`")).collect::<Vec<_>>().join(", "),
    )]
    AmbiguousParameterCombination { values: Vec<String> },

    #[error("mismatched parameter structures: {left_structure} and {right_structure}")]
    MismatchedParameterStructures { left_structure: String, right_structure: String },

    #[error(
        "got multiple nodes with the same parameter path; distinct map keys must have distinct `Debug` texts; \
         duplicate parameter paths: {}",
        paths.iter().map(|path| format!("`{path}`")).collect::<Vec<_>>().join(", "),
    )]
    DuplicateParameterPaths { paths: Vec<String> },
}

/// Helper trait used to encode type equality constraints in the associated type bounds of [`Parameterized`].
/// A type `X` implements [`SameAs<Y>`] only when `X` and `Y` are the exact same type.
pub trait SameAs<T> {}

impl<T> SameAs<T> for T {}

/// Marker trait for leaf parameter values in a [`Parameterized`] type. This trait is intentionally empty. A type
/// implementing [`Parameter`] is treated as an _indivisible leaf_ by [`Parameterized`] traversals. The reason we
/// need this trait in the first place is so that we can distinguish between leaf and container behavior in blanket
/// implementations. For example, `Vec<V>` implements `Parameterized<P>` when `V: Parameterized<P>`. Therefore,
/// `Vec<P>` is treated as a collection of leaf parameters because `P: Parameter` implies `P: Parameterized<P>`,
/// and not as a single leaf. Without this marker, expressing both leaf and container semantics would require
/// overlapping blanket implementations or a stable specialization feature.
///
/// Note that `ryft` provides a derive macro for this trait that you can use for custom types which need to
/// implement [`Parameter`] by tagging them with `#[derive(Parameter)]`.
pub trait Parameter {}

impl Parameter for bool {}
impl Parameter for i8 {}
impl Parameter for i16 {}
impl Parameter for i32 {}
impl Parameter for i64 {}
impl Parameter for i128 {}
impl Parameter for u8 {}
impl Parameter for u16 {}
impl Parameter for u32 {}
impl Parameter for u64 {}
impl Parameter for u128 {}
impl Parameter for bf16 {}
impl Parameter for f16 {}
impl Parameter for f32 {}
impl Parameter for f64 {}
impl Parameter for usize {}

impl<P: Parameter> Parameter for Option<P> {}

/// Placeholder [`Parameter`] type for [`Parameterized`] types that is used represent
/// [`Parameterized::parameter_structure`]. That is, it is used to replace every nested parameter in a [`Parameterized`]
/// type yielding a _structure-only_ representation that can later be used with [`Parameterized::from_parameters`] to
/// instantiate a [`Parameterized`] value with the same shape but different types of parameters.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Placeholder;

impl Display for Placeholder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "<Parameter>")
    }
}

impl Debug for Placeholder {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "<Parameter>")
    }
}

impl Parameter for Placeholder {}

/// Segment in a [`ParameterPath`]. [`Parameterized::named_parameters`], [`Parameterized::named_parameters_mut`],
/// and [`Parameterized::into_named_parameters`] produce paths that are made out of [`ParameterPathSegment`]s.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ParameterPathSegment {
    /// Field in a struct.
    Field(&'static str),

    /// Variant of an enum.
    Variant(&'static str),

    /// Index/position in a tuple.
    TupleIndex(usize),

    /// Index/position in indexable containers (e.g., arrays and [`Vec`]).
    Index(usize),

    /// [`Debug`]-formatted key of a map entry (e.g., [`HashMap`] or [`BTreeMap`]). Paths identify keys by this text
    /// only, and so the entries of two distinct keys of one map with the same [`Debug`] text have the same paths.
    /// The functions that match parameters by path (e.g., [`Parameterized::from_named_parameters`] and
    /// [`Parameterized::broadcast_to_parameter_structure`]) cannot tell such entries apart, and so they reject them
    /// with a [`ParameterError::DuplicateParameterPaths`] error instead of matching one entry in place of the other.
    Key(String),
}

/// Path to a [`Parameter`] nested inside a [`Parameterized`] type instance, composed of [`ParameterPathSegment`]s.
///
/// # Example
///
/// ```rust
/// # use ryft_core::parameters::Parameterized;
///
/// let value = vec![(4, 2)];
/// let paths = value.parameter_paths().map(|path| path.to_string()).collect::<Vec<_>>();
/// assert_eq!(paths, vec!["$[0].0", "$[0].1"]);
/// ```
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParameterPath {
    /// [`ParameterPathSegment`]s stored in leaf-to-root (i.e., reverse) order so that parent containers can append
    /// their [`ParameterPathSegment`] in `O(1)` time in functions like [`Parameterized::named_parameters`],
    /// [`Parameterized::named_parameters_mut`], and [`Parameterized::into_named_parameters`].
    segments: Vec<ParameterPathSegment>,
}

impl ParameterPath {
    /// Creates a new empty (i.e., root) [`ParameterPath`].
    pub fn root() -> Self {
        Self::default()
    }

    /// Returns the number of [`ParameterPathSegment`]s in this [`ParameterPath`].
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Returns `true` if this [`ParameterPath`] contains no [`ParameterPathSegment`]s (i.e., is the root path).
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Returns `true` if this [`ParameterPath`] is the root path (i.e., is empty).
    pub fn is_root(&self) -> bool {
        self.is_empty()
    }

    /// Returns `true` if this [`ParameterPath`] is a prefix of `other`.
    pub fn is_prefix_of(&self, other: &Self) -> bool {
        self.len() <= other.len() && self.segments().zip(other.segments()).all(|(left, right)| left == right)
    }

    /// Returns an iterator over the [`ParameterPathSegment`]s in this [`ParameterPath`] in root-to-leaf order.
    pub fn segments(&self) -> impl DoubleEndedIterator<Item = &ParameterPathSegment> + '_ {
        self.segments.iter().rev()
    }

    /// Returns a new [`ParameterPath`] with the provided [`ParameterPathSegment`] appended to it.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::parameters::ParameterPath;
    /// let path = ParameterPath::root()
    ///     .field("weights")
    ///     .index(1)
    ///     .tuple_index(0);
    /// assert_eq!(path.to_string(), "$.weights[1].0");
    /// ```
    pub fn with_segment(mut self, segment: ParameterPathSegment) -> Self {
        self.segments.insert(0, segment);
        self
    }

    /// Returns a new [`ParameterPath`] with a [`ParameterPathSegment::Field`] segment appended
    /// to the end of this [`ParameterPath`].
    pub fn field(self, name: &'static str) -> Self {
        self.with_segment(ParameterPathSegment::Field(name))
    }

    /// Returns a new [`ParameterPath`] with a [`ParameterPathSegment::Variant`] segment appended
    /// to the end of this [`ParameterPath`].
    pub fn variant(self, name: &'static str) -> Self {
        self.with_segment(ParameterPathSegment::Variant(name))
    }

    /// Returns a new [`ParameterPath`] with a [`ParameterPathSegment::Index`] segment appended
    /// to the end of this [`ParameterPath`].
    pub fn index(self, index: usize) -> Self {
        self.with_segment(ParameterPathSegment::Index(index))
    }

    /// Returns a new [`ParameterPath`] with a [`ParameterPathSegment::TupleIndex`] segment appended
    /// to the end of this [`ParameterPath`].
    pub fn tuple_index(self, index: usize) -> Self {
        self.with_segment(ParameterPathSegment::TupleIndex(index))
    }

    /// Returns a new [`ParameterPath`] with a [`ParameterPathSegment::Key`] segment appended
    /// to the end of this [`ParameterPath`].
    pub fn key<K: Debug>(self, key: K) -> Self {
        self.with_segment(ParameterPathSegment::Key(format!("{key:?}")))
    }
}

impl Display for ParameterPath {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "$")?;
        for segment in self.segments() {
            match segment {
                ParameterPathSegment::Field(name) => write!(formatter, ".{name}")?,
                ParameterPathSegment::Variant(name) => write!(formatter, ".{}", name.to_case(Case::Snake))?,
                ParameterPathSegment::TupleIndex(index) => write!(formatter, ".{index}")?,
                ParameterPathSegment::Index(index) => write!(formatter, "[{index}]")?,
                ParameterPathSegment::Key(key) => write!(formatter, "[{key}]")?,
            }
        }
        Ok(())
    }
}

impl Debug for ParameterPath {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "ParameterPath[{self}]")
    }
}

/// Type-level family that maps [`Parameter`] types nested in a type while preserving its overall
/// structure. This is used internally by [`Parameterized`] and is based on the _type family_ approach described in
/// [this blog post](https://smallcultfollowing.com/babysteps/blog/2016/11/03/associated-type-constructors-part-2-family-traits/).
/// This trait is generic over `P` (instead of using a non-generic family trait with `type To<P: Parameter>`) so
/// that each family can constrain the parameter domain at the `impl` level. For example, a family can implement
/// `ParameterizedFamily<P>` only for `P: Clone + Parameter`. With a generic associated type `To<P>` on a non-generic
/// family trait, the declaration would quantify over all `P: Parameter`, and implementations would not be allowed to
/// add stricter per-family bounds on `P`.
pub trait ParameterizedFamily<P: Parameter>: Sized {
    /// Type obtained by replacing [`Parameter`] types nested in this type with `P`.
    type To: Parameterized<P, Family = Self, ParameterStructure = <Self as ParameterizedFamily<Placeholder>>::To>
    where
        Self: ParameterizedFamily<Placeholder>;
}

/// Recursively traversable data structure that contains arbitrarily nested [`Parameter`] values of type `P`.
///
/// # What is a [`Parameterized`] type?
///
/// A [`Parameterized`] type is a container-like structure that typically contains other container-like structures
/// that eventually bottom out to [`Parameter`] values. For example, we may have a `((usize, P), HashMap<String, P>)`
/// type which contains parameters of type `P` nested inside it. [`Parameterized`] types can be thought of as tree-like
/// structures that contain [`Parameter`] values at their leaves (of type `P`). Values of such types can be viewed as
/// having two parts:
///
///   1. Their _structure_, which can be obtained via [`Self::parameter_structure`].
///   2. Their _parameter_ values, which can be obtained via [`Self::parameters`], [`Self::parameters_mut`],
///      [`Self::into_parameters`], [`Self::named_parameters`], [`Self::named_parameters_mut`],
///      and [`Self::into_named_parameters`].
///
/// In the context of machine learning (ML), a [`Parameterized`] can contain model parameters, dataset entries,
/// reinforcement learning agent observations, etc. Ryft provides built-in [`Parameterized`] implementations for a wide
/// range of container-like types, including but not limited to:
///
///   - **Parameters:** All `P: Parameter` are `Parameterized<P>`.
///   - **Tuples:** `()` is `Parameterized<P>` for all `P: Parameter`, `(V0)` is `Parameterized<P>` when
///     `V0: Parameterized<P>`, `(V0, V1)` is `Parameterized<P>` when `V0: Parameterized<P>` and `V1: Parameterized<P>`,
///     etc., up to size 12. Note that tuples with mixed [`Parameterized`] and non-[`Parameterized`] elements are
///     supported only when they appear within types for which we derive [`Parameterized`] implementations using the
///     `#[derive(Parameterized)]` macro (described in the [_Custom Parameterized Types_](#custom-parameterized-types)
///     section below).
///   - **Options:** `Option<V>` is `Parameterized<P>` when `V: Parameterized<P>`, _only_ when it appears within
///     a type for which we derive a [`Parameterized`] implementation using the `#[derive(Parameterized)]` macro.
///     That is because we need `Option<P>` to be a [`Parameter`] when `P: Parameter` in order to support some of
///     the manipulation functions that are described in the
///     [_Working with Parameterized Values_](#working-with-parameterized-values) section below.
///   - **Arrays:** `[V; N]` is `Parameterized<P>` for any `N` when `V: Parameterized<P>`.
///   - **Vectors:** `Vec<V>` is `Parameterized<P>` when `V: Parameterized<P>`.
///   - **Maps:** `HashMap<K, V>` is `Parameterized<P>` when `K: Clone + Debug + Ord` and `V: Parameterized<P>`,
///     and `BTreeMap<K, V>` is `Parameterized<P>` when `K: Clone + Debug + Ord` and `V: Parameterized<P>`.
///   - **Phantom Data:** `PhantomData<P>` is `Parameterized<P>` for all `P: Parameter`, containing no parameters.
///
/// Note that Ryft does not provide a generic `impl<P: Parameter, V: Parameterized<P>> Parameterized<P> for Box<V>`
/// because it overlaps with the blanket leaf implementation `impl<P: Parameter> Parameterized<P> for P`. Since `Box`
/// is a fundamental type, downstream crates may implement [`Parameter`] for `Box<T>` for some local type `T`, and
/// the generic `Box` implementation would then become non-coherent under Rust's orphan/coherence rules.
///
/// Ryft also provides a convenient `#[derive(Parameterized)]` macro that can be used to automatically derive
/// [`Parameterized`] implementations for custom types. Refer to the
/// [_Custom Parameterized Types_](#custom-parameterized-types) section below for information on that macro.
///
/// The [`Parameterized`] type and the functionality it provides is inspired by and shares a lot of commonalities with
/// [JAX PyTrees](https://docs.jax.dev/en/latest/pytrees.html#working-with-pytrees) and
/// [Equinox's PyTree manipulation APIs](https://docs.kidger.site/equinox/api/manipulation/).
///
/// ## Examples
///
/// The following are simple examples showing what [`Parameterized`] types are and how they are structured:
///
/// ```rust
/// # use std::collections::BTreeMap;
/// # use ryft_core::parameters::*;
///
/// // Simple tuple with 3 [`Parameter`]s.
/// let value = (1, 2, 3);
/// let parameters = value.parameters().collect::<Vec<_>>();
/// assert_eq!(value.parameter_count(), 3);
/// assert_eq!(parameters, vec![&1, &2, &3]);
/// assert_eq!(value.parameter_structure(), (Placeholder, Placeholder, Placeholder));
///
/// // Nested tuple structure with 3 [`Parameter`]s.
/// let value = (1, (2, 3), ());
/// let parameters = value.parameters().collect::<Vec<_>>();
/// assert_eq!(value.parameter_count(), 3);
/// assert_eq!(parameters, vec![&1, &2, &3]);
/// assert_eq!(value.parameter_structure(), (Placeholder, (Placeholder, Placeholder), ()));
///
/// // Nested map and tuple structure with 5 [`Parameter`]s.
/// let value = (1, BTreeMap::from([("a", vec![2]), ("b", vec![3, 4])]), (5,));
/// let parameters = value.parameters().collect::<Vec<_>>();
/// assert_eq!(value.parameter_count(), 5);
/// assert_eq!(parameters, vec![&1, &2, &3, &4, &5]);
/// assert_eq!(
///     value.parameter_structure(),
///     (Placeholder, BTreeMap::from([("a", vec![Placeholder]), ("b", vec![Placeholder, Placeholder])]), (Placeholder,)),
/// );
/// ```
///
/// Note that the value returned by [`Parameterized::parameter_structure`] is effectively a "shape template"
/// for a [`Parameterized`] value: every parameter in that value is replaced by [`Placeholder`], while values of
/// non-parameter types are preserved exactly as they are (i.e., they are cloned). This is useful because flattening,
/// which is described in more detail in the next section, discards structural boundary information, and
/// [`Parameterized::ParameterStructure`] keeps that information so functions like [`Self::from_parameters`]
/// can rebuild values that have the same structure as the original value, but different parameters.
///
/// # Working with Parameterized Values
///
/// The more interesting part about [`Parameterized`] types is what they enable us to do. The following operations are
/// fundamental to [`Parameterized`] types and are almost always involved when working with such types and values:
///
///   - **Flattening:** Given a parameterized value, _flattening_ consists of obtaining a flat iterator over the
///     parameters that are contained in that value. These parameters may be arbitrarily nested in the value, but we
///     guarantee that they will always be returned in the same order. This is crucial in enabling the next operation,
///     _unflattening_, which is the inverse of the flattening operation. This operation is exposed via the
///     [`Self::parameters`], [`Self::parameters_mut`], and [`Self::into_parameters`] functions.
///   - **Unflattening:** Given a [`Parameterized::ParameterStructure`] and an iterator over parameter values,
///     _unflattening_ consists of constructing the fully structured [`Parameterized`] value that contains those
///     parameters. The [`Parameterized::ParameterStructure`] is necessary in enabling this because _flattening_ is a
///     lossy operation for certain types. For example, consider a tuple with two `Vec<P>` elements. After flattening
///     we obtain a single iterator over parameters of type `P` without having any way of recovering the number of
///     parameters that should go in the first vector versus the second. The [`Parameterized::ParameterStructure`] is
///     used to provide that information and make recovery possible. An even more straightforward example is a value
///     contains additional non-parameter information (e.g., consider a `(P, usize)` tuple). The unflattening operation
///     is exposed via the [`Self::from_parameters`] function.
///   - **Mapping:** Given a `Parameterized<P>` value, _mapping_ consists of using a function to map all the parameters
///     nested in that value to new parameters of potentially different types yielding a `Parameterized<T>` value with
///     the same structure. This is effectively a composition of _flattening_, _mapping_ the individual parameters, and
///     _unflattening_ to obtain the final structured value. This operation is important in enabling things like tracing
///     of functions that take parameterized data structures as inputs, whereby we can map all the nested parameters to
///     tracer variables. It can also be used in a similar way to enable automatic differentiation, etc. It is a
///     fundamental component of how the core features of Ryft are implemented. This operation is exposed via the
///     [`Self::map_parameters`] function.
///
/// These core operations are also supported with _named_ parameters, where each parameter is paired with a
/// [`ParameterPath`] specifying where in the nested data structure it belongs. This is useful for things like saving
/// model checkpoints, etc. These _named_ parameter operation variants are exposed via the [`Self::named_parameters`],
/// [`Self::named_parameters_mut`], [`Self::into_named_parameters`], [`Self::from_named_parameters`],
/// [`Self::from_broadcasted_named_parameters`], [`Self::try_map_parameters`], [`Self::map_parameters`],
/// [`Self::try_map_named_parameters`], and [`Self::map_named_parameters`] functions. Relatedly,
/// [`Self::broadcast_to_parameter_structure`] interprets a value as a tree prefix of the provided structure, whose
/// leaves each cover all the nodes at and below their paths. The nodes of both trees (refer to [`Self::paths`]) are
/// matched by path, including the containers without leaves, which have no leaf paths. Every node of the prefix must
/// therefore be a node of the structure, every node of the structure must be a node of the prefix or lie at or below
/// one of its leaves, and the nodes that are matched by path must have distinct paths.
/// [`Self::from_broadcasted_named_parameters`] broadcasts the named leaves of a prefix without its containers instead,
/// and so it cannot check that the nodes of the structure without leaves are covered.
///
/// Note that, the [`Parameterized`] trait also defines a bunch of additional functions that are implemented using the
/// aforementioned core primitives like [`Self::partition_parameters`], [`Self::filter_parameters`], etc. You should
/// refer to the full list of the [`Parameterized`] trait functions and their documentation for more information.
///
/// ## Examples
///
/// The following examples show the flattening, unflattening, and mapping operations in action:
///
/// ```rust
/// # use std::collections::{BTreeMap, HashMap};
/// # use ryft_core::parameters::*;
///
/// type Value = (i32, BTreeMap<&'static str, Vec<i32>>, (i32,));
/// let value = (1, BTreeMap::from([("a", vec![2]), ("b", vec![3, 4])]), (5,));
///
/// // Flattening:
/// assert_eq!(value.parameters().copied().collect::<Vec<_>>(), vec![1, 2, 3, 4, 5]);
///
/// // Unflattening:
/// assert_eq!(
///     Value::from_parameters(value.parameter_structure(), vec![10, 20, 30, 40, 50])?,
///     (10, BTreeMap::from([("a", vec![20]), ("b", vec![30, 40])]), (50,)),
/// );
///
/// // Mapping:
/// assert_eq!(
///     value.clone().map_parameters(|parameter| (parameter as i64) * 10)?,
///     (10_i64, BTreeMap::from([("a", vec![20_i64]), ("b", vec![30_i64, 40_i64])]), (50_i64,)),
/// );
///
/// // Flattening to named parameters:
/// assert_eq!(
///     value.named_parameters().map(|(path, parameter)| (path.to_string(), *parameter)).collect::<Vec<_>>(),
///     vec![
///         ("$.0".to_string(), 1),
///         ("$.1[\"a\"][0]".to_string(), 2),
///         ("$.1[\"b\"][0]".to_string(), 3),
///         ("$.1[\"b\"][1]".to_string(), 4),
///         ("$.2.0".to_string(), 5),
///     ],
/// );
///
/// // Unflattening from named parameters:
/// assert_eq!(
///     Value::from_named_parameters(
///         value.parameter_structure(),
///         value.clone().into_named_parameters().collect::<HashMap<_, _>>(),
///     )?,
///     value,
/// );
///
/// // Mapping named parameters:
/// assert_eq!(
///     value.clone().map_named_parameters(|path, parameter| (parameter as i64) + (path.len() as i64))?,
///     (2_i64, BTreeMap::from([("a", vec![5_i64]), ("b", vec![6_i64, 7_i64])]), (7_i64,)),
/// );
///
/// // Broadcasting a tree prefix:
/// let prefix = (0, BTreeMap::from([("a", 10), ("b", 20)]), 50);
/// assert_eq!(
///     prefix.broadcast_to_parameter_structure::<Value>(value.parameter_structure())?,
///     (0, BTreeMap::from([("a", vec![10]), ("b", vec![20, 20])]), (50,)),
/// );
///
/// // Broadcasting a tree prefix whose leaves cover containers without leaves:
/// let prefix = (0, 10);
/// assert_eq!(
///     prefix.broadcast_to_parameter_structure::<(Vec<i32>, i32)>((Vec::new(), Placeholder))?,
///     (Vec::new(), 10),
/// );
///
/// // Unflattening from the broadcasted named parameters of a tree prefix:
/// let prefix = (0, BTreeMap::from([("a", 10), ("b", 20)]), 50);
/// assert_eq!(
///     Value::from_broadcasted_named_parameters(value.parameter_structure(), prefix.into_named_parameters())?,
///     (0, BTreeMap::from([("a", vec![10]), ("b", vec![20, 20])]), (50,)),
/// );
///
/// # Ok::<(), ryft_core::parameters::ParameterError>(())
/// ```
///
/// # Custom Parameterized Types
///
/// Ryft provides a `#[derive(Parameterized)]` procedural macro (via the `ryft-macros` crate) that can be used to
/// generate implementations of both [`ParameterizedFamily`] and [`Parameterized`] for custom structs and enums.
/// Concretely, the derived implementations:
///
///   - Preserve the container shape and non-parameter data.
///   - Treat every field or nested field that references the parameter type as part of the types parameters.
///   - Recursively delegate traversal to nested [`Parameterized`] field types.
///   - Preserve the variant information when traversing and rebuilding enums.
///
/// The macro has the following requirements:
///
///   - The type on which it is used must be a struct or an enum. Unions are not supported.
///   - There must be exactly one generic type bounded by [`Parameter`].
///   - The reparameterized structure type, with each nested parameter replaced by [`Placeholder`], must implement
///     [`Clone`]. Deriving [`Clone`] on the same struct or enum is usually the simplest way to satisfy this.
///   - The parameter type must be _owned_ in parameter fields (i.e., parameter references or pointers are not allowed).
///   - Nested tuples that mix parameterized and non-parameterized elements are supported inside derived structs and
///     enums. However, the same kinds of mixed tuples are not generally supported inside other generic containers
///     (e.g., `Vec<(P, usize)>`), unless those container element types already satisfy the required [`Parameterized`]
///     bounds.
///   - The generated implementation reserves internal identifiers (e.g., `'__p` and `__P`) for macro-internal
///     lifetime and type parameters. User-defined generics should avoid these names. The safest approach is not use
///     names that start with `__` but, if you do, be aware of the possibility for name conflicts due to these internal
///     identifiers.
///
/// Furthermore, it makes the following assumptions:
///
///   - All fields that reference or depend on the parameter type are treated as parameter fields.
///   - Non-parameter fields are carried through unchanged and may induce additional trait bounds in the generated
///     implementations (e.g., a [`Clone`] bound is required for implementing [`Self::parameter_structure`]).
///   - Generic parameters and their attributes are otherwise carried through as they are.
///
/// This macro also supports a container-level derive attribute, `#[ryft(crate = "...")]`, that overrides the path used
/// reference Ryft types from the generated code. This is primarily meant for deriving implementations inside wrapper
/// crates that re-export `ryft` under a different path. It should not be needed for the majority of use cases. Note
/// also that the `#[ryft(...)]` attribute is not supported on individual struct fields or enum variants.
///
/// The derived implementations only traverse and rebuild containers. To apply capabilities or arithmetic to every
/// parameter of a container (e.g., `model - gradients * learning_rate`), wrap it in a [`Parameterwise`].
///
/// ## Examples
///
/// The following examples show how to use the `#[derive(Parameterized)]` macro:
///
/// ```rust
/// # /// Test-only shim to avoid taking a (circular) dependency on `ryft`.
/// # mod ryft {
/// #     pub use ryft_core::parameters::ParameterError;
/// #     pub use ryft_core::parameters::{
/// #         Parameter, ParameterPath, ParameterPathSegment, Parameterized, ParameterizedFamily,
/// #         PathPrefixedParameterIterator, Placeholder,
/// #     };
/// #     pub use ryft_macros::Parameterized;
/// # }
/// #
/// # use ryft::*;
///
/// #[derive(Debug, Clone, PartialEq, Eq, Parameterized)]
/// struct Layer<P: Parameter> {
///     weights: Vec<P>,
///     bias: P,
///     metadata: (usize, usize),
/// }
///
/// #[derive(Debug, Clone, PartialEq, Eq, Parameterized)]
/// enum Block<P: Parameter> {
///     Identity,
///     Residual {
///         trunk: Layer<P>,
///         shortcut: (P, usize),
///         tag: &'static str,
///     },
/// }
///
/// let layer = Layer { weights: vec![1, 2, 3], bias: 4, metadata: (3, 1) };
/// assert_eq!(layer.parameter_count(), 4);
/// assert_eq!(layer.parameters().collect::<Vec<_>>(), vec![&1, &2, &3, &4]);
/// assert_eq!(
///     layer.named_parameters().map(|(path, parameter)| (path.to_string(), *parameter)).collect::<Vec<_>>(),
///     vec![
///         ("$.weights[0]".to_string(), 1),
///         ("$.weights[1]".to_string(), 2),
///         ("$.weights[2]".to_string(), 3),
///         ("$.bias".to_string(), 4),
///     ],
/// );
/// assert_eq!(
///     layer.parameter_structure(),
///     Layer {
///         weights: vec![Placeholder, Placeholder, Placeholder],
///         bias: Placeholder,
///         metadata: (3, 1),
///     },
/// );
///
/// let block = Block::Residual {
///     trunk: Layer { weights: vec![10_i32, 20, 30], bias: 40, metadata: (2, 3) },
///     shortcut: (50, 7),
///     tag: "residual",
/// };
/// assert_eq!(block.parameter_count(), 5);
/// assert_eq!(block.parameters().collect::<Vec<_>>(), vec![&10, &20, &30, &40, &50]);
/// assert_eq!(
///     block.named_parameters().map(|(path, parameter)| (path.to_string(), *parameter)).collect::<Vec<_>>(),
///     vec![
///         ("$.residual.trunk.weights[0]".to_string(), 10),
///         ("$.residual.trunk.weights[1]".to_string(), 20),
///         ("$.residual.trunk.weights[2]".to_string(), 30),
///         ("$.residual.trunk.bias".to_string(), 40),
///         ("$.residual.shortcut.0".to_string(), 50),
///     ],
/// );
/// assert_eq!(
///     block.parameter_structure(),
///     Block::Residual {
///         trunk: Layer {
///             weights: vec![Placeholder, Placeholder, Placeholder],
///             bias: Placeholder,
///             metadata: (2, 3),
///         },
///         shortcut: (Placeholder, 7),
///         tag: "residual",
///     },
/// );
/// assert_eq!(
///     Block::from_parameters(
///         block.parameter_structure(),
///         vec![1i64, 2i64, 3i64, 4i64, 5i64],
///     )?,
///     Block::Residual {
///         trunk: Layer { weights: vec![1i64, 2i64, 3i64], bias: 4i64, metadata: (2, 3) },
///         shortcut: (5i64, 7),
///         tag: "residual",
///     },
/// );
///
/// # Ok::<(), ryft_core::parameters::ParameterError>(())
/// ```
pub trait Parameterized<P: Parameter>: Sized {
    /// [`ParameterizedFamily`] that this type belongs to and which can be used to reparameterize it.
    type Family: ParameterizedFamily<P, To = Self> + ParameterizedFamily<Placeholder, To = Self::ParameterStructure>;

    /// Reparameterized form of this [`Parameterized`] type with all of its nested `P` parameter types replaced by `T`.
    /// This preserves the same [`Family`](Self::Family) and [`ParameterStructure`](Self::ParameterStructure), and is
    /// such that reparameterizing back to `P` recovers [`Self`].
    type To<T: Parameter>: Parameterized<T, Family = Self::Family, ParameterStructure = Self::ParameterStructure>
        + SameAs<<Self::Family as ParameterizedFamily<T>>::To>
    where
        Self::Family: ParameterizedFamily<T>;

    /// Type that represents a reusable shape-only representation of this [`Parameterized`] type with all nested `P`
    /// parameter types replaced by [`Placeholder`]. This must always be set to `Self::To<Placeholder>`. The only reason
    /// this is not done by default is that defaulted associated types are not supported in stable Rust, and this forces
    /// us to require that all implementations provide an implementation for this associated type as well.
    type ParameterStructure: Clone
        + Parameterized<Placeholder, Family = Self::Family, To<P> = Self>
        + SameAs<Self::To<Placeholder>>;

    /// Iterator returned by [`Self::parameters`] for a borrow of the underlying [`Parameter`]s with lifetime `'t`.
    /// This is an associated type instead of an `impl Iterator` in the corresponding function signature, so that
    /// implementations can expose and reuse a concrete iterator type. In particular, `#[derive(Parameterized)]` for
    /// enums synthesizes concrete enum iterators here, avoiding an additional heap allocation and dynamic dispatch.
    type ParameterIterator<'t, T: 't + Parameter>: 't + Iterator<Item = &'t T>
    where
        Self: 't;

    /// Iterator returned by [`Self::parameters_mut`] for a mutable borrow of the underlying [`Parameter`]s with
    /// lifetime `'t`. Similar to [`Self::ParameterIterator`], this is an associated type instead of an `impl Iterator`
    /// in the corresponding function signature, so that implementations can expose and reuse a concrete iterator type,
    /// potentially avoiding additional heap allocations and dynamic dispatch.
    type ParameterIteratorMut<'t, T: 't + Parameter>: 't + Iterator<Item = &'t mut T>
    where
        Self: 't;

    /// Iterator returned by [`Self::into_parameters`], consuming `self` and returning the underlying [`Parameter`]s.
    /// Similar to [`Self::ParameterIterator`], this is an associated type instead of an `impl Iterator` in the
    /// corresponding function signature, so that implementations can expose and reuse a concrete iterator type,
    /// potentially avoiding additional heap allocations and dynamic dispatch.
    type ParameterIntoIterator<T: Parameter>: Iterator<Item = T>;

    /// Iterator returned by [`Self::named_parameters`], borrowing the underlying [`Parameter`]s and pairing them with
    /// their corresponding [`ParameterPath`]s. Similar to [`Self::ParameterIterator`], this is an associated type
    /// instead of an `impl Iterator` in the corresponding function signature, so that implementations can expose and
    /// reuse a concrete iterator type, potentially avoiding additional heap allocations and dynamic dispatch.
    type NamedParameterIterator<'t, T: 't + Parameter>: 't + Iterator<Item = (ParameterPath, &'t T)>
    where
        Self: 't;

    /// Iterator returned by [`Self::named_parameters_mut`], mutably borrowing the underlying [`Parameter`]s and pairing
    /// them with their corresponding [`ParameterPath`]s. Similar to [`Self::ParameterIterator`], this is an associated
    /// type instead of an `impl Iterator` in the corresponding function signature, so that implementations can expose
    /// and reuse a concrete iterator type, potentially avoiding additional heap allocations and dynamic dispatch.
    type NamedParameterIteratorMut<'t, T: 't + Parameter>: 't + Iterator<Item = (ParameterPath, &'t mut T)>
    where
        Self: 't;

    /// Iterator returned by [`Self::into_named_parameters`], consuming `self` and returning the underlying
    /// [`Parameter`]s together with their corresponding [`ParameterPath`]s. Similar to [`Self::ParameterIterator`],
    /// this is an associated type instead of an `impl Iterator` in the corresponding function signature, so that
    /// implementations can expose and reuse a concrete iterator type, potentially avoiding additional heap allocations
    /// and dynamic dispatch.
    type NamedParameterIntoIterator<T: Parameter>: Iterator<Item = (ParameterPath, T)>;

    /// Returns the number of parameters in this [`Parameterized`] value.
    fn parameter_count(&self) -> usize;

    /// Returns the structure of this value by replacing all of its nested parameters with [`Placeholder`]s.
    fn parameter_structure(&self) -> Self::ParameterStructure;

    /// Returns an iterator over references to all parameters in this [`Parameterized`] value. The returned
    /// iterator traverses the parameters in the same order as [`Self::parameters_mut`], [`Self::into_parameters`],
    /// [`Self::named_parameters`], [`Self::named_parameters_mut`], and [`Self::into_named_parameters`].
    fn parameters(&self) -> Self::ParameterIterator<'_, P>;

    /// Returns an iterator over mutable references to all parameters in this [`Parameterized`] value. The returned
    /// iterator traverses the parameters in the same order as [`Self::parameters`], [`Self::into_parameters`],
    /// [`Self::named_parameters`], [`Self::named_parameters_mut`], and [`Self::into_named_parameters`].
    fn parameters_mut(&mut self) -> Self::ParameterIteratorMut<'_, P>;

    /// Consumes this [`Parameterized`] value and returns an iterator over all parameters contained in it. The
    /// returned iterator traverses the parameters in the same order as [`Self::parameters`], [`Self::parameters_mut`],
    /// [`Self::named_parameters`], [`Self::named_parameters_mut`], and [`Self::into_named_parameters`].
    fn into_parameters(self) -> Self::ParameterIntoIterator<P>;

    /// Returns an iterator over references to all parameters in this [`Parameterized`] value, paired with their
    /// corresponding [`ParameterPath`]s. The returned iterator traverses the parameters in the same order as
    /// [`Self::parameters`], [`Self::parameters_mut`], [`Self::into_parameters`], [`Self::named_parameters_mut`],
    /// and [`Self::into_named_parameters`].
    fn named_parameters(&self) -> Self::NamedParameterIterator<'_, P>;

    /// Returns an iterator over mutable references to all parameters in this [`Parameterized`] value, paired with
    /// their corresponding [`ParameterPath`]s. The returned iterator traverses the parameters in the same order as
    /// [`Self::parameters`], [`Self::parameters_mut`], [`Self::into_parameters`], [`Self::named_parameters`], and
    /// [`Self::into_named_parameters`].
    fn named_parameters_mut(&mut self) -> Self::NamedParameterIteratorMut<'_, P>;

    /// Consumes this [`Parameterized`] value and returns an iterator over all parameters contained in it, paired with
    /// their corresponding [`ParameterPath`]s. The returned iterator traverses the parameters in the same order as
    /// [`Self::parameters`], [`Self::parameters_mut`], [`Self::into_parameters`], [`Self::named_parameters`],
    /// and [`Self::named_parameters_mut`].
    fn into_named_parameters(self) -> Self::NamedParameterIntoIterator<P>;

    /// Returns an iterator over the [`ParameterPath`]s to all nested parameters in this [`Parameterized`] value.
    /// The returned iterator traverses the paths in the same order as [`Self::parameters`], [`Self::parameters_mut`],
    /// [`Self::into_parameters`], [`Self::named_parameters`], [`Self::named_parameters_mut`], and
    /// [`Self::into_named_parameters`].
    fn parameter_paths<'p>(&'p self) -> impl 'p + Iterator<Item = ParameterPath>
    where
        P: 'p,
    {
        self.named_parameters().map(|(path, _)| path)
    }

    /// Returns the [`ParameterPath`]s of all nodes of the tree that this [`Parameterized`] value represents. The value
    /// itself is the root node, and so the root path comes first. It is followed by the node paths of each child of the
    /// value, in the order in which [`Self::named_parameters`] visits the children, each extended with the segment of
    /// that child exactly as in [`Self::named_parameters`]. A leaf returns only the root path, and so does a container
    /// without children (e.g., an empty [`Vec`], an empty [`BTreeMap`], a zero-length array, the unit tuple, a
    /// [`PhantomData`], or a derived struct or enum variant without parameter fields). The leaf paths (i.e.,
    /// [`Self::parameter_paths`]) are therefore a subset of the node paths, and the node paths additionally
    /// describe the containers of the tree, including those without leaves, which have no leaf paths.
    ///
    /// The `#[derive(Parameterized)]` macro treats a struct as a node whose children are its parameter fields, and
    /// an enum as a node whose only child is its current variant, whose children are in turn its parameter fields.
    /// A derived enum thus returns the root path and the path of its variant (e.g., `$.blend` for a `Blend` variant),
    /// followed by the node paths of the fields of that variant, matching the paths of [`Self::named_parameters`].
    /// Fields that the derived implementation does not treat as parameter fields are not nodes. A nested [`Option`]
    /// field is a node whose only child, when it is [`Some`], is its value (e.g., `$.maybe` and `$.maybe.0` for a
    /// `maybe: Option<P>` field that is [`Some`], and only `$.maybe` when it is [`None`]).
    ///
    /// ```rust
    /// # use ryft_core::parameters::*;
    /// let paths = |paths: Vec<ParameterPath>| paths.iter().map(ParameterPath::to_string).collect::<Vec<_>>();
    /// assert_eq!(paths((Vec::<f32>::new(), 1f32).paths()), vec!["$", "$.0", "$.1"]);
    /// assert_eq!(paths(vec![(1, 2)].paths()), vec!["$", "$[0]", "$[0].0", "$[0].1"]);
    /// assert_eq!(paths(Parameterized::<f32>::paths(&())), vec!["$"]);
    /// ```
    fn paths(&self) -> Vec<ParameterPath>;

    /// Validates that this [`Parameterized`] value has the parameter structure whose parameter paths are
    /// `expected_paths`. Two values that are paired parameter by parameter in flat order (e.g., inputs and their
    /// per-input options) must have the same structure, because a value with as many parameters but a different
    /// structure would otherwise be paired silently with the wrong parameters.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError::MismatchedParameterStructures`] when the parameter paths of this value differ from
    /// `expected_paths`.
    fn validate_parameter_structure(&self, expected_paths: &[ParameterPath]) -> Result<(), ParameterError> {
        let paths = self.parameter_paths().collect::<Vec<_>>();
        if paths != expected_paths {
            return Err(ParameterError::MismatchedParameterStructures {
                left_structure: format!(
                    "[{}]",
                    expected_paths.iter().map(ParameterPath::to_string).collect::<Vec<_>>().join(", "),
                ),
                right_structure: format!(
                    "[{}]",
                    paths.iter().map(ParameterPath::to_string).collect::<Vec<_>>().join(", "),
                ),
            });
        }
        Ok(())
    }

    /// Reconstructs a value of this [`Parameterized`] type having the provided `structure` and consuming values
    /// from the provided `parameters` to populate its parameters. This function may not consume all the provided
    /// parameters, but if there are not enough parameters in the provided iterator, it will return a
    /// [`ParameterError::MissingParameters`] error.
    fn from_parameters_with_remainder<I: Iterator<Item = P>>(
        structure: Self::ParameterStructure,
        parameters: &mut I,
    ) -> Result<Self, ParameterError>;

    /// Reconstructs a value of this [`Parameterized`] type having the provided `structure` and consuming values
    /// from the provided `parameters` to populate its parameters. This function expects to fully consume the provided
    /// iterator. If it does not contain enough values, then it will return a [`ParameterError::MissingParameters`]
    /// error, while if it contains too many values, it will return an [`ParameterError::UnusedParameters`]. If you do
    /// not want to fully consume the provided iterator, then you must use [`Self::from_parameters_with_remainder`]
    /// instead.
    fn from_parameters<I: IntoIterator<Item = P>>(
        structure: Self::ParameterStructure,
        parameters: I,
    ) -> Result<Self, ParameterError> {
        let mut parameters = parameters.into_iter();
        let parameterized = Self::from_parameters_with_remainder(structure, &mut parameters)?;
        parameters
            .next()
            .map(|_| Err(ParameterError::UnusedParameters { paths: None }))
            .unwrap_or_else(|| Ok(parameterized))
    }

    /// Reconstructs a value of this [`Parameterized`] type having the provided `structure` and consuming named values
    /// from the provided `parameters` to populate its parameters. Unlike [`Self::from_broadcasted_named_parameters`],
    /// this function is strict in that keys in `parameters` must match exactly leaf [`ParameterPath`]s in `structure`,
    /// and path prefix matching is not being used. If two leaves of `structure` have the same path, or two entries of
    /// `parameters` do, then this function will return a [`ParameterError::DuplicateParameterPaths`] error, because
    /// the paths cannot tell them apart (e.g., the values of two map keys with the same [`Debug`] text, refer to
    /// [`ParameterPathSegment::Key`]). If there are missing parameters preventing reconstruction from being feasible,
    /// then this function will return a [`ParameterError::MissingParameters`] error. Furthermore, if extra paths
    /// remain after reconstruction, then it will return an [`ParameterError::UnusedParameters`] error. For fully
    /// worked examples, refer to the examples provided in the top-level documentation of the [`Parameterized`] trait.
    fn from_named_parameters<I: IntoIterator<Item = (ParameterPath, P)>>(
        structure: Self::ParameterStructure,
        parameters: I,
    ) -> Result<Self, ParameterError> {
        // Collects the provided parameters that remain into a hash map for keyed lookups, and records the paths of
        // those that repeat the path of a matched parameter or of another remaining parameter.
        fn defer<P>(
            parameters: impl Iterator<Item = (ParameterPath, P)>,
            matched_paths: &[ParameterPath],
            duplicate_paths: &mut Vec<String>,
        ) -> HashMap<ParameterPath, P> {
            let matched_paths = matched_paths.iter().collect::<HashSet<_>>();
            let mut deferred = HashMap::with_capacity(parameters.size_hint().0);
            for (path, parameter) in parameters {
                if matched_paths.contains(&path) || deferred.contains_key(&path) {
                    duplicate_paths.push(path.to_string());
                } else {
                    deferred.insert(path, parameter);
                }
            }
            deferred
        }

        // Two leaves of `structure` with the same path would receive the provided parameters in provided order.
        let expected_paths = structure.parameter_paths().collect::<Vec<_>>();
        let mut duplicate_paths = find_duplicate_paths(&expected_paths);
        if !duplicate_paths.is_empty() {
            return Err(ParameterError::DuplicateParameterPaths { paths: duplicate_paths });
        }

        // We try to consume parameters in lockstep with the leaf paths of `structure` first, so that the matched
        // paths are always the first `values.len()` expected paths. If we encounter any out-of-order parameters,
        // we lazily materialize the remaining entries into a hash map and continue with keyed lookups.
        let expected_count = expected_paths.len();
        let mut values = Vec::with_capacity(expected_count);
        let mut missing_paths = Vec::new();
        let mut deferred_parameters = None::<HashMap<ParameterPath, P>>;
        let mut parameters = parameters.into_iter();
        let mut next_parameter = parameters.next();

        for expected_path in &expected_paths {
            if let Some(deferred_parameters) = deferred_parameters.as_mut() {
                // Once we fall back to deferred lookups, reconstruction is direct keyed removal
                // for the remaining expected paths.
                match deferred_parameters.remove(expected_path) {
                    Some(parameter) => values.push(parameter),
                    None => missing_paths.push(expected_path.to_string()),
                }
                continue;
            }

            match next_parameter.take() {
                Some((provided_path, parameter)) if &provided_path == expected_path => {
                    // As long as the provided parameter paths match the structure parameter paths that we are
                    // iterating over, we just consume the provided parameters without materializing a hash map.
                    values.push(parameter);
                    next_parameter = parameters.next();
                }
                Some((provided_path, parameter)) => {
                    // Once there is a mismatch, we materialize the hash map and continue via keyed lookup.
                    let mut deferred = defer(
                        std::iter::once((provided_path, parameter)).chain(parameters.by_ref()),
                        &expected_paths[..values.len()],
                        &mut duplicate_paths,
                    );
                    match deferred.remove(expected_path) {
                        Some(parameter) => values.push(parameter),
                        None => missing_paths.push(expected_path.to_string()),
                    }
                    deferred_parameters = Some(deferred);
                }
                None => missing_paths.push(expected_path.to_string()),
            }
        }

        if deferred_parameters.is_none()
            && let Some((provided_path, parameter)) = next_parameter.take()
        {
            // No mismatch occurred during traversal, but additional parameters remain. We need to materialize
            // the remaining parameters while still detecting those that repeat previously matched paths.
            deferred_parameters = Some(defer(
                std::iter::once((provided_path, parameter)).chain(parameters),
                &expected_paths[..values.len()],
                &mut duplicate_paths,
            ));
        }

        if !duplicate_paths.is_empty() {
            duplicate_paths.sort_unstable();
            duplicate_paths.dedup();
            Err(ParameterError::DuplicateParameterPaths { paths: duplicate_paths })
        } else if !missing_paths.is_empty() {
            Err(ParameterError::MissingParameters { expected_count, paths: Some(missing_paths) })
        } else if deferred_parameters.is_some_and(|deferred_parameters| !deferred_parameters.is_empty()) {
            Err(ParameterError::UnusedParameters { paths: None })
        } else {
            Self::from_parameters(structure, values)
        }
    }

    /// Reconstructs a value of this [`Parameterized`] type having the provided `structure` from the leaves of a tree
    /// prefix of it, `parameters`, each paired with its path. Unlike [`Self::from_named_parameters`], this function
    /// does not require exact leaf paths, and each leaf of `structure` receives the value of its most specific prefix
    /// leaf (i.e., the prefix leaf whose path is the longest prefix of its path). It applies the rules of
    /// [`Self::broadcast_to_parameter_structure`] with the paths of the prefix leaves standing in for the nodes of the
    /// prefix tree. Every leaf of `structure` must therefore have a prefix leaf whose path is equal to or a prefix of
    /// its path, every prefix leaf path must be a node path of `structure` (e.g., the path of a container without
    /// children, which the prefix leaf covers without populating any leaf), and distinct prefix leaves must have
    /// distinct paths, and so must distinct nodes of `structure`, except strictly below a prefix leaf.
    ///
    /// Since the containers of the prefix tree are not provided, this function cannot check that every node of
    /// `structure` without leaves is a node of the prefix tree or lies below one of its leaves (e.g., it accepts
    /// `[[a]]` as a prefix of `[[x], []]`, whose node `$[1]` the prefix tree lacks). Use
    /// [`Self::broadcast_to_parameter_structure`] with the prefix tree itself for the full check. Note that since one
    /// prefix value may need to populate multiple leaves, this function requires `P: Clone`. For fully worked
    /// examples, refer to the examples provided in the top-level documentation of the [`Parameterized`] trait.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError::DuplicateParameterPaths`] when two prefix leaves or, after them, two nodes of
    /// `structure` that do not lie strictly below a prefix leaf have the same path, with the sorted duplicated paths.
    /// Otherwise, it returns [`ParameterError::MissingParameters`] with the paths of the leaves of `structure` that
    /// have no prefix leaf, in the order of [`Self::paths`]. Otherwise, it returns
    /// [`ParameterError::UnusedParameters`] with the sorted paths of the prefix leaves that are not node paths of
    /// `structure` and that have no such prefix leaf below them.
    fn from_broadcasted_named_parameters<I: IntoIterator<Item = (ParameterPath, P)>>(
        structure: Self::ParameterStructure,
        parameters: I,
    ) -> Result<Self, ParameterError>
    where
        P: Clone,
    {
        broadcast_named_parameters(structure, parameters, None)
    }

    /// Broadcasts this [`Parameterized`] value, as a tree prefix, into the provided target `structure`, returning a new
    /// value of type `T`. Each leaf of `structure` receives the value of its most specific prefix leaf (i.e., the leaf
    /// of this value whose path is the longest prefix of its path). The nodes of both trees (refer to [`Self::paths`])
    /// are matched by path, including the containers without leaves, and as in the tree prefix checks of JAX, the
    /// following rules apply to them:
    ///
    ///   - Every leaf of `structure` must have a leaf of this value whose path is equal to or a prefix of its path.
    ///   - Every node path of this value must be a node path of `structure`.
    ///   - Every node path of `structure` must be a node path of this value or lie at or below a leaf of this value.
    ///   - Distinct nodes of this value must have distinct paths, and so must distinct nodes of `structure`, except
    ///     strictly below a leaf of this value, which gives the same value to all the nodes below it.
    ///
    /// For example, `(a, b)` is a prefix of `(Vec::new(), x)`, a single leaf is a prefix of an empty [`Vec`], `[a, b]`
    /// is a prefix of `[[x], []]`, and `[[], b]` is a prefix of `[[], y]`. In contrast, `[a]` and `[[a]]` are not
    /// prefixes of `[[x], []]`, which has the node `$[1]` that they lack and that none of their leaves covers, and
    /// `[[a], []]` is not a prefix of `[[x]]`, nor is `[[]]` a prefix of `[]`, because each has a node (i.e., `$[1]`
    /// and `$[0]`, respectively) that the structure lacks. These checks apply to containers without leaves in both
    /// directions. [`Self::from_broadcasted_named_parameters`] broadcasts the named leaves of a tree prefix without
    /// its containers instead, and so it cannot check the nodes of `structure` without leaves. Note that since one
    /// leaf of this value may need to populate multiple leaves of `structure`, this function requires `P: Clone`. For
    /// fully worked examples, refer to the examples provided in the top-level documentation of the [`Parameterized`]
    /// trait.
    ///
    /// Map keys are identified by their [`Debug`] texts (refer to [`ParameterPathSegment::Key`]), and so the last rule
    /// requires distinct keys of one map to have distinct [`Debug`] texts, unless a leaf of this value above that map
    /// covers all of its entries. For instance, the entries of two keys that both print `[private]` share the path
    /// `$[[private]]`, and an option map with one value for each of them would otherwise give one of the values to
    /// both entries.
    ///
    /// Nodes are matched by their paths only, which is intended. This value may therefore have another type than `T`
    /// whose nodes have the same paths (e.g., a derived struct with the same parameter fields as `T`, or an array for
    /// a [`Vec`]), and a node of the same type whose static metadata (i.e., the values of its non-parameter fields)
    /// differs also matches, in which case the result takes that metadata from `structure`.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError::DuplicateParameterPaths`] when this value or, after it, `structure` has nodes that
    /// violate the fourth rule, with the sorted duplicated paths. Otherwise, it returns
    /// [`ParameterError::MissingParameters`] when `structure` has nodes that violate the first or the third rule. The
    /// descendants of such a node violate them too, and so the error only lists the paths of the violating nodes
    /// without violating children (e.g., the leaves without a leaf of this value above them and the containers
    /// without children that this value does not reach, such as `$[1]` for the prefix `[[a]]` of `[[x], []]`), in the
    /// order of [`Self::paths`]. Otherwise, it returns [`ParameterError::UnusedParameters`] when this value has nodes
    /// that violate the second rule, with the sorted paths of the violating nodes without violating children (e.g.,
    /// `$[0]` for the prefix `[[]]` of `[]`).
    fn broadcast_to_parameter_structure<T: Parameterized<P>>(
        &self,
        structure: T::ParameterStructure,
    ) -> Result<T, ParameterError>
    where
        P: Clone,
    {
        broadcast_named_parameters(
            structure,
            self.named_parameters().map(|(path, parameter)| (path, parameter.clone())),
            Some(self.paths()),
        )
    }

    /// Maps each nested [`Parameter`] of type `P` in this value using the provided fallible `map_fn` to a [`Parameter`]
    /// of type `T`, while preserving the [`Parameterized`] structure of this type. Nested parameters are visited in the
    /// same order as [`Self::parameters`], [`Self::parameters_mut`], [`Self::into_parameters`],
    /// [`Self::named_parameters`], [`Self::named_parameters_mut`], and [`Self::into_named_parameters`]. `E` must
    /// implement [`From<ParameterError>`] so that reconstruction errors can be surfaced through the same error type
    /// as mapping failures.
    #[inline]
    fn try_map_parameters<T: Parameter, E: From<ParameterError>, MapFn: FnMut(P) -> Result<T, E>>(
        self,
        map_fn: MapFn,
    ) -> Result<Self::To<T>, E>
    where
        Self::Family: ParameterizedFamily<T>,
    {
        Self::To::<T>::from_parameters(
            self.parameter_structure(),
            self.into_parameters().map(map_fn).collect::<Result<Vec<_>, _>>()?,
        )
        .map_err(E::from)
    }

    /// Maps each nested [`Parameter`] of type `P` in this value using the provided `map_fn` to a [`Parameter`] of type
    /// `T`, while preserving the [`Parameterized`] structure of this type. Nested parameters are visited in the same
    /// order as [`Self::parameters`], [`Self::parameters_mut`], [`Self::into_parameters`], [`Self::named_parameters`],
    /// [`Self::named_parameters_mut`], and [`Self::into_named_parameters`].
    #[inline]
    fn map_parameters<T: Parameter, MapFn: FnMut(P) -> T>(
        self,
        mut map_fn: MapFn,
    ) -> Result<Self::To<T>, ParameterError>
    where
        Self::Family: ParameterizedFamily<T>,
    {
        self.try_map_parameters(|parameter| Ok(map_fn(parameter)))
    }

    /// Maps each nested [`Parameter`] of type `P` in this value using the provided fallible `map_fn`, which receives
    /// the [`ParameterPath`] for each [`Parameter`] along with its value, and returns a new [`Parameter`] value of type
    /// `T`, while preserving the [`Parameterized`] structure of this type. Nested parameters are visited in the same
    /// order as [`Self::parameters`], [`Self::parameters_mut`], [`Self::into_parameters`],
    /// [`Self::named_parameters`], [`Self::named_parameters_mut`], and [`Self::into_named_parameters`]. `E` must
    /// implement [`From<ParameterError>`] so reconstruction errors can be surfaced through the same error type as
    /// mapping failures.
    fn try_map_named_parameters<
        T: Parameter,
        E: From<ParameterError>,
        MapFn: FnMut(&ParameterPath, P) -> Result<T, E>,
    >(
        self,
        map_fn: MapFn,
    ) -> Result<Self::To<T>, E>
    where
        Self::Family: ParameterizedFamily<T>,
    {
        // The mapped parameters are reassembled by position rather than by path, so that parameters with the same
        // path (e.g., the values of two map keys with the same `Debug` text) stay apart.
        let mut map_fn = map_fn;
        Self::To::<T>::from_parameters(
            self.parameter_structure(),
            self.into_named_parameters()
                .map(|(path, parameter)| map_fn(&path, parameter))
                .collect::<Result<Vec<_>, _>>()?,
        )
        .map_err(E::from)
    }

    /// Maps each nested [`Parameter`] of type `P` in this value using the provided `map_fn`, which receives the
    /// [`ParameterPath`] for each [`Parameter`] along with its value, and returns a new [`Parameter`] value of type
    /// `T`, while preserving the [`Parameterized`] structure of this type. Nested parameters are visited in the
    /// same order as [`Self::parameters`], [`Self::parameters_mut`], [`Self::into_parameters`],
    /// [`Self::named_parameters`], [`Self::named_parameters_mut`], and [`Self::into_named_parameters`].
    #[inline]
    fn map_named_parameters<T: Parameter, MapFn: FnMut(&ParameterPath, P) -> T>(
        self,
        mut map_fn: MapFn,
    ) -> Result<Self::To<T>, ParameterError>
    where
        Self::Family: ParameterizedFamily<T>,
    {
        self.try_map_named_parameters(|path, parameter| Ok(map_fn(path, parameter)))
    }

    /// Filters all nested [`Parameter`]s of type `P` in this value according to the provided `predicate`, producing a
    /// structure-preserving [`Parameterized`] value with `Option<P>` leaves. Specifically, this value is a
    /// `Parameterized<P>` and this function returns a `Parameterized<Option<P>>` where each parameter of this value
    /// for which `predicate` returns `true` is kept as [`Some`], while all other parameters are replaced by [`None`].
    ///
    /// This function is inspired by
    /// [Equinox's `filter` function](https://docs.kidger.site/equinox/api/manipulation/#equinox.filter).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::parameters::Parameterized;
    ///
    /// let value = vec![(1_i32, 2_i32), (3_i32, 4_i32)];
    ///
    /// // Keep only the second tuple element across all top-level entries.
    /// let filtered = value.filter_parameters(|path, _| path.to_string().ends_with(".1"))?;
    ///
    /// assert_eq!(filtered, vec![(None, Some(2)), (None, Some(4))]);
    ///
    /// # Ok::<(), ryft_core::parameters::ParameterError>(())
    /// ```
    fn filter_parameters<F: FnMut(&ParameterPath, &P) -> bool>(
        self,
        predicate: F,
    ) -> Result<Self::To<Option<P>>, ParameterError>
    where
        Self::Family: ParameterizedFamily<Option<P>>,
    {
        let mut predicate = predicate;
        self.map_named_parameters(|path, parameter| predicate(path, &parameter).then_some(parameter))
    }

    /// Partitions all nested [`Parameter`]s of type `P` in this value into two structure-preserving [`Parameterized`]
    /// values, according to the provided `predicate`. Specifically, this value is a `Parameterized<P>` and this
    /// function returns a pair of `Parameterized<Option<P>>` values. The first one contains [`Some`] for each parameter
    /// for which the provided `predicate` returns `true`, and [`None`] elsewhere, and the opposite holds for the second
    /// returned value. This function is equivalent to using [`Self::filter_parameters`] twice on the same value, but it
    /// avoids traversing the structure twice.
    ///
    /// This function is inspired by
    /// [Equinox's `partition` function](https://docs.kidger.site/equinox/api/manipulation/#equinox.partition).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::parameters::Parameterized;
    ///
    /// let value = vec![(1_i32, 2_i32), (3_i32, 4_i32)];
    ///
    /// // Keep only parameters under the second top-level element in `partition_0`.
    /// let (partition_0, partition_1) = value.partition_parameters(|path, _| path.to_string().starts_with("$[1]"))?;
    ///
    /// assert_eq!(partition_0, vec![(None, None), (Some(3), Some(4))]);
    /// assert_eq!(partition_1, vec![(Some(1), Some(2)), (None, None)]);
    ///
    /// # Ok::<(), ryft_core::parameters::ParameterError>(())
    /// ```
    fn partition_parameters<F: FnMut(&ParameterPath, &P) -> bool>(
        self,
        predicate: F,
    ) -> Result<(Self::To<Option<P>>, Self::To<Option<P>>), ParameterError>
    where
        Self::Family: ParameterizedFamily<Option<P>>,
    {
        let structure = self.parameter_structure();
        let mut predicate = predicate;
        let mut partition_0_parameters = Vec::new();
        let mut partition_1_parameters = Vec::new();
        partition_0_parameters.reserve_exact(structure.parameter_count());
        partition_1_parameters.reserve_exact(structure.parameter_count());
        for (path, parameter) in self.into_named_parameters() {
            if predicate(&path, &parameter) {
                partition_0_parameters.push(Some(parameter));
                partition_1_parameters.push(None);
            } else {
                partition_0_parameters.push(None);
                partition_1_parameters.push(Some(parameter));
            }
        }
        let partition_0 = Self::To::from_parameters(structure, partition_0_parameters)?;
        let partition_1 = Self::To::from_parameters(partition_0.parameter_structure(), partition_1_parameters)?;
        Ok((partition_0, partition_1))
    }

    /// Combines multiple structure-aligned `Parameterized<Option<P>>` values into a single `Parameterized<P>` value,
    /// using left-to-right precedence at each parameter location. That is, for each leaf [`ParameterPath`] in
    /// `structure`, this function selects the first [`Some`] value from `values` and uses it for the corresponding
    /// location in the resulting [`Parameterized`] value that the function returns. If multiple non-`None` values are
    /// present for the same leaf, then they must all be equal, and otherwise this function returns an
    /// [`ParameterError::AmbiguousParameterCombination`] error. If no [`Some`] value is found for some leaf
    /// [`ParameterPath`], then this function will return a [`ParameterError::MissingParameters`] error. Furthermore,
    /// if any of the provided `values` contains additional [`Parameter`]s, beyond those that correspond to ones in
    /// `structure`, then this function will return an [`ParameterError::UnusedParameters`] error.
    ///
    /// This function is typically used to reconstruct values from the results of calling [`Self::filter_parameters`]
    /// and [`Self::partition_parameters`]. It is inspired by
    /// [Equinox's `combine` function](https://docs.kidger.site/equinox/api/manipulation/#equinox.combine).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::parameters::Parameterized;
    ///
    /// let value = vec![(1_i32, 2_i32), (3_i32, 4_i32)];
    /// let structure = value.parameter_structure();
    ///
    /// // Split by top-level element and then reconstruct the original value.
    /// let (partition_0, partition_1) = value.partition_parameters(|path, _| path.to_string().starts_with("$[0]"))?;
    /// let combined = Vec::<(i32, i32)>::combine_parameters(
    ///     structure,
    ///     vec![partition_0, partition_1],
    /// )?;
    ///
    /// assert_eq!(combined, vec![(1, 2), (3, 4)]);
    ///
    /// # Ok::<(), ryft_core::parameters::ParameterError>(())
    /// ```
    fn combine_parameters<I: IntoIterator<Item = Self::To<Option<P>>>>(
        structure: Self::ParameterStructure,
        values: I,
    ) -> Result<Self, ParameterError>
    where
        P: Debug + PartialEq,
        Self::Family: ParameterizedFamily<Option<P>>,
    {
        let expected_paths = structure.named_parameters().map(|(path, _)| path).collect::<Vec<_>>();
        let expected_count = expected_paths.len();
        let mut value_parameters = values.into_iter().map(|value| value.into_named_parameters()).collect::<Vec<_>>();
        let mut parameters = Vec::new();
        let mut missing_paths = Vec::new();
        parameters.reserve_exact(expected_count);
        for path in expected_paths {
            let mut collected_values = Vec::new();
            let mut has_missing_candidates = false;
            for iterator in &mut value_parameters {
                let Some((_, candidate)) = iterator.next() else {
                    has_missing_candidates = true;
                    continue;
                };
                let Some(candidate) = candidate else {
                    continue;
                };
                if !collected_values.iter().any(|value| value == &candidate) {
                    collected_values.push(candidate);
                }
            }
            if has_missing_candidates || collected_values.is_empty() {
                missing_paths.push(path.to_string());
            } else if collected_values.len() > 1 {
                return Err(ParameterError::AmbiguousParameterCombination {
                    values: collected_values.into_iter().map(|value| format!("{value:?}")).collect(),
                });
            } else {
                parameters.push(collected_values.pop().unwrap());
            }
        }

        let mut unused_paths = value_parameters
            .iter_mut()
            .flat_map(|iterator| iterator.map(|(path, _)| path.to_string()))
            .collect::<Vec<_>>();

        if !missing_paths.is_empty() {
            Err(ParameterError::MissingParameters { expected_count, paths: Some(missing_paths) })
        } else if !unused_paths.is_empty() {
            unused_paths.sort_unstable();
            Err(ParameterError::UnusedParameters { paths: Some(unused_paths) })
        } else {
            Self::from_parameters(structure, parameters)
        }
    }

    /// Replaces nested [`Parameter`]s of type `P` in this value using a structure-aligned `Parameterized<Option<P>>`
    /// replacement value. For each parameter, [`Some`] in `replacement` overrides the current value from `self`,
    /// while [`None`] keeps the current value unchanged. If `replacement` is missing parameters for the expected
    /// structure, this function will return a [`ParameterError::MissingParameters`] error. Furthermore, if
    /// `replacement` contains extra parameters, this function will return an [`ParameterError::UnusedParameters`]
    /// error.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::parameters::Parameterized;
    ///
    /// let value = vec![(1_i32, 2_i32), (3_i32, 4_i32)];
    ///
    /// // Replace only the first field of the second tuple.
    /// let replaced = value.replace_parameters(vec![(None, None), (Some(99), None)])?;
    ///
    /// assert_eq!(replaced, vec![(1, 2), (99, 4)]);
    ///
    /// # Ok::<(), ryft_core::parameters::ParameterError>(())
    /// ```
    fn replace_parameters(self, replacement: Self::To<Option<P>>) -> Result<Self, ParameterError>
    where
        Self::Family: ParameterizedFamily<Option<P>>,
    {
        let structure = self.parameter_structure();
        let expected_count = structure.parameter_count();
        let expected_paths = structure.named_parameters().map(|(path, _)| path).collect::<Vec<_>>();
        let mut parameters = self.into_named_parameters();
        let mut replacements = replacement.into_named_parameters();
        let mut replaced_parameters = Vec::new();
        replaced_parameters.reserve_exact(expected_count);
        let mut missing_paths = Vec::new();

        for path in expected_paths {
            let parameter = parameters.next().map(|(_, parameter)| parameter);
            let replacement = replacements.next().map(|(_, replacement)| replacement);
            if let (Some(parameter), Some(replacement)) = (parameter, replacement) {
                replaced_parameters.push(replacement.unwrap_or(parameter));
            } else {
                missing_paths.push(path.to_string());
            }
        }

        let mut unused_paths = parameters
            .map(|(path, _)| path.to_string())
            .chain(replacements.map(|(path, _)| path.to_string()))
            .collect::<Vec<_>>();

        if !missing_paths.is_empty() {
            Err(ParameterError::MissingParameters { expected_count, paths: Some(missing_paths) })
        } else if !unused_paths.is_empty() {
            unused_paths.sort_unstable();
            Err(ParameterError::UnusedParameters { paths: Some(unused_paths) })
        } else {
            Self::from_parameters(structure, replaced_parameters)
        }
    }
}

/// Returns the sorted [`Display`] texts of the [`ParameterPath`]s that occur more than once in `paths`, each once.
pub(crate) fn find_duplicate_paths<'p, P: IntoIterator<Item = &'p ParameterPath>>(paths: P) -> Vec<String> {
    let mut seen_paths = HashSet::new();
    let mut duplicate_paths = paths
        .into_iter()
        .filter(|path| !seen_paths.insert(*path))
        .map(ParameterPath::to_string)
        .collect::<Vec<_>>();
    duplicate_paths.sort_unstable();
    duplicate_paths.dedup();
    duplicate_paths
}

/// Reconstructs a value of type `T` having the provided `structure` from a tree prefix of it, which is
/// given as its leaves, `parameters`, and optionally as its nodes, `prefix_node_paths` (i.e., the paths that
/// [`Parameterized::paths`] returns for it, which include the paths of its leaves). This is the shared implementation
/// of [`Parameterized::broadcast_to_parameter_structure`], which provides the node paths and applies all of its rules,
/// and of [`Parameterized::from_broadcasted_named_parameters`], which does not provide them and so skips only the check
/// that the nodes of `structure` without leaves are nodes of the prefix tree or lie below one of its leaves.
fn broadcast_named_parameters<P: Clone + Parameter, T: Parameterized<P>>(
    structure: T::ParameterStructure,
    parameters: impl IntoIterator<Item = (ParameterPath, P)>,
    prefix_node_paths: Option<Vec<ParameterPath>>,
) -> Result<T, ParameterError> {
    // This function uses a small local prefix trie to resolve each path to the most specific prefix leaf (i.e.,
    // the longest matching prefix) in path-depth time, instead of scanning all prefix leaves for every path.
    // That is usually much faster when many prefix leaves are provided, at the cost of the additional trie
    // construction/allocation overhead, which can be less favorable for very small prefix trees.

    #[derive(Default)]
    struct PrefixTrieNode {
        children: HashMap<ParameterPathSegment, usize>,
        selected_prefix_index: Option<usize>,
    }

    // Distinct nodes of one tree that have the same path (e.g., the entries of two map keys with the same `Debug`
    // text) cannot be told apart by path, and so the prefix tree must not have any. Its leaves and its node paths
    // are checked separately because `prefix_node_paths` also contains the paths of the leaves.
    let prefix_parameters = parameters.into_iter().collect::<Vec<_>>();
    let mut duplicate_paths = find_duplicate_paths(prefix_parameters.iter().map(|(path, _)| path));
    if let Some(prefix_node_paths) = &prefix_node_paths {
        duplicate_paths.extend(find_duplicate_paths(prefix_node_paths));
    }
    if !duplicate_paths.is_empty() {
        duplicate_paths.sort_unstable();
        duplicate_paths.dedup();
        return Err(ParameterError::DuplicateParameterPaths { paths: duplicate_paths });
    }

    // Build a trie of the prefix leaf paths so that each path can be resolved by walking segments instead of
    // scanning all prefix leaves. This turns per-path prefix selection into a path-depth walk.
    let mut prefix_trie = vec![PrefixTrieNode::default()];
    for (prefix_index, (path, _)) in prefix_parameters.iter().enumerate() {
        let mut node_index = 0usize;
        for segment in path.segments() {
            let child_index = if let Some(child_index) = prefix_trie[node_index].children.get(segment).copied() {
                child_index
            } else {
                prefix_trie.push(PrefixTrieNode::default());
                let child_index = prefix_trie.len() - 1;
                prefix_trie[node_index].children.insert(segment.clone(), child_index);
                child_index
            };
            node_index = child_index;
        }
        prefix_trie[node_index].selected_prefix_index = Some(prefix_index);
    }

    // Returns the index of the most specific prefix leaf of `path`, if any. As we descend the trie, we track the
    // deepest prefix leaf encountered, which is the "most specific" match.
    let select_prefix_index = |path: &ParameterPath| {
        let mut trie_node_index = 0usize;
        let mut selected_prefix_index = prefix_trie[trie_node_index].selected_prefix_index;
        for segment in path.segments() {
            let Some(next_trie_node_index) = prefix_trie[trie_node_index].children.get(segment).copied() else {
                break;
            };
            trie_node_index = next_trie_node_index;
            if let Some(prefix_index) = prefix_trie[trie_node_index].selected_prefix_index {
                selected_prefix_index = Some(prefix_index);
            }
        }
        selected_prefix_index
    };

    // Returns `true` if a prefix leaf path is a proper prefix of `path`, in which case that prefix leaf gives the
    // same value to the node at `path` and to every other node with the same path.
    let lies_below_prefix_leaf = |path: &ParameterPath| {
        let mut trie_node_index = 0usize;
        for segment in path.segments() {
            if prefix_trie[trie_node_index].selected_prefix_index.is_some() {
                return true;
            }
            let Some(next_trie_node_index) = prefix_trie[trie_node_index].children.get(segment).copied() else {
                return false;
            };
            trie_node_index = next_trie_node_index;
        }
        false
    };

    // Nodes of `structure` with the same path are only told apart when the prefix tree reaches that path with its
    // own nodes, which then match all of them alike.
    let structure_paths = structure.paths();
    let duplicate_paths = find_duplicate_paths(structure_paths.iter().filter(|path| !lies_below_prefix_leaf(path)));
    if !duplicate_paths.is_empty() {
        return Err(ParameterError::DuplicateParameterPaths { paths: duplicate_paths });
    }

    // Returns the paths among `paths` that have no child among `paths`. The nodes that violate a rule are closed
    // under taking descendants, and so these are the deepest violating nodes, whose paths imply all the others.
    let deepest_paths = |paths: Vec<&ParameterPath>| {
        let parent_paths = paths
            .iter()
            .filter(|path| !path.is_root())
            .map(|path| ParameterPath { segments: path.segments[1..].to_vec() })
            .collect::<HashSet<_>>();
        paths
            .into_iter()
            .filter(|path| !parent_paths.contains(*path))
            .map(ParameterPath::to_string)
            .collect()
    };

    // Keep values in `structure.named_parameters()` order so that reconstruction can use `from_parameters`.
    let expected_count = structure.parameter_count();
    let mut values = Vec::with_capacity(expected_count);
    let mut uncovered_parameter_paths = HashSet::new();
    for (path, _) in structure.named_parameters() {
        match select_prefix_index(&path) {
            Some(prefix_index) => values.push(prefix_parameters[prefix_index].1.clone()),
            None => {
                uncovered_parameter_paths.insert(path);
            }
        }
    }

    // Without the node paths of the prefix tree, the paths of its leaves stand in for its nodes, and so the nodes of
    // `structure` without leaves cannot be checked for coverage.
    let has_prefix_node_paths = prefix_node_paths.is_some();
    let prefix_paths = prefix_node_paths
        .into_iter()
        .flatten()
        .chain(prefix_parameters.iter().map(|(path, _)| path.clone()))
        .collect::<HashSet<_>>();
    let missing_paths: Vec<String> = deepest_paths(
        structure_paths
            .iter()
            .filter(|path| {
                uncovered_parameter_paths.contains(*path)
                    || (has_prefix_node_paths && !prefix_paths.contains(*path) && select_prefix_index(path).is_none())
            })
            .collect(),
    );
    if !missing_paths.is_empty() {
        return Err(ParameterError::MissingParameters { expected_count, paths: Some(missing_paths) });
    }

    let structure_paths = structure_paths.iter().collect::<HashSet<_>>();
    let mut unused_paths: Vec<String> =
        deepest_paths(prefix_paths.iter().filter(|path| !structure_paths.contains(path)).collect());
    if !unused_paths.is_empty() {
        unused_paths.sort_unstable();
        return Err(ParameterError::UnusedParameters { paths: Some(unused_paths) });
    }

    T::from_parameters(structure, values)
}

/// Iterator adapter that prefixes each yielded [`ParameterPath`] with a stored [`ParameterPathSegment`]. This exists as
/// a dedicated type (instead of using only standard [`Iterator`] combinators) because many [`Parameterized`] associated
/// iterator types must be named concrete types. A closure-based `map(move |...| ...)` adapter would capture the prefix
/// segment and produce an unnameable closure type, which is not usable directly in those associated type definitions on
/// stable Rust. [`PathPrefixedParameterIterator`] preserves static dispatch and avoids heap allocation and dynamic
/// dispatch.
pub struct PathPrefixedParameterIterator<P, I: Iterator<Item = (ParameterPath, P)>> {
    /// Underlying [`Iterator`] that yields `(path, value)` pairs before prefixing with [`Self::segment`].
    iterator: I,

    /// [`ParameterPathSegment`] to prepend to each path produced by [`Self::iterator`].
    segment: ParameterPathSegment,
}

impl<P, I: Iterator<Item = (ParameterPath, P)>> PathPrefixedParameterIterator<P, I> {
    /// Creates a new iterator that prefixes paths from `iterator` with `segment`.
    #[inline]
    pub fn new(iterator: I, segment: ParameterPathSegment) -> Self {
        Self { iterator, segment }
    }
}

impl<P, I: Iterator<Item = (ParameterPath, P)>> Iterator for PathPrefixedParameterIterator<P, I> {
    type Item = (ParameterPath, P);

    fn next(&mut self) -> Option<Self::Item> {
        self.iterator.next().map(|(mut path, parameter)| {
            path.segments.push(self.segment.clone());
            (path, parameter)
        })
    }
}

pub struct ParameterParameterizedFamily;

impl<P: Parameter> ParameterizedFamily<P> for ParameterParameterizedFamily {
    type To = P;
}

impl<P: Parameter> Parameterized<P> for P {
    type Family = ParameterParameterizedFamily;

    type To<T: Parameter>
        = <Self::Family as ParameterizedFamily<T>>::To
    where
        Self::Family: ParameterizedFamily<T>;

    type ParameterStructure = Self::To<Placeholder>;

    type ParameterIterator<'t, T: 't + Parameter>
        = std::iter::Once<&'t T>
    where
        Self: 't;

    type ParameterIteratorMut<'t, T: 't + Parameter>
        = std::iter::Once<&'t mut T>
    where
        Self: 't;

    type ParameterIntoIterator<T: Parameter> = std::iter::Once<T>;

    type NamedParameterIterator<'t, T: 't + Parameter>
        = std::iter::Once<(ParameterPath, &'t T)>
    where
        Self: 't;

    type NamedParameterIteratorMut<'t, T: 't + Parameter>
        = std::iter::Once<(ParameterPath, &'t mut T)>
    where
        Self: 't;

    type NamedParameterIntoIterator<T: Parameter> = std::iter::Once<(ParameterPath, T)>;

    fn parameter_count(&self) -> usize {
        1
    }

    fn parameter_structure(&self) -> Self::ParameterStructure {
        Placeholder
    }

    fn parameters(&self) -> Self::ParameterIterator<'_, P> {
        std::iter::once(self)
    }

    fn parameters_mut(&mut self) -> Self::ParameterIteratorMut<'_, P> {
        std::iter::once(self)
    }

    fn into_parameters(self) -> Self::ParameterIntoIterator<P> {
        std::iter::once(self)
    }

    fn named_parameters(&self) -> Self::NamedParameterIterator<'_, P> {
        std::iter::once((ParameterPath::root(), self))
    }

    fn named_parameters_mut(&mut self) -> Self::NamedParameterIteratorMut<'_, P> {
        std::iter::once((ParameterPath::root(), self))
    }

    fn into_named_parameters(self) -> Self::NamedParameterIntoIterator<P> {
        std::iter::once((ParameterPath::root(), self))
    }

    fn paths(&self) -> Vec<ParameterPath> {
        vec![ParameterPath::root()]
    }

    fn from_parameters_with_remainder<I: Iterator<Item = P>>(
        _structure: Self::ParameterStructure,
        parameters: &mut I,
    ) -> Result<Self, ParameterError> {
        parameters.next().ok_or(ParameterError::MissingParameters { expected_count: 1, paths: None })
    }
}

pub struct PhantomDataParameterizedFamily;

impl<P: Parameter> ParameterizedFamily<P> for PhantomDataParameterizedFamily {
    type To = PhantomData<P>;
}

impl<P: Parameter> Parameterized<P> for PhantomData<P> {
    type Family = PhantomDataParameterizedFamily;

    type To<T: Parameter>
        = <Self::Family as ParameterizedFamily<T>>::To
    where
        Self::Family: ParameterizedFamily<T>;

    type ParameterStructure = Self::To<Placeholder>;

    type ParameterIterator<'t, T: 't + Parameter>
        = std::iter::Empty<&'t T>
    where
        Self: 't;

    type ParameterIteratorMut<'t, T: 't + Parameter>
        = std::iter::Empty<&'t mut T>
    where
        Self: 't;

    type ParameterIntoIterator<T: Parameter> = std::iter::Empty<T>;

    type NamedParameterIterator<'t, T: 't + Parameter>
        = std::iter::Empty<(ParameterPath, &'t T)>
    where
        Self: 't;

    type NamedParameterIteratorMut<'t, T: 't + Parameter>
        = std::iter::Empty<(ParameterPath, &'t mut T)>
    where
        Self: 't;

    type NamedParameterIntoIterator<T: Parameter> = std::iter::Empty<(ParameterPath, T)>;

    fn parameter_count(&self) -> usize {
        0
    }

    fn parameter_structure(&self) -> Self::ParameterStructure {
        PhantomData
    }

    fn parameters(&self) -> Self::ParameterIterator<'_, P> {
        std::iter::empty()
    }

    fn parameters_mut(&mut self) -> Self::ParameterIteratorMut<'_, P> {
        std::iter::empty()
    }

    fn into_parameters(self) -> Self::ParameterIntoIterator<P> {
        std::iter::empty()
    }

    fn named_parameters(&self) -> Self::NamedParameterIterator<'_, P> {
        std::iter::empty()
    }

    fn named_parameters_mut(&mut self) -> Self::NamedParameterIteratorMut<'_, P> {
        std::iter::empty()
    }

    fn into_named_parameters(self) -> Self::NamedParameterIntoIterator<P> {
        std::iter::empty()
    }

    fn paths(&self) -> Vec<ParameterPath> {
        vec![ParameterPath::root()]
    }

    fn from_parameters_with_remainder<I: Iterator<Item = P>>(
        _structure: Self::ParameterStructure,
        _parameters: &mut I,
    ) -> Result<Self, ParameterError> {
        Ok(PhantomData)
    }
}

// Use declarative macros to provide implementations for tuples of `Parameterized` items. Note that if a tuple
// contains a mix of `Parameterized` and non-`Parameterized` items, then the generated implementations here
// will not cover it. Instead, such tuples are supported when nested within `struct`s or `enum`s by using our
// `#[derive(Parameterized)]` macro as it provides special treatment for them.

macro_rules! tuple_parameterized_family_impl {
    ($($F:ident),*) => {
        impl<P: Parameter, $($F: ParameterizedFamily<P> + ParameterizedFamily<Placeholder>),*> ParameterizedFamily<P>
            for ($($F,)*)
        {
            type To = ($(<$F as ParameterizedFamily<P>>::To,)*);
        }
    };
}

tuple_parameterized_family_impl!();
tuple_parameterized_family_impl!(F0);
tuple_parameterized_family_impl!(F0, F1);
tuple_parameterized_family_impl!(F0, F1, F2);
tuple_parameterized_family_impl!(F0, F1, F2, F3);
tuple_parameterized_family_impl!(F0, F1, F2, F3, F4);
tuple_parameterized_family_impl!(F0, F1, F2, F3, F4, F5);
tuple_parameterized_family_impl!(F0, F1, F2, F3, F4, F5, F6);
tuple_parameterized_family_impl!(F0, F1, F2, F3, F4, F5, F6, F7);
tuple_parameterized_family_impl!(F0, F1, F2, F3, F4, F5, F6, F7, F8);
tuple_parameterized_family_impl!(F0, F1, F2, F3, F4, F5, F6, F7, F8, F9);
tuple_parameterized_family_impl!(F0, F1, F2, F3, F4, F5, F6, F7, F8, F9, F10);
tuple_parameterized_family_impl!(F0, F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11);

macro_rules! tuple_parameterized_impl {
    ($($T:ident:$index:tt),*) => {
        paste! {
            impl<P: Parameter$(, $T: Parameterized<P>)*> Parameterized<P> for ($($T,)*) {
                type Family = ($($T::Family,)*);

                type To<T: Parameter>
                    = <Self::Family as ParameterizedFamily<T>>::To
                where
                    Self::Family: ParameterizedFamily<T>;

                type ParameterStructure = Self::To<Placeholder>;

                type ParameterIterator<'t, T: 't + Parameter>
                    = tuple_parameter_iterator_ty!('t, T, ($($T:$index,)*))
                where
                    Self: 't;

                type ParameterIteratorMut<'t, T: 't + Parameter>
                    = tuple_parameter_iterator_mut_ty!('t, T, ($($T:$index,)*))
                where
                    Self: 't;

                type ParameterIntoIterator<T: Parameter>
                    = tuple_parameter_into_iterator_ty!(T, ($($T:$index,)*));

                type NamedParameterIterator<'t, T: 't + Parameter>
                    = tuple_named_parameter_iterator_ty!('t, T, ($($T:$index,)*))
                where
                    Self: 't;

                type NamedParameterIteratorMut<'t, T: 't + Parameter>
                    = tuple_named_parameter_iterator_mut_ty!('t, T, ($($T:$index,)*))
                where
                    Self: 't;

                type NamedParameterIntoIterator<T: Parameter>
                    = tuple_named_parameter_into_iterator_ty!(T, ($($T:$index,)*));

                fn parameter_count(&self) -> usize {
                    let ($([<$T:lower>],)*) = &self;
                    $([<$T:lower>].parameter_count()+)* 0usize
                }

                fn parameter_structure(&self) -> Self::ParameterStructure {
                    let ($([<$T:lower>],)*) = &self;
                    tuple_parameter_structure!($([<$T:lower>].parameter_structure(),)*)
                }

                fn parameters(&self) -> Self::ParameterIterator<'_, P> {
                    let ($([<$T:lower>],)*) = self;
                    tuple_parameter_iterator!(P, ($([<$T:lower>]:$index,)*))
                }

                fn parameters_mut(&mut self) -> Self::ParameterIteratorMut<'_, P> {
                    let ($([<$T:lower>],)*) = self;
                    tuple_parameter_iterator_mut!(P, ($([<$T:lower>]:$index,)*))
                }

                fn into_parameters(self) -> Self::ParameterIntoIterator<P> {
                    let ($([<$T:lower>],)*) = self;
                    tuple_parameter_into_iterator!(P, ($([<$T:lower>]:$index,)*))
                }

                fn named_parameters(&self) -> Self::NamedParameterIterator<'_, P> {
                    let ($([<$T:lower>],)*) = self;
                    tuple_named_parameter_iterator!(P, ($([<$T:lower>]:$index,)*))
                }

                fn named_parameters_mut(&mut self) -> Self::NamedParameterIteratorMut<'_, P> {
                    let ($([<$T:lower>],)*) = self;
                    tuple_named_parameter_iterator_mut!(P, ($([<$T:lower>]:$index,)*))
                }

                fn into_named_parameters(self) -> Self::NamedParameterIntoIterator<P> {
                    let ($([<$T:lower>],)*) = self;
                    tuple_named_parameter_into_iterator!(P, ($([<$T:lower>]:$index,)*))
                }

                fn paths(&self) -> Vec<ParameterPath> {
                    let ($([<$T:lower>],)*) = self;
                    tuple_paths!(($([<$T:lower>]:$index,)*))
                }

                #[allow(unused_variables)]
                fn from_parameters_with_remainder<I: Iterator<Item = P>>(
                    structure: Self::ParameterStructure,
                    parameters: &mut I,
                ) -> Result<Self, ParameterError> {
                    let ($([<$T:lower _field>],)*) = structure;
                    $(let [<$T:lower>] = $T::from_parameters_with_remainder([<$T:lower _field>], parameters)?;)*
                    Ok(($([<$T:lower>],)*))
                }
            }
        }
    };
}

macro_rules! tuple_parameter_structure {
    () => {
        Default::default()
    };

    ($($structure:expr,)*) => {
        ($($structure,)*)
    };
}

macro_rules! tuple_parameter_iterator_ty {
    ($t:lifetime, $T:ty, ()) => {
        std::iter::Empty<&$t $T>
    };

    ($t:lifetime, $T:ty, ($head:ident:$index:tt, $($tail:ident:$tail_index:tt,)*)) => {
        std::iter::Chain<
            $head::ParameterIterator<$t, $T>,
            tuple_parameter_iterator_ty!($t, $T, ($($tail:$tail_index,)*)),
        >
    };
}

macro_rules! tuple_parameter_iterator_mut_ty {
    ($t:lifetime, $T:ty, ()) => {
        std::iter::Empty<&$t mut $T>
    };

    ($t:lifetime, $T:ty, ($head:ident:$index:tt, $($tail:ident:$tail_index:tt,)*)) => {
        std::iter::Chain<
            $head::ParameterIteratorMut<$t, $T>,
            tuple_parameter_iterator_mut_ty!($t, $T, ($($tail:$tail_index,)*)),
        >
    };
}

macro_rules! tuple_parameter_into_iterator_ty {
    ($T:ty, ()) => {
        std::iter::Empty<$T>
    };

    ($T:ty, ($head:ident:$index:tt, $($tail:ident:$tail_index:tt,)*)) => {
        std::iter::Chain<
            $head::ParameterIntoIterator<$T>,
            tuple_parameter_into_iterator_ty!($T, ($($tail:$tail_index,)*)),
        >
    };
}

macro_rules! tuple_named_parameter_iterator_ty {
    ($t:lifetime, $T:ty, ()) => {
        std::iter::Empty<(ParameterPath, &$t $T)>
    };

    ($t:lifetime, $T:ty, ($head:ident:$index:tt, $($tail:ident:$tail_index:tt,)*)) => {
        std::iter::Chain<
            PathPrefixedParameterIterator<&$t $T, $head::NamedParameterIterator<$t, $T>>,
            tuple_named_parameter_iterator_ty!($t, $T, ($($tail:$tail_index,)*)),
        >
    };
}

macro_rules! tuple_named_parameter_iterator_mut_ty {
    ($t:lifetime, $T:ty, ()) => {
        std::iter::Empty<(ParameterPath, &$t mut $T)>
    };

    ($t:lifetime, $T:ty, ($head:ident:$index:tt, $($tail:ident:$tail_index:tt,)*)) => {
        std::iter::Chain<
            PathPrefixedParameterIterator<&$t mut $T, $head::NamedParameterIteratorMut<$t, $T>>,
            tuple_named_parameter_iterator_mut_ty!($t, $T, ($($tail:$tail_index,)*)),
        >
    };
}

macro_rules! tuple_named_parameter_into_iterator_ty {
    ($T:ty, ()) => {
        std::iter::Empty<(ParameterPath, $T)>
    };

    ($T:ty, ($head:ident:$index:tt, $($tail:ident:$tail_index:tt,)*)) => {
        std::iter::Chain<
            PathPrefixedParameterIterator<$T, $head::NamedParameterIntoIterator<$T>>,
            tuple_named_parameter_into_iterator_ty!($T, ($($tail:$tail_index,)*)),
        >
    };
}

macro_rules! tuple_parameter_iterator {
    ($T:tt, ()) => {
        std::iter::empty::<&'_ $T>()
    };

    ($T:tt, ($head:ident:$index:tt, $($tail:ident:$tail_index:tt,)*)) => {
        $head.parameters().chain(tuple_parameter_iterator!($T, ($($tail:$tail_index,)*)))
    };
}

macro_rules! tuple_parameter_iterator_mut {
    ($T:tt, ()) => {
        std::iter::empty::<&'_ mut $T>()
    };

    ($T:tt, ($head:ident:$index:tt, $($tail:ident:$tail_index:tt,)*)) => {
        $head.parameters_mut().chain(tuple_parameter_iterator_mut!($T, ($($tail:$tail_index,)*)))
    };
}

macro_rules! tuple_parameter_into_iterator {
    ($T:tt, ()) => {
        std::iter::empty::<$T>()
    };

    ($T:tt, ($head:ident:$index:tt, $($tail:ident:$tail_index:tt,)*)) => {
        $head.into_parameters().chain(tuple_parameter_into_iterator!($T, ($($tail:$tail_index,)*)))
    };
}

macro_rules! tuple_named_parameter_iterator {
    ($T:tt, ()) => {
        std::iter::empty::<(ParameterPath, &'_ $T)>()
    };

    ($T:tt, ($head:ident:$index:tt, $($tail:ident:$tail_index:tt,)*)) => {{
        let iterator = $head.named_parameters();
        let iterator = PathPrefixedParameterIterator::new(iterator, ParameterPathSegment::TupleIndex($index));
        iterator.chain(tuple_named_parameter_iterator!($T, ($($tail:$tail_index,)*)))
    }};
}

macro_rules! tuple_named_parameter_iterator_mut {
    ($T:tt, ()) => {
        std::iter::empty::<(ParameterPath, &'_ mut $T)>()
    };

    ($T:tt, ($head:ident:$index:tt, $($tail:ident:$tail_index:tt,)*)) => {{
        let iterator = $head.named_parameters_mut();
        let iterator = PathPrefixedParameterIterator::new(iterator, ParameterPathSegment::TupleIndex($index));
        iterator.chain(tuple_named_parameter_iterator_mut!($T, ($($tail:$tail_index,)*)))
    }};
}

macro_rules! tuple_named_parameter_into_iterator {
    ($T:tt, ()) => {
        std::iter::empty::<(ParameterPath, $T)>()
    };

    ($T:tt, ($head:ident:$index:tt, $($tail:ident:$tail_index:tt,)*)) => {{
        let iterator = $head.into_named_parameters();
        let iterator = PathPrefixedParameterIterator::new(iterator, ParameterPathSegment::TupleIndex($index));
        iterator.chain(tuple_named_parameter_into_iterator!($T, ($($tail:$tail_index,)*)))
    }};
}

macro_rules! tuple_paths {
    (()) => {
        vec![ParameterPath::root()]
    };

    (($($value:ident:$index:tt,)+)) => {{
        let mut paths = vec![ParameterPath::root()];
        $(paths.extend(
            PathPrefixedParameterIterator::new(
                $value.paths().into_iter().map(|path| (path, ())),
                ParameterPathSegment::TupleIndex($index),
            )
            .map(|(path, _)| path),
        );)+
        paths
    }};
}

tuple_parameterized_impl!();
tuple_parameterized_impl!(V0:0);
tuple_parameterized_impl!(V0:0, V1:1);
tuple_parameterized_impl!(V0:0, V1:1, V2:2);
tuple_parameterized_impl!(V0:0, V1:1, V2:2, V3:3);
tuple_parameterized_impl!(V0:0, V1:1, V2:2, V3:3, V4:4);
tuple_parameterized_impl!(V0:0, V1:1, V2:2, V3:3, V4:4, V5:5);
tuple_parameterized_impl!(V0:0, V1:1, V2:2, V3:3, V4:4, V5:5, V6:6);
tuple_parameterized_impl!(V0:0, V1:1, V2:2, V3:3, V4:4, V5:5, V6:6, V7:7);
tuple_parameterized_impl!(V0:0, V1:1, V2:2, V3:3, V4:4, V5:5, V6:6, V7:7, V8:8);
tuple_parameterized_impl!(V0:0, V1:1, V2:2, V3:3, V4:4, V5:5, V6:6, V7:7, V8:8, V9:9);
tuple_parameterized_impl!(V0:0, V1:1, V2:2, V3:3, V4:4, V5:5, V6:6, V7:7, V8:8, V9:9, V10:10);
tuple_parameterized_impl!(V0:0, V1:1, V2:2, V3:3, V4:4, V5:5, V6:6, V7:7, V8:8, V9:9, V10:10, V11:11);

pub struct ArrayParameterizedFamily<F, const N: usize>(PhantomData<fn() -> F>);

impl<P: Parameter, F: ParameterizedFamily<P> + ParameterizedFamily<Placeholder>, const N: usize> ParameterizedFamily<P>
    for ArrayParameterizedFamily<F, N>
{
    type To = [<F as ParameterizedFamily<P>>::To; N];
}

impl<P: Parameter, V: Parameterized<P>, const N: usize> Parameterized<P> for [V; N] {
    type Family = ArrayParameterizedFamily<V::Family, N>;

    type To<T: Parameter>
        = <Self::Family as ParameterizedFamily<T>>::To
    where
        Self::Family: ParameterizedFamily<T>;

    type ParameterStructure = Self::To<Placeholder>;

    type ParameterIterator<'t, T: 't + Parameter>
        = std::iter::FlatMap<
        std::slice::Iter<'t, V>,
        <V as Parameterized<P>>::ParameterIterator<'t, T>,
        fn(&'t V) -> <V as Parameterized<P>>::ParameterIterator<'t, T>,
    >
    where
        Self: 't;

    type ParameterIteratorMut<'t, T: 't + Parameter>
        = std::iter::FlatMap<
        std::slice::IterMut<'t, V>,
        <V as Parameterized<P>>::ParameterIteratorMut<'t, T>,
        fn(&'t mut V) -> <V as Parameterized<P>>::ParameterIteratorMut<'t, T>,
    >
    where
        Self: 't;

    type ParameterIntoIterator<T: Parameter> = std::iter::FlatMap<
        std::array::IntoIter<V, N>,
        <V as Parameterized<P>>::ParameterIntoIterator<T>,
        fn(V) -> <V as Parameterized<P>>::ParameterIntoIterator<T>,
    >;

    type NamedParameterIterator<'t, T: 't + Parameter>
        = std::iter::FlatMap<
        std::iter::Enumerate<std::slice::Iter<'t, V>>,
        PathPrefixedParameterIterator<&'t T, <V as Parameterized<P>>::NamedParameterIterator<'t, T>>,
        fn(
            (usize, &'t V),
        ) -> PathPrefixedParameterIterator<&'t T, <V as Parameterized<P>>::NamedParameterIterator<'t, T>>,
    >
    where
        Self: 't;

    type NamedParameterIteratorMut<'t, T: 't + Parameter>
        = std::iter::FlatMap<
        std::iter::Enumerate<std::slice::IterMut<'t, V>>,
        PathPrefixedParameterIterator<&'t mut T, <V as Parameterized<P>>::NamedParameterIteratorMut<'t, T>>,
        fn(
            (usize, &'t mut V),
        )
            -> PathPrefixedParameterIterator<&'t mut T, <V as Parameterized<P>>::NamedParameterIteratorMut<'t, T>>,
    >
    where
        Self: 't;

    type NamedParameterIntoIterator<T: Parameter> = std::iter::FlatMap<
        std::iter::Enumerate<std::array::IntoIter<V, N>>,
        PathPrefixedParameterIterator<T, <V as Parameterized<P>>::NamedParameterIntoIterator<T>>,
        fn((usize, V)) -> PathPrefixedParameterIterator<T, <V as Parameterized<P>>::NamedParameterIntoIterator<T>>,
    >;

    fn parameter_count(&self) -> usize {
        self.iter().map(|value| value.parameter_count()).sum()
    }

    fn parameter_structure(&self) -> Self::ParameterStructure {
        std::array::from_fn(|i| self[i].parameter_structure())
    }

    fn parameters(&self) -> Self::ParameterIterator<'_, P> {
        self.iter().flat_map(V::parameters)
    }

    fn parameters_mut(&mut self) -> Self::ParameterIteratorMut<'_, P> {
        self.iter_mut().flat_map(V::parameters_mut)
    }

    fn into_parameters(self) -> Self::ParameterIntoIterator<P> {
        self.into_iter().flat_map(V::into_parameters)
    }

    fn named_parameters(&self) -> Self::NamedParameterIterator<'_, P> {
        self.iter().enumerate().flat_map(|(index, value)| {
            PathPrefixedParameterIterator::new(value.named_parameters(), ParameterPathSegment::Index(index))
        })
    }

    fn named_parameters_mut(&mut self) -> Self::NamedParameterIteratorMut<'_, P> {
        self.iter_mut().enumerate().flat_map(|(index, value)| {
            PathPrefixedParameterIterator::new(value.named_parameters_mut(), ParameterPathSegment::Index(index))
        })
    }

    fn into_named_parameters(self) -> Self::NamedParameterIntoIterator<P> {
        self.into_iter().enumerate().flat_map(|(index, value)| {
            PathPrefixedParameterIterator::new(value.into_named_parameters(), ParameterPathSegment::Index(index))
        })
    }

    fn paths(&self) -> Vec<ParameterPath> {
        std::iter::once(ParameterPath::root())
            .chain(self.iter().enumerate().flat_map(|(index, value)| {
                let paths = value.paths().into_iter().map(|path| (path, ()));
                PathPrefixedParameterIterator::new(paths, ParameterPathSegment::Index(index)).map(|(path, _)| path)
            }))
            .collect()
    }

    fn from_parameters_with_remainder<I: Iterator<Item = P>>(
        structure: Self::ParameterStructure,
        parameters: &mut I,
    ) -> Result<Self, ParameterError> {
        // TODO(eaplatanios): Make this more efficient by using `std::array::try_from_fn` once it becomes stable.
        //  Tracking issue: https://github.com/rust-lang/rust/issues/89379.
        Ok(unsafe {
            structure
                .into_iter()
                .map(|value_structure| V::from_parameters_with_remainder(value_structure, parameters))
                .collect::<Result<Vec<V>, _>>()?
                .try_into()
                .unwrap_unchecked()
        })
    }
}

pub struct VecParameterizedFamily<F>(PhantomData<fn() -> F>);

impl<P: Parameter, F: ParameterizedFamily<P> + ParameterizedFamily<Placeholder>> ParameterizedFamily<P>
    for VecParameterizedFamily<F>
{
    type To = Vec<<F as ParameterizedFamily<P>>::To>;
}

impl<P: Parameter, V: Parameterized<P>> Parameterized<P> for Vec<V> {
    type Family = VecParameterizedFamily<V::Family>;

    type To<T: Parameter>
        = <Self::Family as ParameterizedFamily<T>>::To
    where
        Self::Family: ParameterizedFamily<T>;

    type ParameterStructure = Self::To<Placeholder>;

    type ParameterIterator<'t, T: 't + Parameter>
        = std::iter::FlatMap<
        std::slice::Iter<'t, V>,
        <V as Parameterized<P>>::ParameterIterator<'t, T>,
        fn(&'t V) -> <V as Parameterized<P>>::ParameterIterator<'t, T>,
    >
    where
        Self: 't;

    type ParameterIteratorMut<'t, T: 't + Parameter>
        = std::iter::FlatMap<
        std::slice::IterMut<'t, V>,
        <V as Parameterized<P>>::ParameterIteratorMut<'t, T>,
        fn(&'t mut V) -> <V as Parameterized<P>>::ParameterIteratorMut<'t, T>,
    >
    where
        Self: 't;

    type ParameterIntoIterator<T: Parameter> = std::iter::FlatMap<
        std::vec::IntoIter<V>,
        <V as Parameterized<P>>::ParameterIntoIterator<T>,
        fn(V) -> <V as Parameterized<P>>::ParameterIntoIterator<T>,
    >;

    type NamedParameterIterator<'t, T: 't + Parameter>
        = std::iter::FlatMap<
        std::iter::Enumerate<std::slice::Iter<'t, V>>,
        PathPrefixedParameterIterator<&'t T, <V as Parameterized<P>>::NamedParameterIterator<'t, T>>,
        fn(
            (usize, &'t V),
        ) -> PathPrefixedParameterIterator<&'t T, <V as Parameterized<P>>::NamedParameterIterator<'t, T>>,
    >
    where
        Self: 't;

    type NamedParameterIteratorMut<'t, T: 't + Parameter>
        = std::iter::FlatMap<
        std::iter::Enumerate<std::slice::IterMut<'t, V>>,
        PathPrefixedParameterIterator<&'t mut T, <V as Parameterized<P>>::NamedParameterIteratorMut<'t, T>>,
        fn(
            (usize, &'t mut V),
        )
            -> PathPrefixedParameterIterator<&'t mut T, <V as Parameterized<P>>::NamedParameterIteratorMut<'t, T>>,
    >
    where
        Self: 't;

    type NamedParameterIntoIterator<T: Parameter> = std::iter::FlatMap<
        std::iter::Enumerate<std::vec::IntoIter<V>>,
        PathPrefixedParameterIterator<T, <V as Parameterized<P>>::NamedParameterIntoIterator<T>>,
        fn((usize, V)) -> PathPrefixedParameterIterator<T, <V as Parameterized<P>>::NamedParameterIntoIterator<T>>,
    >;

    fn parameter_count(&self) -> usize {
        self.iter().map(|value| value.parameter_count()).sum()
    }

    fn parameter_structure(&self) -> Self::ParameterStructure {
        self.iter().map(|value| value.parameter_structure()).collect()
    }

    fn parameters(&self) -> Self::ParameterIterator<'_, P> {
        self.iter().flat_map(|value| value.parameters())
    }

    fn parameters_mut(&mut self) -> Self::ParameterIteratorMut<'_, P> {
        self.iter_mut().flat_map(|value| value.parameters_mut())
    }

    fn into_parameters(self) -> Self::ParameterIntoIterator<P> {
        self.into_iter().flat_map(|value| value.into_parameters())
    }

    fn named_parameters(&self) -> Self::NamedParameterIterator<'_, P> {
        self.iter().enumerate().flat_map(|(index, value)| {
            PathPrefixedParameterIterator::new(value.named_parameters(), ParameterPathSegment::Index(index))
        })
    }

    fn named_parameters_mut(&mut self) -> Self::NamedParameterIteratorMut<'_, P> {
        self.iter_mut().enumerate().flat_map(|(index, value)| {
            PathPrefixedParameterIterator::new(value.named_parameters_mut(), ParameterPathSegment::Index(index))
        })
    }

    fn into_named_parameters(self) -> Self::NamedParameterIntoIterator<P> {
        self.into_iter().enumerate().flat_map(|(index, value)| {
            PathPrefixedParameterIterator::new(value.into_named_parameters(), ParameterPathSegment::Index(index))
        })
    }

    fn paths(&self) -> Vec<ParameterPath> {
        std::iter::once(ParameterPath::root())
            .chain(self.iter().enumerate().flat_map(|(index, value)| {
                let paths = value.paths().into_iter().map(|path| (path, ()));
                PathPrefixedParameterIterator::new(paths, ParameterPathSegment::Index(index)).map(|(path, _)| path)
            }))
            .collect()
    }

    fn from_parameters_with_remainder<I: Iterator<Item = P>>(
        structure: Self::ParameterStructure,
        parameters: &mut I,
    ) -> Result<Self, ParameterError> {
        let expected_count = structure.len();
        let mut values = Vec::new();
        values.reserve_exact(expected_count);
        for value_structure in structure {
            values.push(V::from_parameters_with_remainder(value_structure, parameters).map_err(
                |error| match error {
                    ParameterError::MissingParameters { paths, .. } => {
                        ParameterError::MissingParameters { expected_count, paths }
                    }
                    error => error,
                },
            )?);
        }
        Ok(values)
    }
}

pub struct HashMapParameterizedFamily<K, F, S>(PhantomData<(K, F, S)>);

impl<
    P: Parameter,
    K: Clone + Debug + Eq + Ord + Hash,
    F: ParameterizedFamily<P> + ParameterizedFamily<Placeholder>,
    S: Clone + BuildHasher,
> ParameterizedFamily<P> for HashMapParameterizedFamily<K, F, S>
{
    type To = HashMap<K, <F as ParameterizedFamily<P>>::To, S>;
}

// TODO(eaplatanios): Find a simple way to avoid the need for sorting.
// The `Parameterized` implementation for `HashMap` is quite inefficient. That is because for most operations that
// require traversal of the `HashMap` we end up having to perform a sort of the results that we obtain (by key) to
// comply with the requirements of the `Parameterized`.
impl<P: Parameter, K: Clone + Debug + Eq + Ord + Hash, V: Parameterized<P>, S: Clone + BuildHasher> Parameterized<P>
    for HashMap<K, V, S>
{
    type Family = HashMapParameterizedFamily<K, V::Family, S>;

    type To<T: Parameter>
        = <Self::Family as ParameterizedFamily<T>>::To
    where
        Self::Family: ParameterizedFamily<T>;

    type ParameterStructure = Self::To<Placeholder>;

    type ParameterIterator<'t, T: 't + Parameter>
        = std::iter::FlatMap<
        std::vec::IntoIter<(K, &'t V)>,
        <V as Parameterized<P>>::ParameterIterator<'t, T>,
        fn((K, &'t V)) -> <V as Parameterized<P>>::ParameterIterator<'t, T>,
    >
    where
        Self: 't;

    type ParameterIteratorMut<'t, T: 't + Parameter>
        = std::iter::FlatMap<
        std::vec::IntoIter<(K, *mut V)>,
        <V as Parameterized<P>>::ParameterIteratorMut<'t, T>,
        fn((K, *mut V)) -> <V as Parameterized<P>>::ParameterIteratorMut<'t, T>,
    >
    where
        Self: 't;

    type ParameterIntoIterator<T: Parameter> = std::iter::FlatMap<
        std::vec::IntoIter<(K, V)>,
        <V as Parameterized<P>>::ParameterIntoIterator<T>,
        fn((K, V)) -> <V as Parameterized<P>>::ParameterIntoIterator<T>,
    >;

    type NamedParameterIterator<'t, T: 't + Parameter>
        = std::iter::FlatMap<
        std::vec::IntoIter<(&'t K, &'t V)>,
        PathPrefixedParameterIterator<&'t T, <V as Parameterized<P>>::NamedParameterIterator<'t, T>>,
        fn(
            (&'t K, &'t V),
        ) -> PathPrefixedParameterIterator<&'t T, <V as Parameterized<P>>::NamedParameterIterator<'t, T>>,
    >
    where
        Self: 't;

    type NamedParameterIteratorMut<'t, T: 't + Parameter>
        = std::iter::FlatMap<
        std::vec::IntoIter<(K, *mut V)>,
        PathPrefixedParameterIterator<&'t mut T, <V as Parameterized<P>>::NamedParameterIteratorMut<'t, T>>,
        fn(
            (K, *mut V),
        )
            -> PathPrefixedParameterIterator<&'t mut T, <V as Parameterized<P>>::NamedParameterIteratorMut<'t, T>>,
    >
    where
        Self: 't;

    type NamedParameterIntoIterator<T: Parameter> = std::iter::FlatMap<
        std::vec::IntoIter<(K, V)>,
        PathPrefixedParameterIterator<T, <V as Parameterized<P>>::NamedParameterIntoIterator<T>>,
        fn((K, V)) -> PathPrefixedParameterIterator<T, <V as Parameterized<P>>::NamedParameterIntoIterator<T>>,
    >;

    fn parameter_count(&self) -> usize {
        self.values().map(|value| value.parameter_count()).sum()
    }

    fn parameter_structure(&self) -> Self::ParameterStructure {
        let mut structure = HashMap::with_capacity_and_hasher(self.len(), self.hasher().clone());
        let mut sorted_entries = self.iter().map(|(k, v)| (k.clone(), v.parameter_structure())).collect::<Vec<_>>();
        sorted_entries.sort_unstable_by(|(l, _), (r, _)| l.cmp(r));
        structure.extend(sorted_entries);
        structure
    }

    fn parameters(&self) -> Self::ParameterIterator<'_, P> {
        let mut sorted_entries = self.iter().map(|(k, v)| (k.clone(), v)).collect::<Vec<_>>();
        sorted_entries.sort_unstable_by(|(l, _), (r, _)| l.cmp(r));
        sorted_entries.into_iter().flat_map(|(_, v)| V::parameters(v))
    }

    fn parameters_mut(&mut self) -> Self::ParameterIteratorMut<'_, P> {
        // Note that each pointer below originates from a distinct `iter_mut()` item, and so the pointers hereare unique
        // and non-overlapping. We do not structurally modify the map after collecting the pointers and so they remain
        // valid for the duration of this traversal.
        let mut sorted_entries = self.iter_mut().map(|(k, v)| (k.clone(), v as *mut V)).collect::<Vec<_>>();
        sorted_entries.sort_unstable_by(|(l, _), (r, _)| l.cmp(r));
        sorted_entries.into_iter().flat_map(|(_, v)| V::parameters_mut(unsafe { &mut *v }))
    }

    fn into_parameters(self) -> Self::ParameterIntoIterator<P> {
        let mut sorted_entries = self.into_iter().collect::<Vec<_>>();
        sorted_entries.sort_unstable_by(|(l, _), (r, _)| l.cmp(r));
        sorted_entries.into_iter().flat_map(|(_, v)| V::into_parameters(v))
    }

    fn named_parameters(&self) -> Self::NamedParameterIterator<'_, P> {
        let mut sorted_entries = self.iter().collect::<Vec<_>>();
        sorted_entries.sort_unstable_by_key(|(l, _)| *l);
        sorted_entries.into_iter().flat_map(|(key, value)| {
            PathPrefixedParameterIterator::new(value.named_parameters(), ParameterPathSegment::Key(format!("{key:?}")))
        })
    }

    fn named_parameters_mut(&mut self) -> Self::NamedParameterIteratorMut<'_, P> {
        // Note that each pointer below originates from a distinct `iter_mut()` item, and so the pointers hereare unique
        // and non-overlapping. We do not structurally modify the map after collecting the pointers and so they remain
        // valid for the duration of this traversal.
        let mut sorted_entries = self.iter_mut().map(|(k, v)| (k.clone(), v as *mut V)).collect::<Vec<_>>();
        sorted_entries.sort_unstable_by(|(l, _), (r, _)| l.cmp(r));
        sorted_entries.into_iter().flat_map(|(key, value)| {
            PathPrefixedParameterIterator::new(
                (unsafe { &mut *value }).named_parameters_mut(),
                ParameterPathSegment::Key(format!("{key:?}")),
            )
        })
    }

    fn into_named_parameters(self) -> Self::NamedParameterIntoIterator<P> {
        let mut sorted_entries = self.into_iter().collect::<Vec<_>>();
        sorted_entries.sort_unstable_by(|(l, _), (r, _)| l.cmp(r));
        sorted_entries.into_iter().flat_map(|(key, value)| {
            PathPrefixedParameterIterator::new(
                value.into_named_parameters(),
                ParameterPathSegment::Key(format!("{key:?}")),
            )
        })
    }

    fn paths(&self) -> Vec<ParameterPath> {
        let mut sorted_entries = self.iter().collect::<Vec<_>>();
        sorted_entries.sort_unstable_by_key(|(l, _)| *l);
        std::iter::once(ParameterPath::root())
            .chain(sorted_entries.into_iter().flat_map(|(key, value)| {
                let paths = value.paths().into_iter().map(|path| (path, ()));
                PathPrefixedParameterIterator::new(paths, ParameterPathSegment::Key(format!("{key:?}")))
                    .map(|(path, _)| path)
            }))
            .collect()
    }

    fn from_parameters_with_remainder<I: Iterator<Item = P>>(
        structure: Self::ParameterStructure,
        parameters: &mut I,
    ) -> Result<Self, ParameterError> {
        let expected_count = structure.len();
        let mut values = HashMap::with_capacity_and_hasher(expected_count, structure.hasher().clone());
        let mut sorted_entries = structure.into_iter().collect::<Vec<_>>();
        sorted_entries.sort_unstable_by(|(l, _), (r, _)| l.cmp(r));
        for (key, value_structure) in sorted_entries {
            values.insert(
                key,
                V::from_parameters_with_remainder(value_structure, parameters).map_err(|error| match error {
                    ParameterError::MissingParameters { paths, .. } => {
                        ParameterError::MissingParameters { expected_count, paths }
                    }
                    error => error,
                })?,
            );
        }
        Ok(values)
    }
}

pub struct BTreeMapParameterizedFamily<K, F>(PhantomData<(K, F)>);

impl<P: Parameter, K: Clone + Debug + Ord, F: ParameterizedFamily<P> + ParameterizedFamily<Placeholder>>
    ParameterizedFamily<P> for BTreeMapParameterizedFamily<K, F>
{
    type To = BTreeMap<K, <F as ParameterizedFamily<P>>::To>;
}

impl<P: Parameter, K: Clone + Debug + Ord, V: Parameterized<P>> Parameterized<P> for BTreeMap<K, V> {
    type Family = BTreeMapParameterizedFamily<K, V::Family>;

    type To<T: Parameter>
        = <Self::Family as ParameterizedFamily<T>>::To
    where
        Self::Family: ParameterizedFamily<T>;

    type ParameterStructure = Self::To<Placeholder>;

    type ParameterIterator<'t, T: 't + Parameter>
        = std::iter::FlatMap<
        std::collections::btree_map::Values<'t, K, V>,
        <V as Parameterized<P>>::ParameterIterator<'t, T>,
        fn(&'t V) -> <V as Parameterized<P>>::ParameterIterator<'t, T>,
    >
    where
        Self: 't;

    type ParameterIteratorMut<'t, T: 't + Parameter>
        = std::iter::FlatMap<
        std::collections::btree_map::ValuesMut<'t, K, V>,
        <V as Parameterized<P>>::ParameterIteratorMut<'t, T>,
        fn(&'t mut V) -> <V as Parameterized<P>>::ParameterIteratorMut<'t, T>,
    >
    where
        Self: 't;

    type ParameterIntoIterator<T: Parameter> = std::iter::FlatMap<
        std::collections::btree_map::IntoValues<K, V>,
        <V as Parameterized<P>>::ParameterIntoIterator<T>,
        fn(V) -> <V as Parameterized<P>>::ParameterIntoIterator<T>,
    >;

    type NamedParameterIterator<'t, T: 't + Parameter>
        = std::iter::FlatMap<
        std::collections::btree_map::Iter<'t, K, V>,
        PathPrefixedParameterIterator<&'t T, <V as Parameterized<P>>::NamedParameterIterator<'t, T>>,
        fn(
            (&'t K, &'t V),
        ) -> PathPrefixedParameterIterator<&'t T, <V as Parameterized<P>>::NamedParameterIterator<'t, T>>,
    >
    where
        Self: 't;

    type NamedParameterIteratorMut<'t, T: 't + Parameter>
        = std::iter::FlatMap<
        std::collections::btree_map::IterMut<'t, K, V>,
        PathPrefixedParameterIterator<&'t mut T, <V as Parameterized<P>>::NamedParameterIteratorMut<'t, T>>,
        fn(
            (&'t K, &'t mut V),
        )
            -> PathPrefixedParameterIterator<&'t mut T, <V as Parameterized<P>>::NamedParameterIteratorMut<'t, T>>,
    >
    where
        Self: 't;

    type NamedParameterIntoIterator<T: Parameter> = std::iter::FlatMap<
        std::collections::btree_map::IntoIter<K, V>,
        PathPrefixedParameterIterator<T, <V as Parameterized<P>>::NamedParameterIntoIterator<T>>,
        fn((K, V)) -> PathPrefixedParameterIterator<T, <V as Parameterized<P>>::NamedParameterIntoIterator<T>>,
    >;

    fn parameter_count(&self) -> usize {
        self.values().map(|value| value.parameter_count()).sum()
    }

    fn parameter_structure(&self) -> Self::ParameterStructure {
        BTreeMap::from_iter(self.iter().map(|(key, value)| (key.clone(), value.parameter_structure())))
    }

    fn parameters(&self) -> Self::ParameterIterator<'_, P> {
        self.values().flat_map(V::parameters)
    }

    fn parameters_mut(&mut self) -> Self::ParameterIteratorMut<'_, P> {
        self.values_mut().flat_map(V::parameters_mut)
    }

    fn into_parameters(self) -> Self::ParameterIntoIterator<P> {
        self.into_values().flat_map(V::into_parameters)
    }

    fn named_parameters(&self) -> Self::NamedParameterIterator<'_, P> {
        self.iter().flat_map(|(key, value)| {
            PathPrefixedParameterIterator::new(value.named_parameters(), ParameterPathSegment::Key(format!("{key:?}")))
        })
    }

    fn named_parameters_mut(&mut self) -> Self::NamedParameterIteratorMut<'_, P> {
        self.iter_mut().flat_map(|(key, value)| {
            PathPrefixedParameterIterator::new(
                value.named_parameters_mut(),
                ParameterPathSegment::Key(format!("{key:?}")),
            )
        })
    }

    fn into_named_parameters(self) -> Self::NamedParameterIntoIterator<P> {
        self.into_iter().flat_map(|(key, value)| {
            PathPrefixedParameterIterator::new(
                value.into_named_parameters(),
                ParameterPathSegment::Key(format!("{key:?}")),
            )
        })
    }

    fn paths(&self) -> Vec<ParameterPath> {
        std::iter::once(ParameterPath::root())
            .chain(self.iter().flat_map(|(key, value)| {
                let paths = value.paths().into_iter().map(|path| (path, ()));
                PathPrefixedParameterIterator::new(paths, ParameterPathSegment::Key(format!("{key:?}")))
                    .map(|(path, _)| path)
            }))
            .collect()
    }

    fn from_parameters_with_remainder<I: Iterator<Item = P>>(
        structure: Self::ParameterStructure,
        parameters: &mut I,
    ) -> Result<Self, ParameterError> {
        let expected_count = structure.len();
        let mut values = BTreeMap::new();
        for (key, value_structure) in structure {
            values.insert(
                key,
                V::from_parameters_with_remainder(value_structure, parameters).map_err(|error| match error {
                    ParameterError::MissingParameters { paths, .. } => {
                        ParameterError::MissingParameters { expected_count, paths }
                    }
                    error => error,
                })?,
            );
        }
        Ok(values)
    }
}

/// Parameterwise view of a [`Parameterized`] structure `S` over parameters of type `P`, which applies capabilities and
/// arithmetic to every parameter of the structure.
///
/// [`Parameterized`] is responsible for traversal and reconstruction only. Wrapping a structure in a [`Parameterwise`]
/// selects parameterwise semantics explicitly: [`map`](Self::map) applies a function to every parameter, and
/// [`zip_map`](Self::zip_map) applies a function to the corresponding parameters of two structures. Any leaf function
/// can be applied this way, including user-defined ones, without registering it anywhere. On top of these, the wrapper
/// implements the elementwise capabilities (e.g., [`Add`](crate::Add), [`Tanh`](crate::Tanh), etc.) whenever `P`
/// implements them, together with the panicking `std::ops` operators `Add`, `Sub`, `Mul`, `Div`, and `Neg`. Those
/// operators combine two wrappers, or a wrapper with a single parameter that is broadcast to every parameter on the
/// right. They delegate to the corresponding operators of `P`, so that staged tracers keep their deferred error
/// reporting. A parameter cannot appear on the left (e.g., `rate * model`) because
/// `impl<P> std::ops::Mul<Parameterwise<P, S>> for P` would violate the orphan rules (E0210). The wrapper belongs to
/// the [`Capability`](crate::Capability) universe of `P`, so that default universe parameters resolve exactly as they
/// do for `P`.
///
/// Two structures correspond when they have the same sequence of [`ParameterPath`]s (refer to
/// [`Parameterized::parameter_paths`]), which mirrors the structure-equality rule of JAX's
/// [`jax.tree.map`](https://docs.jax.dev/en/latest/_autosummary/jax.tree.map.html) without its static auxiliary data:
/// non-parameter (i.e., static) fields are not compared, and outputs take them from the left input. Equal parameter
/// counts alone are not sufficient, but differences that parameter paths cannot observe (e.g., static metadata) are
/// not detected.
///
/// [`Parameterwise`] owns its structure, and converting a structure into it (e.g., through [`From`] or [`Into`])
/// or back (through [`into_inner`](Self::into_inner)) neither traverses nor clones it. Functions that accept
/// `impl Into<Parameterwise<P, S>>` therefore accept both plain and already-wrapped structures.
///
/// # Example
///
/// ```rust
/// # use ryft_core::{Array, Parameterwise};
/// let model = vec![Array::scalar(1.0f32).unwrap(), Array::scalar(2.0f32).unwrap()];
/// let gradients = vec![Array::scalar(10.0f32).unwrap(), Array::scalar(20.0f32).unwrap()];
/// let learning_rate = Array::scalar(0.1f32).unwrap();
/// let model = Parameterwise::from(model) - Parameterwise::from(gradients) * learning_rate;
/// assert_eq!(model.into_inner(), vec![Array::scalar(0.0f32).unwrap(), Array::scalar(0.0f32).unwrap()]);
/// ```
pub struct Parameterwise<P, S> {
    /// Wrapped parameterized value.
    value: S,

    /// Marker for the parameter type that selects how [`Self::value`] is traversed.
    parameter: PhantomData<fn() -> P>,
}

impl<P, S> Parameterwise<P, S> {
    /// Wraps `value`, selecting `P` as the parameter type through which it is traversed.
    #[inline]
    pub fn new(value: S) -> Self {
        Self { value, parameter: PhantomData }
    }

    /// Returns a reference to the wrapped structure.
    #[inline]
    pub fn as_inner(&self) -> &S {
        &self.value
    }

    /// Consumes this [`Parameterwise`] and returns the wrapped structure.
    #[inline]
    pub fn into_inner(self) -> S {
        self.value
    }
}

impl<P: Parameter, S: Parameterized<P>> Parameterwise<P, S> {
    /// Applies `function` to every parameter of this structure in parameter order, and returns the structure that
    /// holds the outputs, with the static fields of this structure. Stops at the first error that `function` returns.
    /// A structure without parameters is reconstructed without calling `function`.
    pub fn map<F: FnMut(&P) -> Result<P, ProgramError>>(&self, function: F) -> Result<Self, ProgramError> {
        let parameters = self.value.parameters().map(function).collect::<Result<Vec<_>, _>>()?;
        Ok(Self::new(S::from_parameters(self.value.parameter_structure(), parameters)?))
    }

    /// Applies `function` to the corresponding parameters of this structure and `right` in parameter order, and returns
    /// the structure that holds the outputs, with the static fields of this structure. Stops at the first error that
    /// `function` returns.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError::InvalidArgument`] without calling `function` if the two structures do not have the
    /// same parameter paths.
    pub fn zip_map<F: FnMut(&P, &P) -> Result<P, ProgramError>>(
        &self,
        right: &Self,
        mut function: F,
    ) -> Result<Self, ProgramError> {
        if !self.value.parameter_paths().eq(right.value.parameter_paths()) {
            return Err(ProgramError::InvalidArgument {
                message: "binary parameterwise operation inputs must have the same parameter structure".to_string(),
            });
        }
        let parameters = self
            .value
            .parameters()
            .zip(right.value.parameters())
            .map(|(left, right)| function(left, right))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::new(S::from_parameters(self.value.parameter_structure(), parameters)?))
    }
}

impl<P, S: Clone> Clone for Parameterwise<P, S> {
    #[inline]
    fn clone(&self) -> Self {
        Self::new(self.value.clone())
    }
}

impl<P, S: Debug> Debug for Parameterwise<P, S> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_tuple("Parameterwise").field(&self.value).finish()
    }
}

impl<P, S: PartialEq> PartialEq for Parameterwise<P, S> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<P: Parameter, S: Parameterized<P>> From<S> for Parameterwise<P, S> {
    #[inline]
    fn from(value: S) -> Self {
        Self::new(value)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};

    use ryft_macros::Parameterized;

    use super::*;

    #[derive(Clone, Debug, PartialEq, Eq)]
    enum TestMappingError {
        Parameter(ParameterError),
        RejectedParameter(i32),
        RejectedNamedParameter { path: String, parameter: i32 },
    }

    impl From<ParameterError> for TestMappingError {
        fn from(error: ParameterError) -> Self {
            Self::Parameter(error)
        }
    }

    /// Map key whose [`Debug`] text hides its value, so that distinct keys produce the same [`ParameterPath`].
    #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    struct RedactedKey(u32);

    impl Debug for RedactedKey {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "[private]")
        }
    }

    #[test]
    fn test_placeholder() {
        let placeholder = Placeholder;
        assert_eq!(format!("{placeholder}"), "<Parameter>");
        assert_eq!(format!("{placeholder:?}"), "<Parameter>");
    }

    #[test]
    fn test_parameter_path() {
        let path = ParameterPath::root();
        assert_eq!(path.len(), 0);
        assert!(path.is_empty());
        assert!(path.is_root());
        let path_prefix = ParameterPath::root().variant("ResidualConnection").field("weights");
        let path = path_prefix.clone().index(2).tuple_index(1).key("alpha");
        assert_eq!(path_prefix.len(), 2);
        assert_eq!(path.len(), 5);
        assert!(!path.is_root());
        assert!(path.is_prefix_of(&path));
        assert!(path_prefix.is_prefix_of(&path));
        assert!(!path.is_prefix_of(&path_prefix));
        assert_eq!(
            path.segments().cloned().collect::<Vec<_>>(),
            vec![
                ParameterPathSegment::Variant("ResidualConnection"),
                ParameterPathSegment::Field("weights"),
                ParameterPathSegment::Index(2),
                ParameterPathSegment::TupleIndex(1),
                ParameterPathSegment::Key(format!("{:?}", "alpha")),
            ],
        );
        assert_eq!(format!("{path}"), "$.residual_connection.weights[2].1[\"alpha\"]");
        assert_eq!(format!("{path:?}"), "ParameterPath[$.residual_connection.weights[2].1[\"alpha\"]]");
    }

    #[test]
    fn test_parameterized() {
        #[derive(Parameterized, Clone, Debug, PartialEq, Eq)]
        struct Block<P: Parameter> {
            pair: (P, usize),
            gain: P,
            name: &'static str,
        }

        #[allow(dead_code)]
        #[derive(Parameterized, Clone, Debug, PartialEq, Eq)]
        enum Controller<P: Parameter> {
            Identity,
            Blend { alpha: P, branch: Block<P> },
        }

        #[derive(Parameterized, Clone, Debug, PartialEq, Eq)]
        struct Layer<P: Parameter> {
            weights: Vec<P>,
            bias: P,
            tag: &'static str,
        }

        #[derive(Parameterized, Clone, Debug, PartialEq, Eq)]
        struct Network<P: Parameter> {
            stem: Layer<P>,
            controller: Controller<P>,
            heads: [Block<P>; 2],
            metadata: (usize, &'static str),
        }

        let value = Network {
            stem: Layer { weights: vec![1, 2], bias: 3, tag: "stem" },
            controller: Controller::Blend { alpha: 4, branch: Block { pair: (5, 10), gain: 6, name: "branch" } },
            heads: [
                Block { pair: (7, 11), gain: 8, name: "head_0" },
                Block { pair: (9, 12), gain: 10, name: "head_1" },
            ],
            metadata: (42, "meta"),
        };

        let expected_parameter_paths = vec![
            "$.stem.weights[0]".to_string(),
            "$.stem.weights[1]".to_string(),
            "$.stem.bias".to_string(),
            "$.controller.blend.alpha".to_string(),
            "$.controller.blend.branch.pair.0".to_string(),
            "$.controller.blend.branch.gain".to_string(),
            "$.heads[0].pair.0".to_string(),
            "$.heads[0].gain".to_string(),
            "$.heads[1].pair.0".to_string(),
            "$.heads[1].gain".to_string(),
        ];

        // Test `Parameterized::parameter_count`.
        assert_eq!(value.parameter_count(), 10);

        // Test `Parameterized::parameter_structure`.
        let structure = value.parameter_structure();
        assert_eq!(
            structure,
            Network {
                stem: Layer { weights: vec![Placeholder, Placeholder], bias: Placeholder, tag: "stem" },
                controller: Controller::Blend {
                    alpha: Placeholder,
                    branch: Block { pair: (Placeholder, 10), gain: Placeholder, name: "branch" },
                },
                heads: [
                    Block { pair: (Placeholder, 11), gain: Placeholder, name: "head_0" },
                    Block { pair: (Placeholder, 12), gain: Placeholder, name: "head_1" },
                ],
                metadata: (42, "meta"),
            },
        );

        // Test `Parameterized::parameters`.
        assert_eq!(value.parameters().copied().collect::<Vec<_>>(), vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        // Test `Parameterized::parameters_mut`.
        let mut value_clone = value.clone();
        for parameter in value_clone.parameters_mut() {
            *parameter += 100;
        }
        assert_eq!(
            value_clone.parameters().copied().collect::<Vec<_>>(),
            vec![101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
        );

        // Test `Parameterized::into_parameters`.
        assert_eq!(value.clone().into_parameters().collect::<Vec<_>>(), vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        // Test `Parameterized::named_parameters`.
        assert_eq!(
            value.named_parameters().map(|(path, parameter)| (path.to_string(), *parameter)).collect::<Vec<_>>(),
            expected_parameter_paths
                .iter()
                .cloned()
                .zip(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                .collect::<Vec<_>>()
        );

        // Test `Parameterized::named_parameters_mut`.
        let mut value_clone = value.clone();
        let mut seen_parameter_paths = Vec::new();
        for (path, parameter) in value_clone.named_parameters_mut() {
            seen_parameter_paths.push(path.to_string());
            *parameter += path.len() as i32;
        }
        assert_eq!(seen_parameter_paths, expected_parameter_paths);
        assert_eq!(value_clone.parameters().copied().collect::<Vec<_>>(), vec![4, 5, 5, 7, 10, 10, 11, 11, 13, 13]);

        // Test `Parameterized::into_named_parameters`.
        assert_eq!(
            value
                .clone()
                .into_named_parameters()
                .map(|(path, parameter)| (path.to_string(), parameter))
                .collect::<Vec<_>>(),
            vec![
                ("$.stem.weights[0]".to_string(), 1),
                ("$.stem.weights[1]".to_string(), 2),
                ("$.stem.bias".to_string(), 3),
                ("$.controller.blend.alpha".to_string(), 4),
                ("$.controller.blend.branch.pair.0".to_string(), 5),
                ("$.controller.blend.branch.gain".to_string(), 6),
                ("$.heads[0].pair.0".to_string(), 7),
                ("$.heads[0].gain".to_string(), 8),
                ("$.heads[1].pair.0".to_string(), 9),
                ("$.heads[1].gain".to_string(), 10),
            ]
        );

        // Test `Parameterized::parameter_paths`.
        assert_eq!(value.parameter_paths().map(|path| path.to_string()).collect::<Vec<_>>(), expected_parameter_paths);

        // Test `Parameterized::paths`.
        assert_eq!(
            value.paths().iter().map(ParameterPath::to_string).collect::<Vec<_>>(),
            vec![
                "$",
                "$.stem",
                "$.stem.weights",
                "$.stem.weights[0]",
                "$.stem.weights[1]",
                "$.stem.bias",
                "$.controller",
                "$.controller.blend",
                "$.controller.blend.alpha",
                "$.controller.blend.branch",
                "$.controller.blend.branch.pair",
                "$.controller.blend.branch.pair.0",
                "$.controller.blend.branch.gain",
                "$.heads",
                "$.heads[0]",
                "$.heads[0].pair",
                "$.heads[0].pair.0",
                "$.heads[0].gain",
                "$.heads[1]",
                "$.heads[1].pair",
                "$.heads[1].pair.0",
                "$.heads[1].gain",
            ],
        );
        let value_with_empty_containers = Network {
            stem: Layer { weights: Vec::new(), bias: 3, tag: "stem" },
            controller: Controller::Identity,
            ..value.clone()
        };
        assert_eq!(
            value_with_empty_containers.paths().iter().map(ParameterPath::to_string).collect::<Vec<_>>(),
            vec![
                "$",
                "$.stem",
                "$.stem.weights",
                "$.stem.bias",
                "$.controller",
                "$.controller.identity",
                "$.heads",
                "$.heads[0]",
                "$.heads[0].pair",
                "$.heads[0].pair.0",
                "$.heads[0].gain",
                "$.heads[1]",
                "$.heads[1].pair",
                "$.heads[1].pair.0",
                "$.heads[1].gain",
            ],
        );

        // Test `Parameterized::from_parameters_with_remainder`.
        let mut parameters = vec![11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 500].into_iter();
        assert_eq!(
            Network::from_parameters_with_remainder(structure.clone(), &mut parameters),
            Ok(Network {
                stem: Layer { weights: vec![11, 12], bias: 13, tag: "stem" },
                controller: Controller::Blend { alpha: 14, branch: Block { pair: (15, 10), gain: 16, name: "branch" } },
                heads: [
                    Block { pair: (17, 11), gain: 18, name: "head_0" },
                    Block { pair: (19, 12), gain: 20, name: "head_1" },
                ],
                metadata: (42, "meta"),
            })
        );
        assert_eq!(parameters.collect::<Vec<_>>(), vec![500]);

        // Test `Parameterized::from_parameters`.
        assert_eq!(
            Network::from_parameters(structure.clone(), vec![21, 22, 23, 24, 25, 26, 27, 28, 29, 30]),
            Ok(Network {
                stem: Layer { weights: vec![21, 22], bias: 23, tag: "stem" },
                controller: Controller::Blend { alpha: 24, branch: Block { pair: (25, 10), gain: 26, name: "branch" } },
                heads: [
                    Block { pair: (27, 11), gain: 28, name: "head_0" },
                    Block { pair: (29, 12), gain: 30, name: "head_1" },
                ],
                metadata: (42, "meta"),
            })
        );
        assert_eq!(
            Network::from_parameters(structure.clone(), vec![31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 999]),
            Err(ParameterError::UnusedParameters { paths: None })
        );

        // Test `Parameterized::from_named_parameters`.
        let mut parameters = value.clone().into_named_parameters().collect::<Vec<_>>();
        parameters.reverse();
        assert_eq!(Network::from_named_parameters(structure.clone(), parameters), Ok(value.clone()));

        // Test `Parameterized::from_broadcasted_named_parameters`.
        assert_eq!(
            Network::from_broadcasted_named_parameters(
                structure.clone(),
                HashMap::from([
                    (ParameterPath::root(), 1),
                    (ParameterPath::root().field("controller"), 2),
                    (ParameterPath::root().field("controller").variant("Blend").field("branch"), 10),
                    (ParameterPath::root().field("controller").variant("Blend").field("branch").field("gain"), 20),
                    (ParameterPath::root().field("heads").index(1), 30),
                ]),
            ),
            Ok(Network {
                stem: Layer { weights: vec![1, 1], bias: 1, tag: "stem" },
                controller: Controller::Blend { alpha: 2, branch: Block { pair: (10, 10), gain: 20, name: "branch" } },
                heads: [
                    Block { pair: (1, 11), gain: 1, name: "head_0" },
                    Block { pair: (30, 12), gain: 30, name: "head_1" },
                ],
                metadata: (42, "meta"),
            })
        );

        // Test `Parameterized::broadcast_to_parameter_structure`.
        assert_eq!(
            42_i32.broadcast_to_parameter_structure::<(i32, (i32, i32))>((Placeholder, (Placeholder, Placeholder))),
            Ok((42, (42, 42))),
        );
        assert!(matches!(
            (1_i32, 2_i32).broadcast_to_parameter_structure::<i32>(Placeholder),
            Err(ParameterError::MissingParameters { .. }),
        ));

        // Test `Parameterized::try_map_parameters`.
        assert_eq!(
            value
                .clone()
                .try_map_parameters(|parameter| Ok::<_, TestMappingError>(i64::from(parameter) * 10))
                .unwrap()
                .into_parameters()
                .collect::<Vec<_>>(),
            vec![10i64, 20i64, 30i64, 40i64, 50i64, 60i64, 70i64, 80i64, 90i64, 100i64],
        );
        assert_eq!(
            value.clone().try_map_parameters::<i64, TestMappingError, _>(|parameter| {
                if parameter == 5 {
                    Err(TestMappingError::RejectedParameter(parameter))
                } else {
                    Ok(i64::from(parameter))
                }
            }),
            Err(TestMappingError::RejectedParameter(5)),
        );

        // Test `Parameterized::map_parameters`.
        assert_eq!(
            value.clone().map_parameters(|parameter| i64::from(parameter) * 10),
            Ok(Network {
                stem: Layer { weights: vec![10i64, 20i64], bias: 30i64, tag: "stem" },
                controller: Controller::Blend {
                    alpha: 40i64,
                    branch: Block { pair: (50i64, 10), gain: 60i64, name: "branch" },
                },
                heads: [
                    Block { pair: (70i64, 11), gain: 80i64, name: "head_0" },
                    Block { pair: (90i64, 12), gain: 100i64, name: "head_1" },
                ],
                metadata: (42, "meta"),
            })
        );

        // Test `Parameterized::try_map_named_parameters`.
        assert_eq!(
            value
                .clone()
                .try_map_named_parameters(|path, parameter| {
                    Ok::<_, TestMappingError>(i64::from(parameter) + (path.len() as i64))
                })
                .unwrap()
                .into_parameters()
                .collect::<Vec<_>>(),
            vec![4i64, 5i64, 5i64, 7i64, 10i64, 10i64, 11i64, 11i64, 13i64, 13i64],
        );
        assert_eq!(
            value.clone().try_map_named_parameters::<i64, TestMappingError, _>(|path, parameter| {
                if path.to_string() == "$.controller.blend.branch.pair.0" {
                    Err(TestMappingError::RejectedNamedParameter { path: path.to_string(), parameter })
                } else {
                    Ok(i64::from(parameter))
                }
            }),
            Err(TestMappingError::RejectedNamedParameter {
                path: "$.controller.blend.branch.pair.0".to_string(),
                parameter: 5,
            }),
        );

        // Test `Parameterized::map_named_parameters`.
        assert_eq!(
            value
                .clone()
                .map_named_parameters(|path, parameter| i64::from(parameter) + (path.len() as i64))
                .unwrap()
                .into_parameters()
                .collect::<Vec<_>>(),
            vec![4i64, 5i64, 5i64, 7i64, 10i64, 10i64, 11i64, 11i64, 13i64, 13i64],
        );

        // Test `Parameterized::filter_parameters`.
        assert_eq!(
            value
                .clone()
                .filter_parameters(|path, _| path.to_string().ends_with(".bias") || path.to_string().ends_with(".gain"))
                .unwrap()
                .into_parameters()
                .collect::<Vec<_>>(),
            vec![None, None, Some(3), None, None, Some(6), None, Some(8), None, Some(10)],
        );

        // Test `Parameterized::partition_parameters`.
        let (partition_0, partition_1) =
            value.clone().partition_parameters(|path, _| path.to_string().starts_with("$.heads[1]")).unwrap();
        assert_eq!(
            partition_0.clone().into_parameters().collect::<Vec<_>>(),
            vec![None, None, None, None, None, None, None, None, Some(9), Some(10)],
        );
        assert_eq!(
            partition_1.clone().into_parameters().collect::<Vec<_>>(),
            vec![Some(1), Some(2), Some(3), Some(4), Some(5), Some(6), Some(7), Some(8), None, None],
        );

        // Test `Parameterized::combine_parameters`.
        assert_eq!(Network::combine_parameters(structure.clone(), vec![partition_0, partition_1]), Ok(value.clone()));

        // Test `Parameterized::replace_parameters`.
        assert_eq!(
            value
                .clone()
                .replace_parameters(
                    value
                        .map_parameters(|parameter| match parameter {
                            2 => Some(200),
                            5 => Some(500),
                            7 => Some(700),
                            10 => Some(1000),
                            _ => None,
                        })
                        .unwrap()
                )
                .unwrap()
                .into_parameters()
                .collect::<Vec<_>>(),
            vec![1, 200, 3, 4, 500, 6, 700, 8, 9, 1000],
        );
    }

    #[test]
    fn test_parameterized_paths() {
        let root = ParameterPath::root;

        // Leaves, `PhantomData`, and the unit tuple have no children, and so they return only their own path.
        assert_eq!(7i32.paths(), vec![root()]);
        assert_eq!(PhantomData::<i32>.paths(), vec![root()]);
        assert_eq!(Parameterized::<i32>::paths(&()), vec![root()]);

        // Tuples return their own path followed by the node paths of their elements.
        assert_eq!((1i32, 2i32).paths(), vec![root(), root().tuple_index(0), root().tuple_index(1)]);
        assert_eq!(
            ((), 1i32, (Vec::<i32>::new(), [0i32; 0]), PhantomData::<i32>).paths(),
            vec![
                root(),
                root().tuple_index(0),
                root().tuple_index(1),
                root().tuple_index(2),
                root().tuple_index(2).tuple_index(0),
                root().tuple_index(2).tuple_index(1),
                root().tuple_index(3),
            ],
        );

        // Arrays return their own path followed by the node paths of their elements.
        assert_eq!([0i32; 0].paths(), vec![root()]);
        assert_eq!([1i32, 2i32].paths(), vec![root(), root().index(0), root().index(1)]);
        assert_eq!(
            [Vec::new(), vec![1i32], Vec::new()].paths(),
            vec![root(), root().index(0), root().index(1), root().index(1).index(0), root().index(2)],
        );

        // Vectors return their own path followed by the node paths of their elements.
        assert_eq!(Vec::<i32>::new().paths(), vec![root()]);
        assert_eq!(
            vec![(1i32, 2i32)].paths(),
            vec![root(), root().index(0), root().index(0).tuple_index(0), root().index(0).tuple_index(1)],
        );
        assert_eq!(vec![Vec::<i32>::new()].paths(), vec![root(), root().index(0)]);
        assert_eq!(
            vec![vec![vec![1i32]], vec![Vec::new(), vec![2i32]], Vec::new()].paths(),
            vec![
                root(),
                root().index(0),
                root().index(0).index(0),
                root().index(0).index(0).index(0),
                root().index(1),
                root().index(1).index(0),
                root().index(1).index(1),
                root().index(1).index(1).index(0),
                root().index(2),
            ],
        );

        // Hash maps return their own path followed by the node paths of their values, in the order of their keys.
        assert_eq!(HashMap::<&str, i32>::new().paths(), vec![root()]);
        assert_eq!(
            HashMap::from([("zeta", Vec::new()), ("alpha", Vec::new()), ("mu", vec![1i32])]).paths(),
            vec![root(), root().key("alpha"), root().key("mu"), root().key("mu").index(0), root().key("zeta")],
        );

        // B-tree maps return their own path followed by the node paths of their values, each with the segment of its
        // key on the root side of the path.
        assert_eq!(BTreeMap::<&str, i32>::new().paths(), vec![root()]);
        assert_eq!(
            BTreeMap::from([("left", 1i32), ("right", 2i32)]).paths(),
            vec![root(), root().key("left"), root().key("right")],
        );
        assert_eq!(
            BTreeMap::from([("left", (1i32, Vec::new())), ("right", (2i32, vec![3i32]))]).paths(),
            vec![
                root(),
                root().key("left"),
                root().key("left").tuple_index(0),
                root().key("left").tuple_index(1),
                root().key("right"),
                root().key("right").tuple_index(0),
                root().key("right").tuple_index(1),
                root().key("right").tuple_index(1).index(0),
            ],
        );
    }

    #[test]
    fn test_parameterized_validate_parameter_structure() {
        let expected_paths = (1, (2, 3)).parameter_paths().collect::<Vec<_>>();
        assert_eq!((4, (5, 6)).validate_parameter_structure(&expected_paths), Ok(()));

        // A value with as many parameters but a different structure is rejected.
        assert_eq!(
            ((4, 5), 6).validate_parameter_structure(&expected_paths),
            Err(ParameterError::MismatchedParameterStructures {
                left_structure: "[$.0, $.1.0, $.1.1]".to_string(),
                right_structure: "[$.0.0, $.0.1, $.1]".to_string(),
            }),
        );
    }

    #[test]
    fn test_parameterized_from_named_parameters_with_duplicate_paths() {
        let root = ParameterPath::root;

        // Two leaves of the structure with the same path would receive the provided parameters in provided order.
        assert_eq!(
            BTreeMap::<RedactedKey, i32>::from_named_parameters(
                BTreeMap::from([(RedactedKey(3), Placeholder), (RedactedKey(7), Placeholder)]),
                vec![(root().key(RedactedKey(7)), 2), (root().key(RedactedKey(3)), 1)],
            ),
            Err(ParameterError::DuplicateParameterPaths { paths: vec!["$[[private]]".to_string()] }),
        );

        // Two provided parameters with the same path are rejected, whether or not they arrive in structure order.
        assert_eq!(
            <(i32, i32)>::from_named_parameters(
                (Placeholder, Placeholder),
                vec![(root().tuple_index(0), 1), (root().tuple_index(1), 2), (root().tuple_index(0), 3)],
            ),
            Err(ParameterError::DuplicateParameterPaths { paths: vec!["$.0".to_string()] }),
        );
        assert_eq!(
            <(i32, i32)>::from_named_parameters(
                (Placeholder, Placeholder),
                vec![(root().tuple_index(1), 2), (root().tuple_index(0), 1), (root().tuple_index(1), 3)],
            ),
            Err(ParameterError::DuplicateParameterPaths { paths: vec!["$.1".to_string()] }),
        );
    }

    #[test]
    fn test_parameterized_from_broadcasted_named_parameters() {
        let root = ParameterPath::root;

        // Prefix leaves resolve leaves to the most specific matching prefix leaf.
        assert_eq!(
            <(i32, (i32, i32))>::from_broadcasted_named_parameters(
                (Placeholder, (Placeholder, Placeholder)),
                vec![(root(), 1), (root().tuple_index(1).tuple_index(0), 2)],
            ),
            Ok((1, (2, 1))),
        );

        // Prefix leaves may cover only containers without leaves, in which case they populate no leaf.
        assert_eq!(
            <(Vec<i32>, i32)>::from_broadcasted_named_parameters(
                (Vec::new(), Placeholder),
                vec![(root().tuple_index(0), 1), (root().tuple_index(1), 2)],
            ),
            Ok((Vec::new(), 2)),
        );
        assert_eq!(
            <((), ()) as Parameterized<i32>>::from_broadcasted_named_parameters(
                ((), ()),
                vec![(root().tuple_index(0), 1), (root().tuple_index(1), 2)],
            ),
            Ok(((), ())),
        );
        assert_eq!(
            <(Vec<i32>, i32)>::from_broadcasted_named_parameters((Vec::new(), Placeholder), vec![(root(), 1)]),
            Ok((Vec::new(), 1)),
        );
        assert_eq!(Vec::<i32>::from_broadcasted_named_parameters(Vec::new(), vec![(root(), 1)]), Ok(Vec::new()));

        // Leaves of the structure without a prefix leaf are missing.
        assert_eq!(
            <(Vec<i32>, i32)>::from_broadcasted_named_parameters(
                (Vec::new(), Placeholder),
                vec![(root().tuple_index(0), 1)],
            ),
            Err(ParameterError::MissingParameters { expected_count: 1, paths: Some(vec!["$.1".to_string()]) }),
        );
        assert_eq!(
            Vec::<Vec<i32>>::from_broadcasted_named_parameters(vec![vec![Placeholder]], Vec::new()),
            Err(ParameterError::MissingParameters { expected_count: 1, paths: Some(vec!["$[0][0]".to_string()]) }),
        );

        // Prefix leaves whose paths are not node paths of the structure are unused, including prefix leaves below a
        // container without children of the structure.
        assert_eq!(
            <(i32, i32)>::from_broadcasted_named_parameters(
                (Placeholder, Placeholder),
                vec![(root(), 1), (root().tuple_index(2), 2)],
            ),
            Err(ParameterError::UnusedParameters { paths: Some(vec!["$.2".to_string()]) }),
        );
        assert_eq!(
            Vec::<i32>::from_broadcasted_named_parameters(Vec::new(), vec![(root().index(0), 1)]),
            Err(ParameterError::UnusedParameters { paths: Some(vec!["$[0]".to_string()]) }),
        );
        assert_eq!(
            Vec::<i32>::from_broadcasted_named_parameters(Vec::new(), vec![(root().index(0).index(0), 1)]),
            Err(ParameterError::UnusedParameters { paths: Some(vec!["$[0][0]".to_string()]) }),
        );

        // The paths of the prefix leaves stand in for the nodes of the prefix tree, and so the nodes of the structure
        // without leaves are not checked, whereas `Parameterized::broadcast_to_parameter_structure` rejects a prefix
        // tree that lacks one of them.
        let prefix = vec![vec![1_i32]];
        let structure = vec![vec![Placeholder], Vec::new()];
        assert_eq!(
            Vec::<Vec<i32>>::from_broadcasted_named_parameters(
                structure.clone(),
                prefix.clone().into_named_parameters(),
            ),
            Ok(vec![vec![1], Vec::new()]),
        );
        assert_eq!(
            prefix.broadcast_to_parameter_structure::<Vec<Vec<i32>>>(structure),
            Err(ParameterError::MissingParameters { expected_count: 1, paths: Some(vec!["$[1]".to_string()]) }),
        );
    }

    #[test]
    fn test_parameterized_from_broadcasted_named_parameters_with_duplicate_paths() {
        let structure = BTreeMap::from([(RedactedKey(3), Placeholder), (RedactedKey(7), Placeholder)]);
        let duplicate_error = ParameterError::DuplicateParameterPaths { paths: vec!["$[[private]]".to_string()] };
        assert_eq!(
            duplicate_error.to_string(),
            "got multiple nodes with the same parameter path; distinct map keys must have distinct `Debug` texts; \
             duplicate parameter paths: `$[[private]]`",
        );

        // Prefix nodes with the same path cannot be told apart (e.g., an option that is requested only for the second
        // of two keys with the same `Debug` text), and so they are rejected, including nodes without leaves.
        assert_eq!(
            BTreeMap::<RedactedKey, Option<i32>>::from_broadcasted_named_parameters(
                structure.clone(),
                BTreeMap::from([(RedactedKey(3), None), (RedactedKey(7), Some(1))]).into_named_parameters(),
            ),
            Err(duplicate_error.clone()),
        );
        assert_eq!(
            BTreeMap::from([(RedactedKey(3), None), (RedactedKey(7), Some(1))])
                .broadcast_to_parameter_structure::<BTreeMap<RedactedKey, Option<i32>>>(structure.clone()),
            Err(duplicate_error.clone()),
        );
        assert_eq!(
            BTreeMap::from([(RedactedKey(3), Vec::<i32>::new()), (RedactedKey(7), Vec::new())])
                .broadcast_to_parameter_structure::<BTreeMap<RedactedKey, Vec<i32>>>(BTreeMap::from([
                    (RedactedKey(3), Vec::new()),
                    (RedactedKey(7), Vec::new()),
                ])),
            Err(duplicate_error.clone()),
        );

        // Structure nodes with the same path are rejected when the prefix tree reaches their path with its own nodes.
        assert_eq!(
            BTreeMap::from([(RedactedKey(3), 1)])
                .broadcast_to_parameter_structure::<BTreeMap<RedactedKey, i32>>(structure.clone()),
            Err(duplicate_error),
        );

        // Structure nodes with the same path below a prefix leaf all receive the value of that leaf.
        assert_eq!(
            5_i32.broadcast_to_parameter_structure::<BTreeMap<RedactedKey, i32>>(structure),
            Ok(BTreeMap::from([(RedactedKey(3), 5), (RedactedKey(7), 5)])),
        );
        assert_eq!(
            (1_i32, 2_i32).broadcast_to_parameter_structure::<(BTreeMap<RedactedKey, (i32, i32)>, i32)>((
                BTreeMap::from([
                    (RedactedKey(3), (Placeholder, Placeholder)),
                    (RedactedKey(7), (Placeholder, Placeholder)),
                ]),
                Placeholder,
            )),
            Ok((BTreeMap::from([(RedactedKey(3), (1, 1)), (RedactedKey(7), (1, 1))]), 2)),
        );
    }

    #[test]
    fn test_parameterized_broadcast_to_parameter_structure() {
        #[derive(Parameterized, Clone, Debug, PartialEq, Eq)]
        struct Left<P: Parameter> {
            value: P,
            tag: &'static str,
        }

        #[derive(Parameterized, Clone, Debug, PartialEq, Eq)]
        struct Right<P: Parameter> {
            value: P,
        }

        // Prefix leaves broadcast over all the nodes below them, including the containers without leaves.
        assert_eq!(
            42_i32.broadcast_to_parameter_structure::<(i32, (i32, i32))>((Placeholder, (Placeholder, Placeholder))),
            Ok((42, (42, 42)))
        );
        assert_eq!(
            vec![1_i32, 2_i32]
                .broadcast_to_parameter_structure::<Vec<Vec<i32>>>(vec![vec![Placeholder, Placeholder], Vec::new()]),
            Ok(vec![vec![1, 1], Vec::new()]),
        );
        assert_eq!(
            (Vec::<i32>::new(), 1_i32).broadcast_to_parameter_structure::<(Vec<i32>, i32)>((Vec::new(), Placeholder)),
            Ok((Vec::new(), 1)),
        );
        assert_eq!(Vec::<i32>::new().broadcast_to_parameter_structure::<Vec<i32>>(Vec::new()), Ok(Vec::new()));
        assert_eq!(
            (1_i32, 2_i32).broadcast_to_parameter_structure::<i32>(Placeholder),
            Err(ParameterError::MissingParameters { expected_count: 1, paths: Some(vec!["$".to_string()]) }),
        );

        // Every node of the prefix must be a node of the structure, and every node of the structure must be a node of
        // the prefix or lie at or below a prefix leaf.
        assert_eq!(
            vec![vec![1_i32]].broadcast_to_parameter_structure::<Vec<Vec<i32>>>(vec![vec![Placeholder], Vec::new()]),
            Err(ParameterError::MissingParameters { expected_count: 1, paths: Some(vec!["$[1]".to_string()]) }),
        );
        assert_eq!(
            vec![vec![1_i32], Vec::new()].broadcast_to_parameter_structure::<Vec<Vec<i32>>>(vec![vec![Placeholder]]),
            Err(ParameterError::UnusedParameters { paths: Some(vec!["$[1]".to_string()]) }),
        );
        assert_eq!(
            vec![Vec::<i32>::new()].broadcast_to_parameter_structure::<Vec<Vec<i32>>>(Vec::new()),
            Err(ParameterError::UnusedParameters { paths: Some(vec!["$[0]".to_string()]) }),
        );
        assert_eq!(
            Vec::<i32>::new().broadcast_to_parameter_structure::<Vec<i32>>(vec![Placeholder]),
            Err(ParameterError::MissingParameters { expected_count: 1, paths: Some(vec!["$[0]".to_string()]) }),
        );

        // The errors describe missing and unused nodes by their paths, including containers without leaves in both
        // trees, and list only the deepest of them.
        let missing_error =
            ParameterError::MissingParameters { expected_count: 0, paths: Some(vec!["$[0]".to_string()]) };
        assert_eq!(
            Vec::<Vec<i32>>::new().broadcast_to_parameter_structure::<Vec<Vec<i32>>>(vec![Vec::new()]),
            Err(missing_error.clone()),
        );
        assert_eq!(missing_error.to_string(), "missing parameter paths: `$[0]`");
        assert_eq!(
            (1_i32, Vec::<i32>::new(), Vec::<i32>::new())
                .broadcast_to_parameter_structure::<(Vec<i32>, Vec<i32>)>((Vec::new(), Vec::new())),
            Err(ParameterError::UnusedParameters { paths: Some(vec!["$.2".to_string()]) }),
        );
        assert_eq!(
            vec![vec![1_i32]].broadcast_to_parameter_structure::<Vec<i32>>(Vec::new()),
            Err(ParameterError::UnusedParameters { paths: Some(vec!["$[0][0]".to_string()]) }),
        );
        assert_eq!(
            ParameterError::UnusedParameters { paths: Some(vec!["$[0]".to_string()]) }.to_string(),
            "unused parameter paths: `$[0]`",
        );

        // Missing paths are listed in the order of the nodes of the structure, and they take precedence over unused
        // paths.
        assert_eq!(
            Right { value: 1_i32 }.broadcast_to_parameter_structure::<(i32, Vec<i32>)>((Placeholder, Vec::new())),
            Err(ParameterError::MissingParameters {
                expected_count: 1,
                paths: Some(vec!["$.0".to_string(), "$.1".to_string()]),
            }),
        );

        // Matching is by path, so containers of different types with the same paths match, and so do nodes of the
        // same type with different static metadata.
        assert_eq!(
            [1_i32, 2_i32].broadcast_to_parameter_structure::<Vec<i32>>(vec![Placeholder, Placeholder]),
            Ok(vec![1, 2])
        );
        assert_eq!(
            Right { value: 1_i32 }
                .broadcast_to_parameter_structure::<Left<i32>>(Left { value: Placeholder, tag: "left" }),
            Ok(Left { value: 1, tag: "left" }),
        );
        assert_eq!(
            Left { value: 1_i32, tag: "source" }
                .broadcast_to_parameter_structure::<Left<i32>>(Left { value: Placeholder, tag: "target" }),
            Ok(Left { value: 1, tag: "target" }),
        );
        assert_eq!(
            vec![Vec::<i32>::new()].broadcast_to_parameter_structure::<[Vec<i32>; 1]>([Vec::new()]),
            Ok([Vec::new()]),
        );
    }

    #[test]
    fn test_parameterized_try_map_named_parameters_with_duplicate_paths() {
        // Mapping keeps the position of every parameter, and so parameters with the same path stay apart, which also
        // holds for the functions that are built on it.
        let value = BTreeMap::from([(RedactedKey(3), 1_i32), (RedactedKey(7), 2_i32)]);
        assert_eq!(
            value.clone().try_map_named_parameters(|path, parameter| {
                Ok::<_, TestMappingError>(format!("{path}") == "$[[private]]" && parameter == 2)
            }),
            Ok(BTreeMap::from([(RedactedKey(3), false), (RedactedKey(7), true)])),
        );
        assert_eq!(
            value.clone().map_named_parameters(|_, parameter| i64::from(parameter) * 10),
            Ok(BTreeMap::from([(RedactedKey(3), 10_i64), (RedactedKey(7), 20_i64)])),
        );
        assert_eq!(
            value.filter_parameters(|_, parameter| *parameter == 2),
            Ok(BTreeMap::from([(RedactedKey(3), None), (RedactedKey(7), Some(2))])),
        );
    }

    #[test]
    fn test_parameterized_parameter() {
        let value = 7i32;
        assert_eq!(value.parameter_count(), 1);
        assert_eq!(value.parameter_structure(), Placeholder);
        assert_eq!(value.parameters().copied().collect::<Vec<_>>(), vec![7]);
        assert_eq!(
            value.named_parameters().map(|(path, parameter)| (path.to_string(), *parameter)).collect::<Vec<_>>(),
            vec![("$".to_string(), 7)],
        );
        assert_eq!(value.parameter_paths().map(|path| path.to_string()).collect::<Vec<_>>(), vec!["$".to_string()]);
    }

    #[test]
    fn test_parameterized_tuple() {
        let value = (0i32, (1i32, 2i32), (3i32, (4i32, 5i32)), 6i32, ((7i32, 8i32), 9i32));
        assert_eq!(value.parameter_count(), 10);
        assert_eq!(
            value.parameter_structure(),
            (
                Placeholder,
                (Placeholder, Placeholder),
                (Placeholder, (Placeholder, Placeholder)),
                Placeholder,
                ((Placeholder, Placeholder), Placeholder),
            ),
        );
        assert_eq!(value.parameters().copied().collect::<Vec<_>>(), vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(
            value.named_parameters().map(|(path, parameter)| (path.to_string(), *parameter)).collect::<Vec<_>>(),
            vec![
                ("$.0".to_string(), 0),
                ("$.1.0".to_string(), 1),
                ("$.1.1".to_string(), 2),
                ("$.2.0".to_string(), 3),
                ("$.2.1.0".to_string(), 4),
                ("$.2.1.1".to_string(), 5),
                ("$.3".to_string(), 6),
                ("$.4.0.0".to_string(), 7),
                ("$.4.0.1".to_string(), 8),
                ("$.4.1".to_string(), 9),
            ],
        );
        assert_eq!(
            value.parameter_paths().map(|path| path.to_string()).collect::<Vec<_>>(),
            vec![
                "$.0".to_string(),
                "$.1.0".to_string(),
                "$.1.1".to_string(),
                "$.2.0".to_string(),
                "$.2.1.0".to_string(),
                "$.2.1.1".to_string(),
                "$.3".to_string(),
                "$.4.0.0".to_string(),
                "$.4.0.1".to_string(),
                "$.4.1".to_string(),
            ],
        );
    }

    #[test]
    fn test_parameterized_array() {
        let value = [(1, 2), (3, 4), (5, 6)];
        assert_eq!(value.parameter_count(), 6);
        assert_eq!(
            value.parameter_structure(),
            [(Placeholder, Placeholder), (Placeholder, Placeholder), (Placeholder, Placeholder)],
        );
        assert_eq!(value.parameters().copied().collect::<Vec<_>>(), vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(
            value.named_parameters().map(|(path, parameter)| (path.to_string(), *parameter)).collect::<Vec<_>>(),
            vec![
                ("$[0].0".to_string(), 1),
                ("$[0].1".to_string(), 2),
                ("$[1].0".to_string(), 3),
                ("$[1].1".to_string(), 4),
                ("$[2].0".to_string(), 5),
                ("$[2].1".to_string(), 6),
            ],
        );
        assert_eq!(
            value.parameter_paths().map(|path| path.to_string()).collect::<Vec<_>>(),
            vec![
                "$[0].0".to_string(),
                "$[0].1".to_string(),
                "$[1].0".to_string(),
                "$[1].1".to_string(),
                "$[2].0".to_string(),
                "$[2].1".to_string(),
            ],
        );
    }

    #[test]
    fn test_parameterized_vec() {
        let value = vec![(1, 2), (3, 4)];
        assert_eq!(value.parameter_count(), 4);
        assert_eq!(value.parameter_structure(), vec![(Placeholder, Placeholder), (Placeholder, Placeholder)]);
        assert_eq!(value.parameters().copied().collect::<Vec<_>>(), vec![1, 2, 3, 4]);
        assert_eq!(
            value.named_parameters().map(|(path, parameter)| (path.to_string(), *parameter)).collect::<Vec<_>>(),
            vec![
                ("$[0].0".to_string(), 1),
                ("$[0].1".to_string(), 2),
                ("$[1].0".to_string(), 3),
                ("$[1].1".to_string(), 4),
            ],
        );
        assert_eq!(
            value.parameter_paths().map(|path| path.to_string()).collect::<Vec<_>>(),
            vec!["$[0].0".to_string(), "$[0].1".to_string(), "$[1].0".to_string(), "$[1].1".to_string()],
        );
    }

    #[test]
    fn test_parameterized_hash_map() {
        let value = HashMap::from([("zeta", (1, 2)), ("alpha", (3, 4)), ("mu", (5, 6))]);
        assert_eq!(value.parameter_count(), 6);
        assert_eq!(
            value.parameter_structure(),
            HashMap::from([
                ("alpha", (Placeholder, Placeholder)),
                ("mu", (Placeholder, Placeholder)),
                ("zeta", (Placeholder, Placeholder)),
            ]),
        );
        assert_eq!(value.parameters().copied().collect::<Vec<_>>(), vec![3, 4, 5, 6, 1, 2]);
        assert_eq!(
            value.named_parameters().map(|(path, parameter)| (path.to_string(), *parameter)).collect::<Vec<_>>(),
            vec![
                ("$[\"alpha\"].0".to_string(), 3),
                ("$[\"alpha\"].1".to_string(), 4),
                ("$[\"mu\"].0".to_string(), 5),
                ("$[\"mu\"].1".to_string(), 6),
                ("$[\"zeta\"].0".to_string(), 1),
                ("$[\"zeta\"].1".to_string(), 2),
            ],
        );
        assert_eq!(
            value.parameter_paths().map(|path| path.to_string()).collect::<Vec<_>>(),
            vec![
                "$[\"alpha\"].0".to_string(),
                "$[\"alpha\"].1".to_string(),
                "$[\"mu\"].0".to_string(),
                "$[\"mu\"].1".to_string(),
                "$[\"zeta\"].0".to_string(),
                "$[\"zeta\"].1".to_string(),
            ],
        );
    }

    #[test]
    fn test_parameterized_b_tree_map() {
        let value = BTreeMap::from([("left", (1, 2)), ("right", (3, 4))]);
        assert_eq!(value.parameter_count(), 4);
        assert_eq!(
            value.parameter_structure(),
            BTreeMap::from([("left", (Placeholder, Placeholder)), ("right", (Placeholder, Placeholder))]),
        );
        assert_eq!(value.parameters().copied().collect::<Vec<_>>(), vec![1, 2, 3, 4]);
        assert_eq!(
            value.named_parameters().map(|(path, parameter)| (path.to_string(), *parameter)).collect::<Vec<_>>(),
            vec![
                ("$[\"left\"].0".to_string(), 1),
                ("$[\"left\"].1".to_string(), 2),
                ("$[\"right\"].0".to_string(), 3),
                ("$[\"right\"].1".to_string(), 4),
            ],
        );
        assert_eq!(
            value.parameter_paths().map(|path| path.to_string()).collect::<Vec<_>>(),
            vec![
                "$[\"left\"].0".to_string(),
                "$[\"left\"].1".to_string(),
                "$[\"right\"].0".to_string(),
                "$[\"right\"].1".to_string(),
            ],
        );
    }

    #[test]
    fn test_parameterwise() {
        // Wrapping and unwrapping neither traverse nor clone the structure.
        let wrapped = Parameterwise::<i32, _>::new(vec![1i32, 2]);
        assert_eq!(wrapped.as_inner(), &vec![1, 2]);
        assert_eq!(wrapped.clone().into_inner(), vec![1, 2]);
        assert_eq!(format!("{wrapped:?}"), "Parameterwise([1, 2])");
        assert_eq!(wrapped, Parameterwise::from(vec![1i32, 2]));
        assert_ne!(wrapped, Parameterwise::from(vec![1i32, 3]));

        // `Into` accepts both plain and already-wrapped structures at function boundaries.
        fn sum<L: Into<Parameterwise<i32, Vec<i32>>>, R: Into<Parameterwise<i32, Vec<i32>>>>(
            left: L,
            right: R,
        ) -> Vec<i32> {
            (left.into() + right.into()).into_inner()
        }

        assert_eq!(sum(vec![1, 2], vec![10, 20]), vec![11, 22]);
        assert_eq!(sum(vec![1, 2], Parameterwise::from(vec![10, 20])), vec![11, 22]);
        assert_eq!(sum(Parameterwise::from(vec![1, 2]), vec![10, 20]), vec![11, 22]);
        assert_eq!(sum(Parameterwise::from(vec![1, 2]), Parameterwise::from(vec![10, 20])), vec![11, 22]);
    }

    #[test]
    fn test_parameterwise_map() {
        #[derive(Clone, Debug, PartialEq, Parameterized)]
        enum Branch<P: Parameter> {
            Leaf(P),
            Pair { left: P, right: Option<P> },
        }

        #[derive(Clone, Debug, PartialEq, Parameterized)]
        struct Model<P: Parameter> {
            weights: Vec<P>,
            scales: (P, Option<P>),
            branches: BTreeMap<String, Branch<P>>,
            name: &'static str,
        }

        let model = Model {
            weights: vec![1i32, 2],
            scales: (3, None),
            branches: BTreeMap::from([
                ("a".to_string(), Branch::Leaf(4)),
                ("b".to_string(), Branch::Pair { left: 5, right: Some(6) }),
            ]),
            name: "model",
        };

        // Mapping preserves the container shape, enum variants, optional fields, and static fields.
        assert_eq!(
            Parameterwise::from(model.clone())
                .map(|parameter| Ok(parameter * 10))
                .map(Parameterwise::into_inner),
            Ok(Model {
                weights: vec![10, 20],
                scales: (30, None),
                branches: BTreeMap::from([
                    ("a".to_string(), Branch::Leaf(40)),
                    ("b".to_string(), Branch::Pair { left: 50, right: Some(60) }),
                ]),
                name: "model",
            }),
        );

        // A single leaf is its own structure, and an empty structure is rebuilt without calling the function.
        assert_eq!(Parameterwise::<i32, i32>::from(7).map(|parameter| Ok(parameter + 1)), Ok(Parameterwise::new(8)));
        assert_eq!(
            Parameterwise::<i32, Vec<i32>>::from(Vec::new()).map(|_| panic!("the function must not be called")),
            Ok(Parameterwise::new(Vec::new())),
        );

        // Mapping stops at the first error.
        let mut visited = Vec::new();
        assert_eq!(
            Parameterwise::from(model).map(|parameter| {
                visited.push(*parameter);
                if *parameter == 2 {
                    Err(ProgramError::InvalidArgument { message: "rejected parameter".to_string() })
                } else {
                    Ok(*parameter)
                }
            }),
            Err(ProgramError::InvalidArgument { message: "rejected parameter".to_string() }),
        );
        assert_eq!(visited, vec![1, 2]);

        // Neither the leaves nor the container need to implement `Clone`.
        #[derive(Debug, PartialEq)]
        struct Leaf(i32);

        impl Parameter for Leaf {}

        assert_eq!(
            Parameterwise::<Leaf, _>::from(vec![Leaf(1), Leaf(2)])
                .map(|parameter| Ok(Leaf(-parameter.0)))
                .map(Parameterwise::into_inner),
            Ok(vec![Leaf(-1), Leaf(-2)]),
        );
    }

    #[test]
    fn test_parameterwise_zip_map() {
        #[derive(Clone, Debug, PartialEq, Parameterized)]
        enum Branch<P: Parameter> {
            First(P),
            Second(P),
        }

        #[derive(Clone, Debug, PartialEq, Parameterized)]
        struct Model<P: Parameter> {
            weights: Vec<P>,
            bias: Option<P>,
            branch: Branch<P>,
            name: &'static str,
        }

        let model = |weights: Vec<i32>, bias: Option<i32>, branch: Branch<i32>, name: &'static str| {
            Parameterwise::from(Model { weights, bias, branch, name })
        };
        let left = model(vec![1, 2], Some(3), Branch::First(4), "left");

        // Corresponding parameters are combined in parameter order, and static fields come from the left input.
        assert_eq!(
            left.zip_map(&model(vec![10, 20], Some(30), Branch::First(40), "right"), |left, right| Ok(left + right)),
            Ok(model(vec![11, 22], Some(33), Branch::First(44), "left")),
        );

        // Structures with different parameter paths are rejected before the function is called, including structures
        // with different optional fields, different parameter counts, and equal counts but different enum variants.
        let mismatch = Err(ProgramError::InvalidArgument {
            message: "binary parameterwise operation inputs must have the same parameter structure".to_string(),
        });
        let reject = |_: &i32, _: &i32| -> Result<i32, ProgramError> { panic!("the function must not be called") };
        assert_eq!(left.zip_map(&model(vec![10, 20], None, Branch::First(40), "left"), reject), mismatch);
        assert_eq!(left.zip_map(&model(vec![10], Some(30), Branch::First(40), "left"), reject), mismatch);
        assert_eq!(left.zip_map(&model(vec![10, 20], Some(30), Branch::Second(40), "left"), reject), mismatch);

        // Combining stops at the first error.
        let mut visited = Vec::new();
        assert_eq!(
            left.zip_map(&left, |left, right| {
                visited.push((*left, *right));
                Err(ProgramError::InvalidArgument { message: "rejected parameter".to_string() })
            }),
            Err(ProgramError::InvalidArgument { message: "rejected parameter".to_string() }),
        );
        assert_eq!(visited, vec![(1, 1)]);
    }
}
