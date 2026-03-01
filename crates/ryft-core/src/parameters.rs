use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;

use convert_case::{Case, Casing};
use half::{bf16, f16};
use paste::paste;

use crate::errors::Error;

/// Helper trait used to encode type equality constraints in the associated type bounds of [`Parameterized`].
/// A type `X` implements [`SameAs<Y>`] only when `X` and `Y` are the exact same type.
pub trait SameAs<T> {}

impl<T> SameAs<T> for T {}

/// Marker trait for leaf parameter values in a [`Parameterized`] tree. This trait is intentionally empty. A type
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

/// Placeholder leaf type for [`Parameterized`] trees that is used represent [`Parameterized::parameter_structure`].
/// That is, it is used to replace every nested parameter in a [`Parameterized`] type yielding a _structure-only_
/// representation that can later be used with [`Parameterized::from_parameters`] to instantiate a [`Parameterized`]
/// value with the same shape but different types of parameters.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Placeholder;

impl Display for Placeholder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<Parameter>")
    }
}

impl Debug for Placeholder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<Parameter>")
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

    /// [`Debug`]-formatted key of a [`HashMap`] entry.
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
    /// # use ryft_core::parameters::{ParameterPath, ParameterPathSegment};
    /// let path = ParameterPath::root()
    ///     .with_segment(ParameterPathSegment::Field("weights"))
    ///     .with_segment(ParameterPathSegment::Index(1))
    ///     .with_segment(ParameterPathSegment::TupleIndex(0));
    /// assert_eq!(path.to_string(), "$.weights[1].0");
    /// ```
    pub fn with_segment(mut self, segment: ParameterPathSegment) -> Self {
        self.segments.insert(0, segment);
        self
    }
}

impl Display for ParameterPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "$")?;
        for segment in self.segments() {
            match segment {
                ParameterPathSegment::Field(name) => write!(f, ".{name}")?,
                ParameterPathSegment::Variant(name) => write!(f, ".{}", name.to_case(Case::Snake))?,
                ParameterPathSegment::TupleIndex(index) => write!(f, ".{index}")?,
                ParameterPathSegment::Index(index) => write!(f, "[{index}]")?,
                ParameterPathSegment::Key(key) => write!(f, "[{key}]")?,
            }
        }
        Ok(())
    }
}

impl Debug for ParameterPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ParameterPath[{self}]")
    }
}

/// Type-level family that maps [`Parameter`] types nested in a type while preserving its overall
/// structure. This is used internally by [`Parameterized`] and is based on the _type family_ approach described in
/// [this blog post](https://smallcultfollowing.com/babysteps/blog/2016/11/03/associated-type-constructors-part-2-family-traits/).
/// This trait is generic over `P` (instead of using a non-generic family trait with `type To<P: Parameter>`) so
/// that each family can constrain the parameter domain at the `impl` level. For example, a family can implement
/// `ParameterizedFamily<P>` only for `P: Parameter + Clone`. With a generic associated type `To<P>` on a non-generic
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
///  1. Their _structure_, which can be obtained via [`Self::parameter_structure`].
///  2. Their _parameter_ values, which can be obtained via [`Self::parameters`], [`Self::parameters_mut`],
///     [`Self::into_parameters`], [`Self::named_parameters`], [`Self::named_parameters_mut`],
///     and [`Self::into_named_parameters`].
///
/// In the context of machine learning (ML), a [`Parameterized`] can contain model parameters (thus the name), dataset
/// entries, reinforcement learning agent observations, etc. Ryft provides built-in [`Parameterized`] implementations
/// for a wide range of container-like types, including but not limited to:
///
/// TODO(eaplatanios): List of supported types.
///
/// Furthermore, Ryft provides a convenient `#[derive(Parameterized)]` macro that can be used to automatically derive
/// [`Parameterized`] implementations for custom types. Refer to the [`Custom Types`](#custom-types) section below for
/// more information on that macro.
///
/// The [`Parameterized`] type and the functionality it provides is inspired by
/// [JAX PyTrees](https://docs.jax.dev/en/latest/pytrees.html#working-with-pytrees)
/// and [Equinox's PyTree manipulation APIs](https://docs.kidger.site/equinox/api/manipulation/).
///
/// ## Examples
///
/// The following are simple examples showing what [`Parameterized`] types are and how they are structured:
///
/// ```rust
/// # use std::collections::HashMap;
/// # use ryft_core::parameters::Parameterized;
///
/// // Simple tuple with 3 [`Parameter`]s.
/// let value = (1, 2, 3);
/// let parameters = value.parameters().collect::<Vec<_>>();
/// assert_eq!(value.parameter_count(), parameters.len());
/// assert_eq!(parameters, vec![&1, &2, &3]);
/// // TODO(eaplatanios): Show the parameter structure.
///
/// // Nested tuple structure with 3 [`Parameter`]s.
/// let value = (1, (2, 3), ());
/// let parameters = value.parameters().collect::<Vec<_>>();
/// assert_eq!(value.parameter_count(), parameters.len());
/// assert_eq!(parameters, vec![&1, &2, &3]);
/// // TODO(eaplatanios): Show the parameter structure.
///
/// // Nested map and tuple structure with 5 [`Parameter`]s.
/// let value = (1, HashMap::from([("a", vec![2]), ("b", vec![3, 4])]), (5,));
/// let parameters = value.parameters().collect::<Vec<_>>();
/// assert_eq!(value.parameter_count(), parameters.len());
/// assert_eq!(parameters, vec![&1, &2, &3, &4, &5]);
/// // TODO(eaplatanios): Show the parameter structure.
/// ```
///
/// # Custom Types
///
/// TODO(eaplatanios): Introduce section about the derive macro and include examples.
///
///
///
///
///
///
///
///
///
///
/// Any nested container whose leaves implement [`Parameter`] can be traversed as a [`Parameterized`] tree.
/// A single [`Parameter`] leaf is also a valid tree.
///
/// ```rust
/// # use ryft_core::parameters::Parameterized;
/// let value = vec![(1_i32, 2_i32), (3_i32, 4_i32)];
/// assert_eq!(value.parameter_count(), 4);
/// assert_eq!(value.parameters().copied().collect::<Vec<_>>(), vec![1, 2, 3, 4]);
/// ```
///
/// # Common Parameterized Functions
///
/// The most commonly used operations are flattening, rebuilding, and mapping:
///
/// - Flatten leaves: [`parameters`](Self::parameters) together with
///   [`parameter_structure`](Self::parameter_structure).
/// - Rebuild from leaves: [`from_parameters`](Self::from_parameters) or
///   [`from_parameters_with_remainder`](Self::from_parameters_with_remainder).
/// - Map leaves: [`map_parameters`](Self::map_parameters) or [`map_named_parameters`](Self::map_named_parameters).
///
/// In direct JAX terms:
///
/// - `jax.tree.flatten(x)` corresponds to [`parameters`](Self::parameters) +
///   [`parameter_structure`](Self::parameter_structure).
/// - `jax.tree.unflatten(treedef, leaves)` corresponds to [`from_parameters`](Self::from_parameters).
/// - `jax.tree.map(f, x)` corresponds to [`map_parameters`](Self::map_parameters).
///
/// ## Common function: `map_parameters`
///
/// ```rust
/// # use ryft_core::parameters::Parameterized;
/// let list_of_lists = vec![vec![1_i32, 2, 3], vec![1, 2], vec![1, 2, 3, 4]];
/// let doubled = list_of_lists.map_parameters(|value| value * 2)?;
/// assert_eq!(doubled, vec![vec![2, 4, 6], vec![2, 4], vec![2, 4, 6, 8]]);
/// # Ok::<(), ryft_core::errors::Error>(())
/// ```
///
/// ## Common function: `map_named_parameters`
///
/// ```rust
/// # use ryft_core::parameters::Parameterized;
/// let value = vec![(1_i32, 2_i32), (3_i32, 4_i32)];
/// let scaled = value.map_named_parameters(|path, parameter| {
///     if path.to_string().ends_with(".1") { parameter * 10 } else { parameter }
/// })?;
/// assert_eq!(scaled, vec![(1, 20), (3, 40)]);
/// # Ok::<(), ryft_core::errors::Error>(())
/// ```
///
/// # Viewing The Parameterized Definition Of A Value
///
/// [`parameter_structure`](Self::parameter_structure) returns the same tree shape with all leaves replaced by
/// [`Placeholder`] values. This is analogous to inspecting a pytree definition (`treedef`) for debugging.
///
/// ```rust
/// # use ryft_core::parameters::{Parameterized, Placeholder};
/// let value = vec![(1.0_f32, 2.0_f32), (3.0_f32, 4.0_f32)];
/// let structure = value.parameter_structure();
/// assert_eq!(structure, vec![(Placeholder, Placeholder), (Placeholder, Placeholder)]);
/// assert_eq!(structure.parameter_count(), 4);
///
/// let rebuilt = <Vec<(f32, f32)> as Parameterized<f32>>::from_parameters(
///     structure,
///     vec![10.0_f32, 20.0_f32, 30.0_f32, 40.0_f32],
/// )?;
/// assert_eq!(rebuilt, vec![(10.0, 20.0), (30.0, 40.0)]);
/// # Ok::<(), ryft_core::errors::Error>(())
/// ```
///
/// # Parameterized Trees In Ryft Workflows
///
/// Similar to how JAX transformations and APIs consume pytrees, many `ryft` utilities operate on parameterized trees:
///
/// - [`from_named_parameters`](Self::from_named_parameters) rebuilds from path-indexed leaves.
/// - [`from_broadcasted_named_parameters`](Self::from_broadcasted_named_parameters) applies prefix-based matching
///   (longest-prefix-wins), analogous to tree-prefix matching patterns in JAX APIs.
/// - [`filter_parameters`](Self::filter_parameters), [`partition_parameters`](Self::partition_parameters),
///   [`combine_optional_parameters`](Self::combine_optional_parameters), and
///   [`apply_parameter_updates`](Self::apply_parameter_updates) mirror the common Equinox manipulation flow.
///
/// # Explicit Parameter Paths
///
/// Like JAX key paths, each leaf has a deterministic structural path represented by [`ParameterPath`].
///
/// ```rust
/// # use ryft_core::parameters::Parameterized;
/// let value = vec![(1_i32, 2_i32)];
/// let paths = value.parameter_paths().map(|path| path.to_string()).collect::<Vec<_>>();
/// assert_eq!(paths, vec!["$[0].0".to_string(), "$[0].1".to_string()]);
/// ```
///
/// ```rust
/// # use ryft_core::parameters::Parameterized;
/// let value = vec![(1_i32, 2_i32)];
/// let named = value
///     .named_parameters()
///     .map(|(path, parameter)| (path.to_string(), *parameter))
///     .collect::<Vec<_>>();
/// assert_eq!(named, vec![("$[0].0".to_string(), 1), ("$[0].1".to_string(), 2),]);
/// ```
///
/// # Common Parameterized Gotchas
///
/// ## Mistaking container nodes for leaves
///
/// Traversal happens at leaf granularity. Container nodes (e.g., tuples, vectors, maps) are traversed recursively
/// and are not passed to mapping closures as a whole.
///
/// ```rust
/// # use ryft_core::parameters::Parameterized;
/// let value = vec![(2_i32, 3_i32)];
/// let mapped = value.map_parameters(|leaf| leaf * 10)?;
/// assert_eq!(mapped, vec![(20, 30)]);
/// # Ok::<(), ryft_core::errors::Error>(())
/// ```
///
/// ## Handling `Option` and `None`
///
/// Unlike JAX's default treatment of `None` (absence of a node), `Option<P>` is a leaf parameter type in `ryft`
/// whenever `P: Parameter`. This is frequently useful for optional updates/masks.
///
/// ```rust
/// # use ryft_core::parameters::Parameterized;
/// let value = vec![Some(1_i32), None, Some(3_i32)];
/// assert_eq!(value.parameter_count(), 3);
/// assert_eq!(value.parameters().copied().collect::<Vec<_>>(), vec![Some(1), None, Some(3)]);
/// ```
///
/// ## Rebuilding from named paths is strict
///
/// [`from_named_parameters`](Self::from_named_parameters) requires exact path coverage (no missing or extra paths).
/// Use [`from_broadcasted_named_parameters`](Self::from_broadcasted_named_parameters) when you want prefix-based
/// assignment behavior instead.
///
/// # Common Parameterized Patterns
///
/// ## Partition, modify, and combine
///
/// ```rust
/// # use ryft_core::parameters::Parameterized;
/// let value = vec![(1_i32, 2_i32), (3_i32, 4_i32)];
/// let structure = value.parameter_structure();
/// let (selected, rejected) = value.partition_parameters(|path, _| path.to_string().starts_with("$[0]"))?;
/// let recombined = <Vec<(i32, i32)> as Parameterized<i32>>::combine_optional_parameters(
///     structure,
///     vec![selected, rejected],
/// )?;
/// assert_eq!(recombined, vec![(1, 2), (3, 4)]);
/// # Ok::<(), ryft_core::errors::Error>(())
/// ```
///
/// ## Tree surgery with selectors
///
/// ```rust
/// # use ryft_core::parameters::Parameterized;
/// let value = vec![(1_i32, 2_i32), (3_i32, 4_i32)];
/// let updated = value.tree_at(
///     |structure| {
///         structure
///             .named_parameters()
///             .map(|(path, _)| path)
///             .filter(|path| path.to_string().ends_with(".0"))
///             .collect::<Vec<_>>()
///     },
///     vec![10_i32, 30_i32],
/// )?;
/// assert_eq!(updated, vec![(10, 2), (30, 4)]);
/// # Ok::<(), ryft_core::errors::Error>(())
/// ```
///
/// # Relationship To JAX And Equinox
///
/// - JAX `flatten` / `unflatten` / `map` correspond to [`parameters`](Self::parameters) +
///   [`parameter_structure`](Self::parameter_structure), [`from_parameters`](Self::from_parameters), and
///   [`map_parameters`](Self::map_parameters), respectively.
/// - Equinox `filter` / `partition` / `combine` / `apply_updates` / `tree_at` correspond to
///   [`filter_parameters`](Self::filter_parameters), [`partition_parameters`](Self::partition_parameters),
///   [`combine_optional_parameters`](Self::combine_optional_parameters),
///   [`apply_parameter_updates`](Self::apply_parameter_updates), and [`tree_at`](Self::tree_at) /
///   [`tree_at_paths`](Self::tree_at_paths), respectively.
///
/// # References
///
/// - JAX PyTrees: <https://docs.jax.dev/en/latest/pytrees.html>
/// - JAX Custom PyTrees: <https://docs.jax.dev/en/latest/custom_pytrees.html>
/// - Equinox PyTree Manipulation: <https://docs.kidger.site/equinox/api/manipulation/>
///
/// # Implementations Provided In This Module
///
/// - Every `P: Parameter` is a leaf and therefore implements [`Parameterized<P>`].
/// - [`PhantomData<P>`] implements [`Parameterized<P>`] and contributes zero parameters.
/// - Tuples whose elements are all themselves [`Parameterized`] are supported for arities of 1 through 12.
///   Tuples with mixed [`Parameterized`] and non-[`Parameterized`] elements are supposed only when they appear within
///   types for which we derive [`Parameterized`] implementations using the `#[derive(Parameterized)]` macro.
/// - Options are supported only when they appear within types for which we derive [`Parameterized`] implementations
///   using the `#[derive(Parameterized)]` macro.
/// - Arrays (`[T; N]`) and [`Vec<T>`] are supported when `T: Parameterized<P>`.
/// - [`HashMap<K, T, S>`] is supported when `K: Clone + Eq + Debug + Hash`, `S: BuildHasher + Clone`,
///   and `T: Parameterized<P>`.
/// - [`Box<T>`] is intentionally not supported (see the coherence note below).
///
/// Macro:
/// - Supports both structs and enums already.
/// - `#[derive(Parameterized)]` provides support for custom structs and enums, which also support nested tuples
///   that mix [Parameterized] and non-[Parameterized] fields. However, they can only be nested within other tuples.
///   If, for example, they appear in e.g., `Vec<(P, usize)>`, then those tuples are not supported.
/// - The parameter type must be a generic type parameter bounded by [Parameter].
/// - There must be only one such generic type parameter. Not zero and not more than one.
/// - All fields that reference / depend on the parameter type are considered parameter fields.
/// - Attributes of generic parameters are not visited/transformed and they are always carried around as they are.
/// - Configurable `macro_parameter_lifetime` and `macro_parameter_type`.
///
/// # Coherence Note For `Box<T>`
///
/// We cannot currently provide `impl<P: Parameter, V: Parameterized<P>> Parameterized<P> for Box<V>` because it
/// overlaps with the blanket leaf implementation `impl<P: Parameter> Parameterized<P> for P`. Since `Box` is a
/// fundamental type, downstream crates may implement [`Parameter`] for `Box<LocalType>`, and the generic `Box` impl
/// then becomes non-coherent under Rust's orphan/coherence rules.
///
/// # Ordering Invariant
///
/// Implementations must preserve leaf order consistently across traversal and reconstruction. In other words, reading
/// parameters with [`parameters`](Self::parameters) and then rebuilding with [`from_parameters`](Self::from_parameters)
/// must produce the original value.
pub trait Parameterized<P: Parameter>: Sized {
    /// [`ParameterizedFamily`] that this type belongs to and which can be used to reparameterize it.
    type Family: ParameterizedFamily<P, To = Self> + ParameterizedFamily<Placeholder, To = Self::ParameterStructure>;

    /// Reparameterized form of this [`Parameterized`] type with all of its nested `P` types replaced by `T`. This
    /// preserves the same [`Family`](Self::Family) and [`ParameterStructure`](Self::ParameterStructure), and is such that
    /// reparameterizing back to `P` recovers [`Self`].
    type To<T: Parameter>: Parameterized<T, Family = Self::Family, ParameterStructure = Self::ParameterStructure>
        + SameAs<<Self::Family as ParameterizedFamily<T>>::To>
    where
        Self::Family: ParameterizedFamily<T>;

    /// Shape-only representation of this [`Parameterized`] type with all parameter leaves replaced by [`Placeholder`].
    /// This is always set to `Self::To<Placeholder>`. The only reason this is not included here is that defaulted
    /// associated types are not supported in stable Rust.
    type ParameterStructure: Parameterized<Placeholder, Family = Self::Family, To<P> = Self>
        + SameAs<Self::To<Placeholder>>;

    /// Iterator returned by [`parameters`](Self::parameters) for a borrow of the underlying [`Parameter`]s with
    /// lifetime `'t`. This is an associated type instead of an `impl Iterator` in the corresponding function signature,
    /// so that implementations can expose and reuse a concrete iterator type. In particular, `#[derive(Parameterized)]`
    /// for enums synthesizes concrete enum iterators here, avoiding an additional heap allocation and dynamic dispatch.
    type ParameterIterator<'t, T: 't + Parameter>: 't + Iterator<Item = &'t T>
    where
        Self: 't;

    /// Iterator returned by [`parameters_mut`](Self::parameters_mut) for a mutable borrow of the underlying
    /// [`Parameter`]s with lifetime `'t`. Similar to [`ParameterIterator`](Self::ParameterIterator), this is an
    /// associated type instead of an `impl Iterator` in the corresponding function signature, so that implementations
    /// can expose and reuse a concrete iterator type, potentially avoiding additional heap allocations and dynamic
    /// dispatch.
    type ParameterIteratorMut<'t, T: 't + Parameter>: 't + Iterator<Item = &'t mut T>
    where
        Self: 't;

    /// Iterator returned by [`into_parameters`](Self::into_parameters), consuming `self` and returning the underlying
    /// [`Parameter`]s. Similar to [`ParameterIterator`](Self::ParameterIterator), this is an associated type instead of
    /// an `impl Iterator` in the corresponding function signature, so that implementations can expose and reuse
    /// a concrete iterator type, potentially avoiding additional heap allocations and dynamic dispatch.
    type ParameterIntoIterator<T: Parameter>: Iterator<Item = T>;

    /// Iterator returned by [`Self::named_parameters`], borrowing the underlying [`Parameter`]s and pairing them with
    /// their corresponding [`ParameterPath`]s.
    type NamedParameterIterator<'t, T: 't + Parameter>: 't + Iterator<Item = (ParameterPath, &'t T)>
    where
        Self: 't;

    /// Iterator returned by [`Self::named_parameters_mut`], mutably borrowing the underlying [`Parameter`]s and pairing
    /// them with their corresponding [`ParameterPath`]s.
    type NamedParameterIteratorMut<'t, T: 't + Parameter>: 't + Iterator<Item = (ParameterPath, &'t mut T)>
    where
        Self: 't;

    /// Iterator returned by [`Self::into_named_parameters`], consuming `self` and returning the underlying
    /// [`Parameter`]s together with their corresponding [`ParameterPath`]s.
    type NamedParameterIntoIterator<T: Parameter>: Iterator<Item = (ParameterPath, T)>;

    /// Returns the number of parameters in this [Parameterized] instance.
    fn parameter_count(&self) -> usize;

    /// Returns the parameter structure of this value by replacing all leaves with [`Placeholder`]s.
    fn parameter_structure(&self) -> Self::ParameterStructure;

    /// Returns an iterator over references to all parameters in this value.
    fn parameters(&self) -> Self::ParameterIterator<'_, P>;

    /// Returns an iterator over mutable references to all parameters in this value.
    fn parameters_mut(&mut self) -> Self::ParameterIteratorMut<'_, P>;

    /// Consumes this value and returns an iterator over all parameters.
    fn into_parameters(self) -> Self::ParameterIntoIterator<P>;

    /// Returns an iterator over all parameters in this value paired with their structural paths.
    fn named_parameters(&self) -> Self::NamedParameterIterator<'_, P>;

    /// Returns an iterator over mutable references to all parameters in this value paired with their structural paths.
    fn named_parameters_mut(&mut self) -> Self::NamedParameterIteratorMut<'_, P>;

    /// Consumes this value and returns an iterator over all parameters paired with their structural paths.
    fn into_named_parameters(self) -> Self::NamedParameterIntoIterator<P>;

    /// Returns an iterator over all structural paths to parameters in this value.
    ///
    /// Paths are returned in the same traversal order as [`named_parameters`](Self::named_parameters),
    /// [`named_parameters_mut`](Self::named_parameters_mut), and [`Self::into_named_parameters`].
    fn parameter_paths<'p>(&'p self) -> impl 'p + Iterator<Item = ParameterPath>
    where
        P: 'p,
    {
        self.named_parameters().map(|(path, _)| path)
    }

    /// Reconstructs a value from `structure`, consuming parameters from `parameters` and leaving any remainder
    /// untouched.
    fn from_parameters_with_remainder<I: Iterator<Item = P>>(
        structure: Self::ParameterStructure,
        parameters: &mut I,
    ) -> Result<Self, Error>;

    /// Reconstructs a value from `structure` using all provided parameters.
    ///
    /// Returns [`Error::UnusedParameters`] if there are leftover parameters.
    fn from_parameters<I: IntoIterator<Item = P>>(
        structure: Self::ParameterStructure,
        parameters: I,
    ) -> Result<Self, Error> {
        let mut parameters = parameters.into_iter();
        let parameterized = Self::from_parameters_with_remainder(structure, &mut parameters)?;
        parameters.next().map(|_| Err(Error::UnusedParameters)).unwrap_or_else(|| Ok(parameterized))
    }

    /// Reconstructs a value from `structure` using all provided named parameters.
    ///
    /// # Parameters
    ///
    ///   - `structure`: Parameter structure to reconstruct.
    ///   - `parameters`: Map from parameter paths to parameter values.
    ///
    /// The map is consumed and path order is not considered. Returns [`Error::InsufficientParameters`] when fewer than
    /// `structure.parameter_count()` expected parameters are provided, [`Error::NamedParameterPathMismatch`] if a
    /// required path is missing while other paths remain, and [`Error::UnusedParameters`] if extra paths remain after
    /// reconstruction.
    fn from_named_parameters(
        structure: Self::ParameterStructure,
        mut parameters: HashMap<ParameterPath, P>,
    ) -> Result<Self, Error> {
        let expected_count = structure.parameter_count();
        let mut values = Vec::with_capacity(expected_count);
        for (expected_path, _) in structure.named_parameters() {
            match parameters.remove(&expected_path) {
                Some(parameter) => values.push(parameter),
                None if parameters.is_empty() => return Err(Error::InsufficientParameters { expected_count }),
                None => {
                    let actual_path = parameters.keys().next().map(ToString::to_string).unwrap_or_default();
                    return Err(Error::NamedParameterPathMismatch {
                        expected_path: expected_path.to_string(),
                        actual_path,
                    });
                }
            }
        }
        if parameters.is_empty() { Self::from_parameters(structure, values) } else { Err(Error::UnusedParameters) }
    }

    /// Reconstructs a value from `structure` using named parameters interpreted as path prefixes.
    ///
    /// # Parameters
    ///
    ///   - `structure`: Parameter structure to reconstruct.
    ///   - `parameters`: Map from path prefixes to parameter values.
    ///
    /// Prefixes are matched against leaf paths in `structure`, and each leaf receives the value of the most-specific
    /// matching prefix (i.e., longest-prefix-wins). Returns [`Error::MissingPrefixForPath`] if any leaf path is not
    /// covered by a provided prefix, and [`Error::UnusedPrefixPath`] if any provided prefix matches no leaf path.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::collections::HashMap;
    /// # use ryft_core::parameters::{ParameterPath, ParameterPathSegment, Parameterized, Placeholder};
    /// let structure = vec![(Placeholder, Placeholder), (Placeholder, Placeholder)];
    /// let rebuilt = <Vec<(i32, i32)> as Parameterized<i32>>::from_broadcasted_named_parameters(
    ///     structure,
    ///     HashMap::from([
    ///         (ParameterPath::root(), 0),
    ///         (
    ///             ParameterPath::root()
    ///                 .with_segment(ParameterPathSegment::Index(1))
    ///                 .with_segment(ParameterPathSegment::TupleIndex(0)),
    ///             7,
    ///         ),
    ///     ]),
    /// )?;
    /// assert_eq!(rebuilt, vec![(0, 0), (7, 0)]);
    /// # Ok::<(), ryft_core::errors::Error>(())
    /// ```
    fn from_broadcasted_named_parameters(
        structure: Self::ParameterStructure,
        parameters: HashMap<ParameterPath, P>,
    ) -> Result<Self, Error>
    where
        P: Clone,
    {
        let leaf_paths = structure.named_parameters().map(|(path, _)| path).collect::<Vec<_>>();
        let mut prefixes = parameters.into_iter().map(|(path, value)| (path, value, 0_usize)).collect::<Vec<_>>();
        let mut expanded_parameters = HashMap::with_capacity(leaf_paths.len());
        for leaf_path in leaf_paths {
            let mut selected_prefix = None;
            for (index, (prefix_path, _, _)) in prefixes.iter().enumerate() {
                if prefix_path.is_prefix_of(&leaf_path) {
                    let prefix_depth = prefix_path.len();
                    if selected_prefix.map(|(_, depth)| prefix_depth > depth).unwrap_or(true) {
                        selected_prefix = Some((index, prefix_depth));
                    }
                }
            }
            let (selected_prefix_index, _) =
                selected_prefix.ok_or_else(|| Error::MissingPrefixForPath { path: leaf_path.to_string() })?;
            let (_, value, matched_count) = &mut prefixes[selected_prefix_index];
            expanded_parameters.insert(leaf_path, value.clone());
            *matched_count += 1;
        }
        let mut unused_prefix_paths = prefixes
            .into_iter()
            .filter_map(|(path, _, matched_count)| if matched_count == 0 { Some(path.to_string()) } else { None })
            .collect::<Vec<_>>();
        if !unused_prefix_paths.is_empty() {
            unused_prefix_paths.sort_unstable();
            return Err(Error::UnusedPrefixPath { path: unused_prefix_paths[0].clone() });
        }
        Self::from_named_parameters(structure, expanded_parameters)
    }

    /// Maps each nested [`Parameter`] of type `P` in this value using the provided `map_fn` to a [`Parameter`] of type
    /// `T`, while preserving the [`Parameterized`] tree structure of this type. Nested parameters are visited in the
    /// same order as [`Self::parameters`], [`Self::parameters_mut`], [`Self::into_parameters`], and their named
    /// counterparts.
    fn map_parameters<T: Parameter, F: FnMut(P) -> T>(self, map_fn: F) -> Result<Self::To<T>, Error>
    where
        Self::Family: ParameterizedFamily<T>,
    {
        Self::To::<T>::from_parameters(self.parameter_structure(), self.into_parameters().map(map_fn))
    }

    /// Maps each nested [`Parameter`] of type `P` in this value using the provided `map_fn`, which receives the
    /// [`ParameterPath`] for each [`Parameter`] along with its value, and returns a new [`Parameter`] value of type
    /// `T`, while preserving the [`Parameterized`] tree structure of this type. Nested parameters are visited in the
    /// same order as [`Self::parameters`], [`Self::parameters_mut`], [`Self::into_parameters`], and their named
    /// counterparts.
    fn map_named_parameters<T: Parameter, F: FnMut(&ParameterPath, P) -> T>(
        self,
        map_fn: F,
    ) -> Result<Self::To<T>, Error>
    where
        Self::Family: ParameterizedFamily<T>,
    {
        let mut map_fn = map_fn;
        let structure = self.parameter_structure();
        let mut mapped_parameters = HashMap::with_capacity(structure.parameter_count());
        for (path, parameter) in self.into_named_parameters() {
            let mapped_parameter = map_fn(&path, parameter);
            mapped_parameters.insert(path, mapped_parameter);
        }
        Self::To::<T>::from_named_parameters(structure, mapped_parameters)
    }

    /// Splits this value into selected and rejected optional trees according to `predicate`.
    ///
    /// This is inspired by Equinox
    /// [`partition`](https://docs.kidger.site/equinox/api/manipulation/#equinoxpartition).
    /// It is equivalent to filtering with `predicate` and its inverse, but traverses this tree only once.
    ///
    /// # Parameters
    ///
    ///   - `predicate`: Called on every leaf as `predicate(path, parameter)`. Return `true` to place that leaf in the
    ///     first output tree; return `false` to place it in the second output tree.
    ///
    /// # Returns
    ///
    /// A tuple `(selected, rejected)` where both trees preserve the input structure. For each leaf, exactly one side
    /// contains `Some(value)` and the other side contains `None`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::parameters::Parameterized;
    /// let value = vec![(1_i32, 2_i32), (3_i32, 4_i32)];
    /// let (selected, rejected) = value.partition_parameters(|path, _| path.to_string().starts_with("$[1]"))?;
    /// assert_eq!(selected, vec![(None, None), (Some(3), Some(4))]);
    /// assert_eq!(rejected, vec![(Some(1), Some(2)), (None, None)]);
    /// # Ok::<(), ryft_core::errors::Error>(())
    /// ```
    fn partition_parameters<F: FnMut(&ParameterPath, &P) -> bool>(
        self,
        predicate: F,
    ) -> Result<(Self::To<Option<P>>, Self::To<Option<P>>), Error>
    where
        Self::Family: ParameterizedFamily<Option<P>>,
    {
        let mut predicate = predicate;
        let structure = self.parameter_structure();
        let mut selected_parameters = Vec::with_capacity(structure.parameter_count());
        let mut rejected_parameters = Vec::with_capacity(structure.parameter_count());
        for (path, parameter) in self.into_named_parameters() {
            if predicate(&path, &parameter) {
                selected_parameters.push(Some(parameter));
                rejected_parameters.push(None);
            } else {
                selected_parameters.push(None);
                rejected_parameters.push(Some(parameter));
            }
        }
        let selected = Self::To::<Option<P>>::from_parameters(structure, selected_parameters)?;
        let rejected = Self::To::<Option<P>>::from_parameters(selected.parameter_structure(), rejected_parameters)?;
        Ok((selected, rejected))
    }

    /// Keeps leaves that satisfy `predicate` and replaces all other leaves with `None`.
    ///
    /// This is inspired by Equinox
    /// [`filter`](https://docs.kidger.site/equinox/api/manipulation/#equinoxfilter), using `Option<P>` as the
    /// replacement mechanism.
    ///
    /// # Parameters
    ///
    ///   - `predicate`: Called on every leaf as `predicate(path, parameter)`. Return `true` to keep that leaf as
    ///     `Some(parameter)`; return `false` to replace it with `None`.
    ///
    /// # Returns
    ///
    /// A structure-preserving optional tree (`Self::To<Option<P>>`).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::parameters::Parameterized;
    /// let value = vec![(1_i32, 2_i32), (3_i32, 4_i32)];
    /// let filtered = value.filter_parameters(|path, _| path.to_string().ends_with(".1"))?;
    /// assert_eq!(filtered, vec![(None, Some(2)), (None, Some(4))]);
    /// # Ok::<(), ryft_core::errors::Error>(())
    /// ```
    fn filter_parameters<F: FnMut(&ParameterPath, &P) -> bool>(self, predicate: F) -> Result<Self::To<Option<P>>, Error>
    where
        Self::Family: ParameterizedFamily<Option<P>>,
    {
        let mut predicate = predicate;
        self.map_named_parameters(|path, parameter| predicate(path, &parameter).then_some(parameter))
    }

    /// Combines optional trees into one value using left-to-right `Some` precedence per leaf.
    ///
    /// This is inspired by Equinox
    /// [`combine`](https://docs.kidger.site/equinox/api/manipulation/#equinoxcombine) and is typically used to undo
    /// [`Self::filter_parameters`] or [`Self::partition_parameters`].
    ///
    /// # Parameters
    ///
    ///   - `structure`: Target parameter structure for the reconstructed output.
    ///   - `optional_trees`: Optional trees with the same structure as `structure`.
    ///
    /// # Returns
    ///
    /// A reconstructed value where each leaf is the first non-`None` candidate found at that position.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InsufficientParameters`] if any optional tree is too short, [`Error::MissingNamedParameterPath`]
    /// if all candidates are `None` for a leaf, and [`Error::UnusedParameters`] if any optional tree has extra leaves.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::parameters::{Parameterized, Placeholder};
    /// let structure = vec![(Placeholder, Placeholder)];
    /// let combined = <Vec<(i32, i32)> as Parameterized<i32>>::combine_optional_parameters(
    ///     structure,
    ///     vec![
    ///         vec![(Some(10), None)],
    ///         vec![(Some(20), Some(30))],
    ///     ],
    /// )?;
    /// assert_eq!(combined, vec![(10, 30)]);
    /// # Ok::<(), ryft_core::errors::Error>(())
    /// ```
    fn combine_optional_parameters<I>(structure: Self::ParameterStructure, optional_trees: I) -> Result<Self, Error>
    where
        I: IntoIterator<Item = Self::To<Option<P>>>,
        Self::Family: ParameterizedFamily<Option<P>>,
    {
        let expected_paths = structure.named_parameters().map(|(path, _)| path).collect::<Vec<_>>();
        let expected_count = expected_paths.len();
        let mut optional_iterators = optional_trees.into_iter().map(|tree| tree.into_parameters()).collect::<Vec<_>>();
        let mut combined_parameters = Vec::with_capacity(expected_count);
        for path in expected_paths {
            let mut selected = None;
            for iterator in &mut optional_iterators {
                let candidate = iterator.next().ok_or(Error::InsufficientParameters { expected_count })?;
                if selected.is_none() {
                    selected = candidate;
                }
            }
            let parameter = selected.ok_or_else(|| Error::MissingNamedParameterPath { path: path.to_string() })?;
            combined_parameters.push(parameter);
        }
        if optional_iterators.iter_mut().any(|iterator| iterator.next().is_some()) {
            return Err(Error::UnusedParameters);
        }
        Self::from_parameters(structure, combined_parameters)
    }

    /// Replaces parameters using an optional replacement tree.
    ///
    /// This is a convenience operation built on top of [`Self::combine_optional_parameters`]. Leaves with
    /// `Some(value)` in `replacements` take precedence over existing leaf values in `self`, while `None` leaves
    /// preserve the current value.
    ///
    /// # Parameters
    ///
    ///   - `replacements`: Optional replacement tree aligned with `self`.
    ///
    /// # Returns
    ///
    /// A value with the same structure as `self`, where each leaf is either replaced or kept.
    ///
    /// # Errors
    ///
    /// Propagates the same structural errors as [`Self::combine_optional_parameters`] when the replacement tree shape
    /// does not align with `self`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::parameters::Parameterized;
    /// let value = vec![(1_i32, 2_i32), (3_i32, 4_i32)];
    /// let replaced = value.replace_parameters(vec![(None, None), (Some(99), None)])?;
    /// assert_eq!(replaced, vec![(1, 2), (99, 4)]);
    /// # Ok::<(), ryft_core::errors::Error>(())
    /// ```
    fn replace_parameters(self, replacements: Self::To<Option<P>>) -> Result<Self, Error>
    where
        Self::Family: ParameterizedFamily<Option<P>>,
    {
        let structure = self.parameter_structure();
        let current = self.map_parameters(Some)?;
        Self::combine_optional_parameters(structure, vec![replacements, current])
    }

    /// Applies optional updates leaf-wise using `update_fn(base, update)` whenever an update is present.
    ///
    /// This is the general form of the Equinox-inspired
    /// [`apply_updates`](https://docs.kidger.site/equinox/api/manipulation/#equinoxapply_updates) pattern.
    ///
    /// # Parameters
    ///
    ///   - `updates`: Optional update tree aligned with `self`.
    ///   - `update_fn`: Custom update rule used only when a leaf in `updates` is `Some(update)`.
    ///
    /// # Returns
    ///
    /// A value with the same structure as `self`, where each leaf is either unchanged (`None` update) or updated via
    /// `update_fn`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InsufficientParameters`] or [`Error::UnusedParameters`] when the flattened lengths of
    /// `self` and `updates` do not match.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::parameters::Parameterized;
    /// let value = vec![(2_i32, 3_i32)];
    /// let updated = value.apply_parameter_updates_with(
    ///     vec![(None, Some(10))],
    ///     |base, scale| base * scale,
    /// )?;
    /// assert_eq!(updated, vec![(2, 30)]);
    /// # Ok::<(), ryft_core::errors::Error>(())
    /// ```
    fn apply_parameter_updates_with<U: Parameter, F>(
        self,
        updates: Self::To<Option<U>>,
        update_fn: F,
    ) -> Result<Self, Error>
    where
        F: FnMut(P, U) -> P,
        Self::Family: ParameterizedFamily<Option<U>>,
    {
        let structure = self.parameter_structure();
        let expected_count = structure.parameter_count();
        let mut parameters = self.into_parameters();
        let mut updates = updates.into_parameters();
        let mut update_fn = update_fn;
        let mut updated_parameters = Vec::with_capacity(expected_count);
        for _ in 0..expected_count {
            let parameter = parameters.next().ok_or(Error::InsufficientParameters { expected_count })?;
            let update = updates.next().ok_or(Error::InsufficientParameters { expected_count })?;
            let updated = match update {
                Some(update) => update_fn(parameter, update),
                None => parameter,
            };
            updated_parameters.push(updated);
        }
        if parameters.next().is_some() || updates.next().is_some() {
            return Err(Error::UnusedParameters);
        }
        Self::from_parameters(structure, updated_parameters)
    }

    /// Applies additive updates to parameters where the update tree contains `Some(delta)`.
    ///
    /// This mirrors Equinox's `apply_updates` behavior for additive updates and is equivalent to
    /// [`Self::apply_parameter_updates_with`] with `|base, delta| base + delta`.
    ///
    /// # Parameters
    ///
    ///   - `updates`: Optional additive deltas aligned with `self`.
    ///
    /// # Returns
    ///
    /// A value with updated leaves where updates are present and unchanged leaves where updates are `None`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::parameters::Parameterized;
    /// let value = vec![(1_i32, 2_i32), (3_i32, 4_i32)];
    /// let updated = value.apply_parameter_updates(vec![(Some(10), None), (None, Some(-2))])?;
    /// assert_eq!(updated, vec![(11, 2), (3, 2)]);
    /// # Ok::<(), ryft_core::errors::Error>(())
    /// ```
    fn apply_parameter_updates(self, updates: Self::To<Option<P>>) -> Result<Self, Error>
    where
        P: std::ops::Add<Output = P>,
        Self::Family: ParameterizedFamily<Option<P>>,
    {
        self.apply_parameter_updates_with(updates, |base, delta| base + delta)
    }

    /// Replaces parameters at explicit paths with values from `replacements`.
    ///
    /// This is an explicit-path variant of the Equinox-inspired
    /// [`tree_at`](https://docs.kidger.site/equinox/api/manipulation/#equinoxtree_at) behavior.
    ///
    /// # Parameters
    ///
    ///   - `paths`: Exact [`ParameterPath`] values identifying the leaves to replace.
    ///   - `replacements`: Replacement parameter values, in one-to-one correspondence with `paths`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ReplacementCountMismatch`] when `paths` and `replacements` have different lengths, and
    /// [`Error::UnknownNamedParameterPath`] if any requested path is not a valid leaf path.
    fn tree_at_paths<I, R>(self, paths: I, replacements: R) -> Result<Self, Error>
    where
        I: IntoIterator<Item = ParameterPath>,
        R: IntoIterator<Item = P>,
        Self::Family: ParameterizedFamily<Option<P>>,
    {
        let paths = paths.into_iter().collect::<Vec<_>>();
        let replacements = replacements.into_iter().collect::<Vec<_>>();
        if paths.len() != replacements.len() {
            return Err(Error::ReplacementCountMismatch {
                expected_count: paths.len(),
                actual_count: replacements.len(),
            });
        }
        let structure = self.parameter_structure();
        let leaf_paths = structure.named_parameters().map(|(path, _)| path).collect::<HashSet<_>>();
        if let Some(path) = paths.iter().filter(|path| !leaf_paths.contains(*path)).min() {
            return Err(Error::UnknownNamedParameterPath { path: path.to_string() });
        }
        let mut replacement_map = paths
            .into_iter()
            .zip(replacements)
            .map(|(path, replacement)| (path, Some(replacement)))
            .collect::<HashMap<_, _>>();
        let replacement_parameters = structure
            .named_parameters()
            .map(|(path, _)| replacement_map.remove(&path).unwrap_or(None))
            .collect::<Vec<_>>();
        let replacement_tree = Self::To::<Option<P>>::from_parameters(structure, replacement_parameters)?;
        self.replace_parameters(replacement_tree)
    }

    /// Replaces selected leaves using a selector closure over the parameter structure.
    ///
    /// This corresponds to the selector-based style of Equinox
    /// [`tree_at`](https://docs.kidger.site/equinox/api/manipulation/#equinoxtree_at), with explicit
    /// [`ParameterPath`] selection.
    ///
    /// # Parameters
    ///
    ///   - `selector`: Function that receives this value's parameter structure and returns the paths to replace.
    ///   - `replacements`: Replacement parameter values for the selected paths.
    ///
    /// # Returns
    ///
    /// A value with selected leaves replaced and all other leaves unchanged.
    ///
    /// # Errors
    ///
    /// Propagates the same validation errors as [`Self::tree_at_paths`] (e.g.,
    /// [`Error::ReplacementCountMismatch`] and [`Error::UnknownNamedParameterPath`]).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::parameters::Parameterized;
    /// let value = vec![(1_i32, 2_i32), (3_i32, 4_i32)];
    /// let updated = value.tree_at(
    ///     |structure| {
    ///         structure
    ///             .named_parameters()
    ///             .map(|(path, _)| path)
    ///             .filter(|path| path.to_string() == "$[1].1")
    ///             .collect::<Vec<_>>()
    ///     },
    ///     vec![99_i32],
    /// )?;
    /// assert_eq!(updated, vec![(1, 2), (3, 99)]);
    /// # Ok::<(), ryft_core::errors::Error>(())
    /// ```
    fn tree_at<F: FnOnce(&Self::ParameterStructure) -> Vec<ParameterPath>, R: IntoIterator<Item = P>>(
        self,
        selector: F,
        replacements: R,
    ) -> Result<Self, Error>
    where
        Self::Family: ParameterizedFamily<Option<P>>,
    {
        let structure = self.parameter_structure();
        let paths = selector(&structure);
        self.tree_at_paths(paths, replacements)
    }
}

/// Iterator adapter that prefixes each yielded [`ParameterPath`] with [`Self::segment`]. This exists as a dedicated
/// type (instead of using only standard [`Iterator`] combinators) because many [`Parameterized`] associated iterator
/// types must be named concrete types. A closure-based `map(move |...| ...)` adapter would capture the prefix segment
/// and produce an unnameable closure type, which is not usable directly in those associated type definitions on stable
/// Rust. [`PathPrefixedParameterIterator`] preserves static dispatch and avoids heap allocation and dynamic dispatch.
pub struct PathPrefixedParameterIterator<P, I: Iterator<Item = (ParameterPath, P)>> {
    /// Underlying [`Iterator`] that yields `(path, value)` pairs before prefixing with [`Self::segment`].
    pub iterator: I,

    /// [`ParameterPathSegment`] to prepend to each path produced by [`Self::iterator`].
    pub segment: ParameterPathSegment,
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

/// Parameterization family for leaf parameter types.
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

    fn from_parameters_with_remainder<I: Iterator<Item = P>>(
        _structure: Self::ParameterStructure,
        parameters: &mut I,
    ) -> Result<Self, Error> {
        parameters.next().ok_or(Error::InsufficientParameters { expected_count: 1 })
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

    fn from_parameters_with_remainder<I: Iterator<Item = P>>(
        _structure: Self::ParameterStructure,
        _parameters: &mut I,
    ) -> Result<Self, Error> {
        Ok(PhantomData)
    }
}

// Use declarative macros to provide implementations for tuples of [`Parameterized`] items. Note that if a tuple
// contains a mix of [`Parameterized`] and non-[`Parameterized`] items, then the generated implementations here
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
                    ($([<$T:lower>].parameter_structure(),)*)
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

                #[allow(unused_variables)]
                fn from_parameters_with_remainder<I: Iterator<Item = P>>(
                    structure: Self::ParameterStructure,
                    parameters: &mut I,
                ) -> Result<Self, Error> {
                    let ($([<$T:lower _field>],)*) = structure;
                    $(let [<$T:lower>] = $T::from_parameters_with_remainder([<$T:lower _field>], parameters)?;)*
                    Ok(($([<$T:lower>],)*))
                }
            }
        }
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
        let iterator = PathPrefixedParameterIterator { iterator, segment: ParameterPathSegment::TupleIndex($index) };
        iterator.chain(tuple_named_parameter_iterator!($T, ($($tail:$tail_index,)*)))
    }};
}

macro_rules! tuple_named_parameter_iterator_mut {
    ($T:tt, ()) => {
        std::iter::empty::<(ParameterPath, &'_ mut $T)>()
    };

    ($T:tt, ($head:ident:$index:tt, $($tail:ident:$tail_index:tt,)*)) => {{
        let iterator = $head.named_parameters_mut();
        let iterator = PathPrefixedParameterIterator { iterator, segment: ParameterPathSegment::TupleIndex($index) };
        iterator.chain(tuple_named_parameter_iterator_mut!($T, ($($tail:$tail_index,)*)))
    }};
}

macro_rules! tuple_named_parameter_into_iterator {
    ($T:tt, ()) => {
        std::iter::empty::<(ParameterPath, $T)>()
    };

    ($T:tt, ($head:ident:$index:tt, $($tail:ident:$tail_index:tt,)*)) => {{
        let iterator = $head.into_named_parameters();
        let iterator = PathPrefixedParameterIterator { iterator, segment: ParameterPathSegment::TupleIndex($index) };
        iterator.chain(tuple_named_parameter_into_iterator!($T, ($($tail:$tail_index,)*)))
    }};
}

tuple_parameterized_impl!();
tuple_parameterized_impl!(T0:0);
tuple_parameterized_impl!(T0:0, T1:1);
tuple_parameterized_impl!(T0:0, T1:1, T2:2);
tuple_parameterized_impl!(T0:0, T1:1, T2:2, T3:3);
tuple_parameterized_impl!(T0:0, T1:1, T2:2, T3:3, T4:4);
tuple_parameterized_impl!(T0:0, T1:1, T2:2, T3:3, T4:4, T5:5);
tuple_parameterized_impl!(T0:0, T1:1, T2:2, T3:3, T4:4, T5:5, T6:6);
tuple_parameterized_impl!(T0:0, T1:1, T2:2, T3:3, T4:4, T5:5, T6:6, T7:7);
tuple_parameterized_impl!(T0:0, T1:1, T2:2, T3:3, T4:4, T5:5, T6:6, T7:7, T8:8);
tuple_parameterized_impl!(T0:0, T1:1, T2:2, T3:3, T4:4, T5:5, T6:6, T7:7, T8:8, T9:9);
tuple_parameterized_impl!(T0:0, T1:1, T2:2, T3:3, T4:4, T5:5, T6:6, T7:7, T8:8, T9:9, T10:10);
tuple_parameterized_impl!(T0:0, T1:1, T2:2, T3:3, T4:4, T5:5, T6:6, T7:7, T8:8, T9:9, T10:10, T11:11);

pub struct ArrayParameterizedFamily<F, const N: usize>(PhantomData<F>);

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
        self.iter().enumerate().flat_map(|(index, value)| PathPrefixedParameterIterator {
            iterator: value.named_parameters(),
            segment: ParameterPathSegment::Index(index),
        })
    }

    fn named_parameters_mut(&mut self) -> Self::NamedParameterIteratorMut<'_, P> {
        self.iter_mut().enumerate().flat_map(|(index, value)| PathPrefixedParameterIterator {
            iterator: value.named_parameters_mut(),
            segment: ParameterPathSegment::Index(index),
        })
    }

    fn into_named_parameters(self) -> Self::NamedParameterIntoIterator<P> {
        self.into_iter().enumerate().flat_map(|(index, value)| PathPrefixedParameterIterator {
            iterator: value.into_named_parameters(),
            segment: ParameterPathSegment::Index(index),
        })
    }

    fn from_parameters_with_remainder<I: Iterator<Item = P>>(
        structure: Self::ParameterStructure,
        parameters: &mut I,
    ) -> Result<Self, Error> {
        // Make this more efficient by using [std::array::try_from_fn] once it becomes stable.
        // Tracking issue: https://github.com/rust-lang/rust/issues/89379.
        let values = structure
            .into_iter()
            .map(|value_structure| V::from_parameters_with_remainder(value_structure, parameters))
            .collect::<Result<Vec<V>, _>>()?;
        Ok(unsafe { values.try_into().unwrap_unchecked() })
    }
}

pub struct VecParameterizedFamily<F>(PhantomData<F>);

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
        self.iter().enumerate().flat_map(|(index, value)| PathPrefixedParameterIterator {
            iterator: value.named_parameters(),
            segment: ParameterPathSegment::Index(index),
        })
    }

    fn named_parameters_mut(&mut self) -> Self::NamedParameterIteratorMut<'_, P> {
        self.iter_mut().enumerate().flat_map(|(index, value)| PathPrefixedParameterIterator {
            iterator: value.named_parameters_mut(),
            segment: ParameterPathSegment::Index(index),
        })
    }

    fn into_named_parameters(self) -> Self::NamedParameterIntoIterator<P> {
        self.into_iter().enumerate().flat_map(|(index, value)| PathPrefixedParameterIterator {
            iterator: value.into_named_parameters(),
            segment: ParameterPathSegment::Index(index),
        })
    }

    fn from_parameters_with_remainder<I: Iterator<Item = P>>(
        structure: Self::ParameterStructure,
        parameters: &mut I,
    ) -> Result<Self, Error> {
        let expected_count = structure.len();
        let mut values = Vec::new();
        values.reserve_exact(expected_count);
        for value_structure in structure {
            values.push(V::from_parameters_with_remainder(value_structure, parameters).map_err(
                |error| match error {
                    Error::InsufficientParameters { .. } => Error::InsufficientParameters { expected_count },
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
    K: Clone + Debug + Eq + Hash,
    F: ParameterizedFamily<P> + ParameterizedFamily<Placeholder>,
    S: BuildHasher + Clone,
> ParameterizedFamily<P> for HashMapParameterizedFamily<K, F, S>
{
    type To = HashMap<K, <F as ParameterizedFamily<P>>::To, S>;
}

impl<P: Parameter, K: Clone + Debug + Eq + Hash, V: Parameterized<P>, S: BuildHasher + Clone> Parameterized<P>
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
        std::collections::hash_map::Values<'t, K, V>,
        <V as Parameterized<P>>::ParameterIterator<'t, T>,
        fn(&'t V) -> <V as Parameterized<P>>::ParameterIterator<'t, T>,
    >
    where
        Self: 't;

    type ParameterIteratorMut<'t, T: 't + Parameter>
        = std::iter::FlatMap<
        std::collections::hash_map::ValuesMut<'t, K, V>,
        <V as Parameterized<P>>::ParameterIteratorMut<'t, T>,
        fn(&'t mut V) -> <V as Parameterized<P>>::ParameterIteratorMut<'t, T>,
    >
    where
        Self: 't;

    type ParameterIntoIterator<T: Parameter> = std::iter::FlatMap<
        std::collections::hash_map::IntoValues<K, V>,
        <V as Parameterized<P>>::ParameterIntoIterator<T>,
        fn(V) -> <V as Parameterized<P>>::ParameterIntoIterator<T>,
    >;

    type NamedParameterIterator<'t, T: 't + Parameter>
        = std::iter::FlatMap<
        std::collections::hash_map::Iter<'t, K, V>,
        PathPrefixedParameterIterator<&'t T, <V as Parameterized<P>>::NamedParameterIterator<'t, T>>,
        fn(
            (&'t K, &'t V),
        ) -> PathPrefixedParameterIterator<&'t T, <V as Parameterized<P>>::NamedParameterIterator<'t, T>>,
    >
    where
        Self: 't;

    type NamedParameterIteratorMut<'t, T: 't + Parameter>
        = std::iter::FlatMap<
        std::collections::hash_map::IterMut<'t, K, V>,
        PathPrefixedParameterIterator<&'t mut T, <V as Parameterized<P>>::NamedParameterIteratorMut<'t, T>>,
        fn(
            (&'t K, &'t mut V),
        )
            -> PathPrefixedParameterIterator<&'t mut T, <V as Parameterized<P>>::NamedParameterIteratorMut<'t, T>>,
    >
    where
        Self: 't;

    type NamedParameterIntoIterator<T: Parameter> = std::iter::FlatMap<
        std::collections::hash_map::IntoIter<K, V>,
        PathPrefixedParameterIterator<T, <V as Parameterized<P>>::NamedParameterIntoIterator<T>>,
        fn((K, V)) -> PathPrefixedParameterIterator<T, <V as Parameterized<P>>::NamedParameterIntoIterator<T>>,
    >;

    fn parameter_count(&self) -> usize {
        self.values().map(|value| value.parameter_count()).sum()
    }

    fn parameter_structure(&self) -> Self::ParameterStructure {
        let mut structure = HashMap::with_capacity_and_hasher(self.len(), self.hasher().clone());
        structure.extend(self.iter().map(|(key, value)| (key.clone(), value.parameter_structure())));
        structure
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
        self.iter().flat_map(|(key, value)| PathPrefixedParameterIterator {
            iterator: value.named_parameters(),
            segment: ParameterPathSegment::Key(format!("{key:?}")),
        })
    }

    fn named_parameters_mut(&mut self) -> Self::NamedParameterIteratorMut<'_, P> {
        self.iter_mut().flat_map(|(key, value)| PathPrefixedParameterIterator {
            iterator: value.named_parameters_mut(),
            segment: ParameterPathSegment::Key(format!("{key:?}")),
        })
    }

    fn into_named_parameters(self) -> Self::NamedParameterIntoIterator<P> {
        self.into_iter().flat_map(|(key, value)| PathPrefixedParameterIterator {
            iterator: value.into_named_parameters(),
            segment: ParameterPathSegment::Key(format!("{key:?}")),
        })
    }

    fn from_parameters_with_remainder<I: Iterator<Item = P>>(
        structure: Self::ParameterStructure,
        parameters: &mut I,
    ) -> Result<Self, Error> {
        let expected_count = structure.len();
        let mut values = HashMap::with_capacity_and_hasher(expected_count, structure.hasher().clone());
        for (key, value_structure) in structure {
            values.insert(
                key,
                V::from_parameters_with_remainder(value_structure, parameters).map_err(|error| match error {
                    Error::InsufficientParameters { .. } => Error::InsufficientParameters { expected_count },
                    error => error,
                })?,
            );
        }
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fmt::Debug;
    use std::marker::PhantomData;

    use crate::errors::Error;

    use super::{Parameter, ParameterPath, ParameterPathSegment, Parameterized, Placeholder};

    fn assert_roundtrip_parameterized<V>(value: V, expected_parameters: Vec<i32>)
    where
        V: Clone + Debug + PartialEq + Parameterized<i32>,
        V::ParameterStructure: Clone + Debug + PartialEq,
    {
        assert_eq!(value.parameter_count(), expected_parameters.len());
        assert_eq!(value.parameters().copied().collect::<Vec<_>>(), expected_parameters);
        assert_eq!(value.clone().into_parameters().collect::<Vec<_>>(), expected_parameters);

        let structure = value.parameter_structure();
        assert_eq!(V::from_parameters(structure.clone(), expected_parameters.clone()), Ok(value.clone()));

        let mut parameters_with_remainder = expected_parameters.iter().copied().chain(std::iter::once(-1));
        assert_eq!(V::from_parameters_with_remainder(structure, &mut parameters_with_remainder), Ok(value));
        assert_eq!(parameters_with_remainder.collect::<Vec<_>>(), vec![-1]);
    }

    fn assert_parameters_mut_increments<V>(value: V, expected_before: Vec<i32>)
    where
        V: Clone + Debug + PartialEq + Parameterized<i32>,
        V::ParameterStructure: Clone + Debug + PartialEq,
    {
        let mut mutable_value = value;
        for parameter in mutable_value.parameters_mut() {
            *parameter += 1;
        }
        let expected_after = expected_before.iter().map(|parameter| parameter + 1).collect::<Vec<_>>();
        assert_eq!(mutable_value.parameters().copied().collect::<Vec<_>>(), expected_after);
    }

    fn assert_named_roundtrip_parameterized<V>(value: V, expected_paths: Vec<String>, expected_parameters: Vec<i32>)
    where
        V: Clone + Debug + PartialEq + Parameterized<i32>,
        V::ParameterStructure: Clone + Debug + PartialEq,
    {
        let named_refs =
            value.named_parameters().map(|(path, parameter)| (path.to_string(), *parameter)).collect::<Vec<_>>();
        assert_eq!(
            named_refs,
            expected_paths.iter().cloned().zip(expected_parameters.iter().copied()).collect::<Vec<_>>(),
        );

        let named_owned = value.clone().into_named_parameters().collect::<Vec<_>>();
        assert_eq!(
            named_owned.iter().map(|(path, parameter)| (path.to_string(), *parameter)).collect::<Vec<_>>(),
            expected_paths.iter().cloned().zip(expected_parameters.iter().copied()).collect::<Vec<_>>(),
        );

        let structure = value.parameter_structure();
        let named_map = named_owned.into_iter().collect::<HashMap<_, _>>();
        assert_eq!(V::from_named_parameters(structure, named_map), Ok(value));
    }

    fn assert_named_parameters_mut_increments<V>(value: V, expected_paths: Vec<String>, expected_before: Vec<i32>)
    where
        V: Clone + Debug + PartialEq + Parameterized<i32>,
        V::ParameterStructure: Clone + Debug + PartialEq,
    {
        let mut mutable_value = value;
        let mut seen_paths = Vec::new();
        for (path, parameter) in mutable_value.named_parameters_mut() {
            seen_paths.push(path.to_string());
            *parameter += 1;
        }
        assert_eq!(seen_paths, expected_paths);
        let expected_after = expected_before.iter().map(|parameter| parameter + 1).collect::<Vec<_>>();
        assert_eq!(mutable_value.parameters().copied().collect::<Vec<_>>(), expected_after);
    }

    fn assert_parameter_paths<V>(value: &V, expected_paths: Vec<String>)
    where
        V: Parameterized<i32>,
    {
        assert_eq!(value.parameter_paths().map(|path| path.to_string()).collect::<Vec<_>>(), expected_paths);
    }

    mod derive_support {
        pub use crate::errors::Error;
        pub use crate::parameters::{
            Parameter, ParameterPath, ParameterPathSegment, Parameterized, ParameterizedFamily,
            PathPrefixedParameterIterator, Placeholder,
        };
    }

    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    struct Rate32(i32);

    impl Parameter for Rate32 {}

    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    struct Rate64(i64);

    impl Parameter for Rate64 {}

    #[derive(ryft_macros::Parameterized, Clone, Debug, PartialEq, Eq)]
    #[ryft(crate = "crate::parameters::tests::derive_support")]
    struct DomainRates<P: Parameter + Clone> {
        first: P,
        second: P,
    }

    #[derive(ryft_macros::Parameterized, Clone, Debug, PartialEq, Eq)]
    #[ryft(crate = "crate::parameters::tests::derive_support")]
    struct DomainRatesVec<P: Parameter>
    where
        P: Clone,
    {
        values: Vec<DomainRates<P>>,
    }

    #[derive(ryft_macros::Parameterized, Clone, Debug, PartialEq, Eq)]
    #[ryft(crate = "crate::parameters::tests::derive_support")]
    struct DomainOptionalTuple<P: Parameter> {
        maybe_pair: Option<(P, usize)>,
    }

    #[derive(ryft_macros::Parameterized, Clone, Debug, PartialEq, Eq)]
    #[ryft(crate = "crate::parameters::tests::derive_support")]
    enum DomainRatesEnum<P: Parameter> {
        Scalar(P),
        Pair { left: P, right: P },
        Empty,
    }

    macro_rules! assert_tuple_impl {
        (($($value:expr),+ $(,)?)) => {{
            let expected_parameters = vec![$($value),+];
            let expected_paths = (0..expected_parameters.len())
                .map(|index| format!("$.{index}"))
                .collect::<Vec<_>>();
            assert_roundtrip_parameterized(($($value,)+), expected_parameters.clone());
            assert_parameters_mut_increments(($($value,)+), expected_parameters.clone());
            assert_named_roundtrip_parameterized(($($value,)+), expected_paths.clone(), expected_parameters.clone());
            assert_named_parameters_mut_increments(($($value,)+), expected_paths, expected_parameters);
        }};
    }

    #[test]
    fn test_leaf_parameterized_impl() {
        assert_roundtrip_parameterized(7, vec![7]);
        assert_parameters_mut_increments(7, vec![7]);
        assert_named_roundtrip_parameterized(7, vec!["$".to_string()], vec![7]);
        assert_named_parameters_mut_increments(7, vec!["$".to_string()], vec![7]);
        assert_parameter_paths(&7, vec!["$".to_string()]);
    }

    #[test]
    fn test_parameter_path_builder_helpers() {
        let path = ParameterPath::root()
            .with_segment(ParameterPathSegment::Variant("Pair"))
            .with_segment(ParameterPathSegment::Field("weights"))
            .with_segment(ParameterPathSegment::Index(1))
            .with_segment(ParameterPathSegment::TupleIndex(0));
        assert_eq!(path.to_string(), "$.pair.weights[1].0");
        assert!(ParameterPath::root().with_segment(ParameterPathSegment::Variant("Pair")).is_prefix_of(&path));

        let key_path = ParameterPath::root().with_segment(ParameterPathSegment::Key(format!("{:?}", "left")));
        assert_eq!(key_path.to_string(), "$[\"left\"]");
    }

    #[test]
    fn test_phantom_data_parameterized_impl() {
        assert_roundtrip_parameterized(PhantomData::<i32>, vec![]);
        assert_parameters_mut_increments(PhantomData::<i32>, vec![]);
        assert_named_roundtrip_parameterized(PhantomData::<i32>, vec![], vec![]);
        assert_named_parameters_mut_increments(PhantomData::<i32>, vec![], vec![]);
        assert_eq!(PhantomData::<i32>.parameter_structure(), PhantomData::<Placeholder>);
    }

    #[test]
    fn test_tuple_parameterized_impls_up_to_arity_twelve() {
        assert_tuple_impl!((0));
        assert_tuple_impl!((0, 1));
        assert_tuple_impl!((0, 1, 2));
        assert_tuple_impl!((0, 1, 2, 3));
        assert_tuple_impl!((0, 1, 2, 3, 4));
        assert_tuple_impl!((0, 1, 2, 3, 4, 5));
        assert_tuple_impl!((0, 1, 2, 3, 4, 5, 6));
        assert_tuple_impl!((0, 1, 2, 3, 4, 5, 6, 7));
        assert_tuple_impl!((0, 1, 2, 3, 4, 5, 6, 7, 8));
        assert_tuple_impl!((0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
        assert_tuple_impl!((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
        assert_tuple_impl!((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11));
    }

    #[test]
    fn test_array_parameterized_impl() {
        assert_roundtrip_parameterized([1, 2, 3], vec![1, 2, 3]);
        assert_parameters_mut_increments([1, 2, 3], vec![1, 2, 3]);
        assert_named_roundtrip_parameterized(
            [1, 2, 3],
            vec!["$[0]".to_string(), "$[1]".to_string(), "$[2]".to_string()],
            vec![1, 2, 3],
        );
        assert_named_parameters_mut_increments(
            [1, 2, 3],
            vec!["$[0]".to_string(), "$[1]".to_string(), "$[2]".to_string()],
            vec![1, 2, 3],
        );
        assert_roundtrip_parameterized([(1, 2), (3, 4)], vec![1, 2, 3, 4]);
        assert_named_roundtrip_parameterized(
            [(1, 2), (3, 4)],
            vec!["$[0].0".to_string(), "$[0].1".to_string(), "$[1].0".to_string(), "$[1].1".to_string()],
            vec![1, 2, 3, 4],
        );
    }

    #[test]
    fn test_vec_parameterized_impl() {
        assert_roundtrip_parameterized(vec![1, 2, 3], vec![1, 2, 3]);
        assert_parameters_mut_increments(vec![1, 2, 3], vec![1, 2, 3]);
        assert_named_roundtrip_parameterized(
            vec![1, 2, 3],
            vec!["$[0]".to_string(), "$[1]".to_string(), "$[2]".to_string()],
            vec![1, 2, 3],
        );
        assert_named_parameters_mut_increments(
            vec![1, 2, 3],
            vec!["$[0]".to_string(), "$[1]".to_string(), "$[2]".to_string()],
            vec![1, 2, 3],
        );
        assert_roundtrip_parameterized(vec![(1, 2), (3, 4)], vec![1, 2, 3, 4]);
        assert_parameter_paths(
            &vec![(1, 2), (3, 4)],
            vec!["$[0].0".to_string(), "$[0].1".to_string(), "$[1].0".to_string(), "$[1].1".to_string()],
        );
        assert_named_roundtrip_parameterized(
            vec![(1, 2), (3, 4)],
            vec!["$[0].0".to_string(), "$[0].1".to_string(), "$[1].0".to_string(), "$[1].1".to_string()],
            vec![1, 2, 3, 4],
        );

        let mapped: Vec<(i64, i64)> = vec![(1, 2), (3, 4)]
            .map_named_parameters(|path, parameter| {
                let mut offset = 0_i64;
                for segment in path.segments() {
                    match segment {
                        ParameterPathSegment::Index(index) => offset += ((*index as i64) + 1) * 100,
                        ParameterPathSegment::TupleIndex(index) => offset += (*index as i64) + 1,
                        _ => {}
                    }
                }
                i64::from(parameter) + offset
            })
            .unwrap();
        assert_eq!(mapped, vec![(102, 104), (204, 206)]);
    }

    #[test]
    fn test_hash_map_parameterized_impl() {
        let mut value = HashMap::new();
        value.insert("left", (1, 2));
        value.insert("right", (3, 4));

        assert_eq!(value.parameter_count(), 4);
        let parameters = value.parameters().copied().collect::<Vec<_>>();
        assert_eq!(value.clone().into_parameters().collect::<Vec<_>>(), parameters);
        assert_parameters_mut_increments(value.clone(), parameters);

        let structure = value.parameter_structure();
        let mut parameters_in_structure_iteration_order = Vec::new();
        for key in structure.keys() {
            parameters_in_structure_iteration_order.extend(value.get(key).copied().unwrap().into_parameters());
        }
        assert_eq!(
            <HashMap<&str, (i32, i32)> as Parameterized<i32>>::from_parameters(
                structure.clone(),
                parameters_in_structure_iteration_order.clone(),
            ),
            Ok(value.clone())
        );

        let mut parameters_with_remainder =
            parameters_in_structure_iteration_order.iter().copied().chain(std::iter::once(-1));
        assert_eq!(
            <HashMap<&str, (i32, i32)> as Parameterized<i32>>::from_parameters_with_remainder(
                structure,
                &mut parameters_with_remainder
            ),
            Ok(value)
        );
        assert_eq!(parameters_with_remainder.collect::<Vec<_>>(), vec![-1]);
    }

    #[test]
    fn test_hash_map_named_parameterized_impl() {
        let mut value = HashMap::new();
        value.insert("left", (1, 2));
        value.insert("right", (3, 4));

        let named = value.named_parameters().collect::<Vec<_>>();
        assert_eq!(named.len(), 4);
        for (path, parameter) in named {
            let segments = path.segments().cloned().collect::<Vec<_>>();
            assert_eq!(segments.len(), 2);
            assert!(matches!(segments[0], ParameterPathSegment::Key(_)));
            assert!(matches!(segments[1], ParameterPathSegment::TupleIndex(0) | ParameterPathSegment::TupleIndex(1)));
            assert!(matches!(*parameter, 1..=4));
        }

        let expected = value.clone();
        let structure = value.parameter_structure();
        let named_owned = value.into_named_parameters().collect::<HashMap<_, _>>();
        assert_eq!(
            <HashMap<&str, (i32, i32)> as Parameterized<i32>>::from_named_parameters(structure, named_owned),
            Ok(expected)
        );
    }

    #[test]
    fn test_partition_named_parameters_splits_by_predicate() {
        let value = vec![(1, 2), (3, 4)];
        let (selected, rejected) = value.partition_parameters(|path, _| path.to_string().starts_with("$[1]")).unwrap();
        assert_eq!(selected, vec![(None, None), (Some(3), Some(4))]);
        assert_eq!(rejected, vec![(Some(1), Some(2)), (None, None)]);
    }

    #[test]
    fn test_filter_named_parameters_returns_selected_only() {
        let value = vec![(1, 2), (3, 4)];
        let filtered = value.filter_parameters(|path, _| path.to_string().ends_with(".1")).unwrap();
        assert_eq!(filtered, vec![(None, Some(2)), (None, Some(4))]);
    }

    #[test]
    fn test_combine_optional_parameters_roundtrips_partitioned_output() {
        let value = vec![(1, 2), (3, 4)];
        let structure = value.parameter_structure();
        let (selected, rejected) =
            value.clone().partition_parameters(|path, _| path.to_string().starts_with("$[0]")).unwrap();
        let combined =
            <Vec<(i32, i32)> as Parameterized<i32>>::combine_optional_parameters(structure, vec![selected, rejected]);
        assert_eq!(combined, Ok(value));
    }

    #[test]
    fn test_combine_optional_parameters_leftmost_precedence() {
        let structure = vec![(Placeholder, Placeholder)];
        let combined = <Vec<(i32, i32)> as Parameterized<i32>>::combine_optional_parameters(
            structure,
            vec![vec![(Some(10), None)], vec![(Some(20), Some(30))]],
        );
        assert_eq!(combined, Ok(vec![(10, 30)]));
    }

    #[test]
    fn test_combine_optional_parameters_reports_missing_path() {
        let structure = vec![(Placeholder, Placeholder)];
        let combined = <Vec<(i32, i32)> as Parameterized<i32>>::combine_optional_parameters(
            structure,
            vec![vec![(Some(10), None)]],
        );
        assert_eq!(combined, Err(Error::MissingNamedParameterPath { path: "$[0].1".to_string() }));
    }

    #[test]
    fn test_replace_parameters_replaces_subset() {
        let value = vec![(1, 2), (3, 4)];
        let replaced = value.replace_parameters(vec![(None, None), (Some(99), None)]).unwrap();
        assert_eq!(replaced, vec![(1, 2), (99, 4)]);
    }

    #[test]
    fn test_apply_parameter_updates_additive_subset() {
        let value = vec![(1, 2), (3, 4)];
        let updated = value.apply_parameter_updates(vec![(None, Some(100)), (Some(10), None)]).unwrap();
        assert_eq!(updated, vec![(1, 102), (13, 4)]);
    }

    #[test]
    fn test_apply_parameter_updates_with_custom_function() {
        let value = vec![(2, 3), (4, 5)];
        let updated = value
            .apply_parameter_updates_with(vec![(Some(10), None), (None, Some(-1))], |base, scale| base * scale)
            .unwrap();
        assert_eq!(updated, vec![(20, 3), (4, -5)]);
    }

    #[test]
    fn test_apply_parameter_updates_reports_shape_mismatch() {
        let value = vec![(1, 2), (3, 4)];
        let updated = value.apply_parameter_updates(vec![(Some(10), None)]);
        assert_eq!(updated, Err(Error::InsufficientParameters { expected_count: 4 }));
    }

    #[test]
    fn test_tree_at_paths_replaces_single_and_multiple_paths() {
        let value = vec![(1, 2), (3, 4)];
        let single = value
            .clone()
            .tree_at_paths(
                vec![
                    ParameterPath::root()
                        .with_segment(ParameterPathSegment::Index(0))
                        .with_segment(ParameterPathSegment::TupleIndex(1)),
                ],
                vec![99],
            )
            .unwrap();
        assert_eq!(single, vec![(1, 99), (3, 4)]);

        let multiple = value
            .tree_at_paths(
                vec![
                    ParameterPath::root()
                        .with_segment(ParameterPathSegment::Index(0))
                        .with_segment(ParameterPathSegment::TupleIndex(0)),
                    ParameterPath::root()
                        .with_segment(ParameterPathSegment::Index(1))
                        .with_segment(ParameterPathSegment::TupleIndex(1)),
                ],
                vec![10, 20],
            )
            .unwrap();
        assert_eq!(multiple, vec![(10, 2), (3, 20)]);
    }

    #[test]
    fn test_tree_at_selector_selects_from_structure() {
        let value = vec![(1, 2), (3, 4)];
        let updated = value
            .tree_at(
                |structure| {
                    structure
                        .named_parameters()
                        .filter_map(|(path, _)| {
                            let selected =
                                matches!(path.segments().next_back(), Some(ParameterPathSegment::TupleIndex(1)));
                            selected.then_some(path)
                        })
                        .collect::<Vec<_>>()
                },
                vec![20, 40],
            )
            .unwrap();
        assert_eq!(updated, vec![(1, 20), (3, 40)]);
    }

    #[test]
    fn test_tree_at_reports_replacement_count_mismatch() {
        let value = vec![(1, 2), (3, 4)];
        let updated = value.tree_at_paths(
            vec![
                ParameterPath::root()
                    .with_segment(ParameterPathSegment::Index(0))
                    .with_segment(ParameterPathSegment::TupleIndex(0)),
                ParameterPath::root()
                    .with_segment(ParameterPathSegment::Index(1))
                    .with_segment(ParameterPathSegment::TupleIndex(1)),
            ],
            vec![10],
        );
        assert_eq!(updated, Err(Error::ReplacementCountMismatch { expected_count: 2, actual_count: 1 }));
    }

    #[test]
    fn test_tree_at_reports_unknown_path() {
        let value = vec![(1, 2), (3, 4)];
        let updated = value.tree_at_paths(
            vec![
                ParameterPath::root()
                    .with_segment(ParameterPathSegment::Index(2))
                    .with_segment(ParameterPathSegment::TupleIndex(0)),
            ],
            vec![10],
        );
        assert_eq!(updated, Err(Error::UnknownNamedParameterPath { path: "$[2].0".to_string() }));
    }

    #[test]
    fn test_tree_at_paths_supports_leaf_parameterized_value() {
        let value = 7;
        let updated = value.tree_at_paths(vec![ParameterPath::root()], vec![11]).unwrap();
        assert_eq!(updated, 11);
    }

    #[test]
    fn test_tree_at_paths_supports_hash_map_key_paths() {
        let mut value = HashMap::new();
        value.insert("left", (1, 2));
        value.insert("right", (3, 4));
        let replaced = value
            .tree_at_paths(
                vec![
                    ParameterPath::root()
                        .with_segment(ParameterPathSegment::Key(format!("{:?}", "right")))
                        .with_segment(ParameterPathSegment::TupleIndex(1)),
                ],
                vec![99],
            )
            .unwrap();
        assert_eq!(replaced.get("left"), Some(&(1, 2)));
        assert_eq!(replaced.get("right"), Some(&(3, 99)));
    }

    #[test]
    fn test_derive_supports_additional_parameter_bounds() {
        let value = DomainRates { first: Rate32(3), second: Rate32(7) };
        assert_eq!(value.parameter_count(), 2);
        assert_eq!(value.parameters().copied().collect::<Vec<_>>(), vec![Rate32(3), Rate32(7)]);
        assert_eq!(
            value.named_parameters().map(|(path, parameter)| (path.to_string(), *parameter)).collect::<Vec<_>>(),
            vec![("$.first".to_string(), Rate32(3)), ("$.second".to_string(), Rate32(7))],
        );
        assert_eq!(value.parameter_structure(), DomainRates { first: Placeholder, second: Placeholder });
        assert_eq!(
            DomainRates::from_named_parameters(
                value.parameter_structure(),
                value.clone().into_named_parameters().collect::<HashMap<_, _>>(),
            ),
            Ok(value.clone())
        );

        let mapped: DomainRates<Rate64> = value.clone().map_parameters(|rate| Rate64(i64::from(rate.0) * 10)).unwrap();
        assert_eq!(mapped, DomainRates { first: Rate64(30), second: Rate64(70) });

        let mapped_named: DomainRates<Rate64> = value
            .map_named_parameters(|path, rate| {
                let multiplier = match path.segments().next() {
                    Some(ParameterPathSegment::Field("first")) => 10_i64,
                    Some(ParameterPathSegment::Field("second")) => 100_i64,
                    _ => 1_i64,
                };
                Rate64(i64::from(rate.0) * multiplier)
            })
            .unwrap();
        assert_eq!(mapped_named, DomainRates { first: Rate64(30), second: Rate64(700) });
    }

    #[test]
    fn test_derive_supports_nested_option_with_mixed_tuple_fields() {
        let some = DomainOptionalTuple { maybe_pair: Some((Rate32(9), 42)) };
        assert_eq!(some.parameter_count(), 1);
        assert_eq!(some.parameters().copied().collect::<Vec<_>>(), vec![Rate32(9)]);
        assert_eq!(
            some.named_parameters().map(|(path, parameter)| (path.to_string(), *parameter)).collect::<Vec<_>>(),
            vec![("$.maybe_pair.0.0".to_string(), Rate32(9))],
        );
        assert_eq!(some.parameter_structure(), DomainOptionalTuple { maybe_pair: Some((Placeholder, 42)) });
        assert_eq!(
            DomainOptionalTuple::from_named_parameters(
                some.parameter_structure(),
                some.clone().into_named_parameters().collect::<HashMap<_, _>>(),
            ),
            Ok(some.clone())
        );

        let mapped: DomainOptionalTuple<Rate64> =
            some.clone().map_parameters(|rate| Rate64(i64::from(rate.0) + 5)).unwrap();
        assert_eq!(mapped, DomainOptionalTuple { maybe_pair: Some((Rate64(14), 42)) });

        let none = DomainOptionalTuple::<Rate32> { maybe_pair: None };
        assert_eq!(none.parameter_count(), 0);
        assert!(none.named_parameters().next().is_none());
        assert_eq!(none.parameter_structure(), DomainOptionalTuple { maybe_pair: None::<(Placeholder, usize)> });
        assert_eq!(
            DomainOptionalTuple::from_named_parameters(
                none.parameter_structure(),
                none.clone().into_named_parameters().collect::<HashMap<_, _>>(),
            ),
            Ok(none)
        );
    }

    #[test]
    fn test_derive_supports_additional_parameter_bounds_in_where_clause() {
        let value = DomainRatesVec {
            values: vec![
                DomainRates { first: Rate32(1), second: Rate32(2) },
                DomainRates { first: Rate32(3), second: Rate32(4) },
            ],
        };
        assert_eq!(value.parameter_count(), 4);
        assert_eq!(
            value.parameter_structure(),
            DomainRatesVec {
                values: vec![
                    DomainRates { first: Placeholder, second: Placeholder },
                    DomainRates { first: Placeholder, second: Placeholder },
                ],
            }
        );
        assert_eq!(
            DomainRatesVec::from_parameters(
                value.parameter_structure(),
                vec![Rate32(1), Rate32(2), Rate32(3), Rate32(4)],
            ),
            Ok(value.clone())
        );
        assert_eq!(
            value.named_parameters().map(|(path, parameter)| (path.to_string(), *parameter)).collect::<Vec<_>>(),
            vec![
                ("$.values[0].first".to_string(), Rate32(1)),
                ("$.values[0].second".to_string(), Rate32(2)),
                ("$.values[1].first".to_string(), Rate32(3)),
                ("$.values[1].second".to_string(), Rate32(4)),
            ],
        );
        assert_eq!(
            DomainRatesVec::from_named_parameters(
                value.parameter_structure(),
                value.clone().into_named_parameters().collect::<HashMap<_, _>>(),
            ),
            Ok(value.clone())
        );

        let mapped: DomainRatesVec<Rate64> = value.map_parameters(|rate| Rate64(i64::from(rate.0) + 5)).unwrap();
        assert_eq!(
            mapped,
            DomainRatesVec {
                values: vec![
                    DomainRates { first: Rate64(6), second: Rate64(7) },
                    DomainRates { first: Rate64(8), second: Rate64(9) },
                ],
            }
        );
    }

    #[test]
    fn test_derive_enum_supports_named_parameters() {
        let scalar = DomainRatesEnum::Scalar(Rate32(9));
        assert_eq!(
            scalar
                .named_parameters()
                .map(|(path, parameter)| (path.to_string(), *parameter))
                .collect::<Vec<_>>(),
            vec![("$.scalar.0".to_string(), Rate32(9))],
        );
        assert_eq!(
            DomainRatesEnum::from_named_parameters(
                scalar.parameter_structure(),
                scalar.clone().into_named_parameters().collect::<HashMap<_, _>>(),
            ),
            Ok(scalar.clone())
        );

        let pair = DomainRatesEnum::Pair { left: Rate32(1), right: Rate32(2) };
        assert_eq!(
            pair.named_parameters().map(|(path, parameter)| (path.to_string(), *parameter)).collect::<Vec<_>>(),
            vec![("$.pair.left".to_string(), Rate32(1)), ("$.pair.right".to_string(), Rate32(2)),],
        );
        assert_eq!(
            DomainRatesEnum::from_named_parameters(
                pair.parameter_structure(),
                pair.clone().into_named_parameters().collect::<HashMap<_, _>>(),
            ),
            Ok(pair.clone())
        );
        let pair_mapped: DomainRatesEnum<Rate64> = pair
            .map_named_parameters(|path, rate| {
                let mut bonus = 0_i64;
                for segment in path.segments() {
                    match segment {
                        ParameterPathSegment::Variant("Pair") => bonus += 1_000,
                        ParameterPathSegment::Field("left") => bonus += 10,
                        ParameterPathSegment::Field("right") => bonus += 20,
                        _ => {}
                    }
                }
                Rate64(i64::from(rate.0) + bonus)
            })
            .unwrap();
        assert_eq!(pair_mapped, DomainRatesEnum::Pair { left: Rate64(1011), right: Rate64(1022) });

        let empty = DomainRatesEnum::<Rate32>::Empty;
        assert!(empty.named_parameters().next().is_none());
        assert_eq!(
            DomainRatesEnum::from_named_parameters(
                empty.parameter_structure(),
                empty.clone().into_named_parameters().collect::<HashMap<_, _>>(),
            ),
            Ok(empty)
        );
    }

    #[test]
    fn test_from_parameters_reports_unused_parameters() {
        assert_eq!(<i32 as Parameterized<i32>>::from_parameters(Placeholder, vec![3, 4]), Err(Error::UnusedParameters));
    }

    #[test]
    fn test_from_named_parameters_reports_unused_parameters() {
        let extra_path = ParameterPath::root().with_segment(ParameterPathSegment::Field("extra"));
        assert_eq!(
            <i32 as Parameterized<i32>>::from_named_parameters(
                Placeholder,
                HashMap::from([(ParameterPath::root(), 3), (extra_path, 4)]),
            ),
            Err(Error::UnusedParameters)
        );
    }

    #[test]
    fn test_from_parameters_reports_insufficient_parameters_for_vec() {
        let structure = vec![Placeholder, Placeholder, Placeholder];
        let result = <Vec<i32> as Parameterized<i32>>::from_parameters(structure, vec![1, 2]);
        assert_eq!(result, Err(Error::InsufficientParameters { expected_count: 3 }));
    }

    #[test]
    fn test_from_named_parameters_reports_insufficient_parameters_for_vec() {
        let structure = vec![Placeholder, Placeholder, Placeholder];
        let parameters = vec![1, 2].into_named_parameters().collect::<HashMap<_, _>>();
        let result = <Vec<i32> as Parameterized<i32>>::from_named_parameters(structure, parameters);
        assert_eq!(result, Err(Error::InsufficientParameters { expected_count: 3 }));
    }

    #[test]
    fn test_from_parameters_reports_insufficient_parameters_for_hash_map() {
        let mut structure = HashMap::new();
        structure.insert("left", Placeholder);
        structure.insert("right", Placeholder);
        structure.insert("middle", Placeholder);
        let result = <HashMap<&str, i32> as Parameterized<i32>>::from_parameters(structure, vec![1, 2]);
        assert_eq!(result, Err(Error::InsufficientParameters { expected_count: 3 }));
    }

    #[test]
    fn test_from_named_parameters_reports_path_mismatch() {
        let value = vec![10, 20];
        let mut named = value.clone().into_named_parameters().collect::<HashMap<_, _>>();
        let replaced_value = named
            .remove(&ParameterPath::root().with_segment(ParameterPathSegment::Index(1)))
            .expect("Expected to remove path $[1] from named parameter map.");
        named.insert(ParameterPath::root().with_segment(ParameterPathSegment::Index(2)), replaced_value);
        let result = <Vec<i32> as Parameterized<i32>>::from_named_parameters(value.parameter_structure(), named);
        assert!(matches!(
            result,
            Err(Error::NamedParameterPathMismatch { expected_path, actual_path })
                if expected_path == "$[1]" && actual_path == "$[2]",
        ));
    }

    #[test]
    fn test_from_named_parameters_is_order_independent() {
        let value = vec![(1, 2), (3, 4)];
        let mut named = value.clone().into_named_parameters().collect::<Vec<_>>();
        named.reverse();
        let named_map = named.into_iter().collect::<HashMap<_, _>>();
        assert_eq!(
            <Vec<(i32, i32)> as Parameterized<i32>>::from_named_parameters(value.parameter_structure(), named_map),
            Ok(value)
        );
    }

    #[test]
    fn test_from_named_parameters_with_broadcasting_root_prefix() {
        let structure = vec![(Placeholder, Placeholder), (Placeholder, Placeholder)];
        let rebuilt = <Vec<(i32, i32)> as Parameterized<i32>>::from_broadcasted_named_parameters(
            structure,
            HashMap::from([(ParameterPath::root(), 9)]),
        );
        assert_eq!(rebuilt, Ok(vec![(9, 9), (9, 9)]));
    }

    #[test]
    fn test_from_named_parameters_with_broadcasting_uses_most_specific_prefix() {
        let structure = vec![(Placeholder, Placeholder), (Placeholder, Placeholder)];
        let rebuilt = <Vec<(i32, i32)> as Parameterized<i32>>::from_broadcasted_named_parameters(
            structure,
            HashMap::from([
                (ParameterPath::root(), 1),
                (ParameterPath::root().with_segment(ParameterPathSegment::Index(1)), 10),
                (
                    ParameterPath::root()
                        .with_segment(ParameterPathSegment::Index(1))
                        .with_segment(ParameterPathSegment::TupleIndex(0)),
                    99,
                ),
            ]),
        );
        assert_eq!(rebuilt, Ok(vec![(1, 1), (99, 10)]));
    }

    #[test]
    fn test_from_named_parameters_with_broadcasting_reports_missing_prefix() {
        let structure = vec![(Placeholder, Placeholder), (Placeholder, Placeholder)];
        let result = <Vec<(i32, i32)> as Parameterized<i32>>::from_broadcasted_named_parameters(
            structure,
            HashMap::from([(ParameterPath::root().with_segment(ParameterPathSegment::Index(0)), 5)]),
        );
        assert_eq!(result, Err(Error::MissingPrefixForPath { path: "$[1].0".to_string() }));
    }

    #[test]
    fn test_from_named_parameters_with_broadcasting_reports_unused_prefix() {
        let structure = vec![(Placeholder, Placeholder)];
        let result = <Vec<(i32, i32)> as Parameterized<i32>>::from_broadcasted_named_parameters(
            structure,
            HashMap::from([
                (ParameterPath::root(), 5),
                (ParameterPath::root().with_segment(ParameterPathSegment::Index(1)), 10),
            ]),
        );
        assert_eq!(result, Err(Error::UnusedPrefixPath { path: "$[1]".to_string() }));
    }

    #[test]
    fn test_from_named_parameters_with_broadcasting_matches_exact_named_behavior() {
        let value = vec![(1, 2), (3, 4)];
        let structure = value.parameter_structure();
        let named_parameters = value.into_named_parameters().collect::<HashMap<_, _>>();
        assert_eq!(
            <Vec<(i32, i32)> as Parameterized<i32>>::from_broadcasted_named_parameters(structure, named_parameters),
            Ok(vec![(1, 2), (3, 4)]),
        );
    }

    #[test]
    fn test_from_named_parameters_with_broadcasting_supports_hash_map_key_paths() {
        let mut structure = HashMap::new();
        structure.insert("left", (Placeholder, Placeholder));
        structure.insert("right", (Placeholder, Placeholder));
        let result = <HashMap<&str, (i32, i32)> as Parameterized<i32>>::from_broadcasted_named_parameters(
            structure,
            HashMap::from([
                (ParameterPath::root().with_segment(ParameterPathSegment::Key(format!("{:?}", "left"))), 10),
                (ParameterPath::root().with_segment(ParameterPathSegment::Key(format!("{:?}", "right"))), 30),
                (
                    ParameterPath::root()
                        .with_segment(ParameterPathSegment::Key(format!("{:?}", "right")))
                        .with_segment(ParameterPathSegment::TupleIndex(1)),
                    20,
                ),
            ]),
        )
        .unwrap();
        assert_eq!(result.get("left"), Some(&(10, 10)));
        assert_eq!(result.get("right"), Some(&(30, 20)));
    }
}
