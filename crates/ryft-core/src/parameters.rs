use std::collections::{BTreeMap, HashMap};
use std::fmt::{Debug, Display};
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;

use convert_case::{Case, Casing};
use half::{bf16, f16};
use paste::paste;
use thiserror::Error;

/// Represents [`Parameter`]-related errors.
#[derive(Error, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ParameterError {
    #[error(
        "{}",
        match paths.as_deref() {
            None => "got more parameters than expected".to_string(),
            Some(paths) => format!(
                "got more parameters than expected; unused parameter paths: {}",
                paths.iter().map(|path| format!("'{path}'")).collect::<Vec<_>>().join(", "),
            ),
        }
    )]
    UnusedParameters { paths: Option<Vec<String>> },

    #[error(
        "{}",
        match paths.as_deref() {
            None => format!("got fewer parameters than expected; expected at least {expected_count}"),
            Some(paths) => format!(
                "got fewer parameters than expected; expected at least {expected_count}; missing parameter paths: {}",
                paths.iter().map(|path| format!("'{path}'")).collect::<Vec<_>>().join(", "),
            ),
        }
    )]
    MissingParameters { expected_count: usize, paths: Option<Vec<String>> },

    #[error(
        "got ambiguous parameter values while combining parameterized values; conflicting values: {}",
        values.iter().map(|value| format!("'{value}'")).collect::<Vec<_>>().join(", "),
    )]
    AmbiguousParameterCombination { values: Vec<String> },
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

    /// [`Debug`]-formatted key of a map entry (e.g., [`HashMap`] or [`BTreeMap`]).
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
/// [`Self::from_broadcasted_named_parameters`], [`Self::map_named_parameters`], and [`Self::map_named_parameters`]
/// functions.
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
/// // Unflattening from broadcasted named parameters:
/// assert_eq!(
///     Value::from_broadcasted_named_parameters(
///         value.parameter_structure(),
///         HashMap::from([
///             (ParameterPath::root(), 0),
///             (ParameterPath::root().with_segment(ParameterPathSegment::TupleIndex(1)), 10),
///             (
///                 ParameterPath::root()
///                     .with_segment(ParameterPathSegment::TupleIndex(1))
///                     .with_segment(ParameterPathSegment::Key(format!("{:?}", "b")))
///                     .with_segment(ParameterPathSegment::Index(1)),
///                 30,
///             ),
///         ]),
///     )?,
///     (0, BTreeMap::from([("a", vec![10]), ("b", vec![10, 30])]), (0,)),
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

    /// Type that represents a shape-only representation of this [`Parameterized`] type with all nested `P` parameter
    /// types replaced by [`Placeholder`]. This must always be set to `Self::To<Placeholder>`. The only reason this is
    /// not done by default is that defaulted associated types are not supported in stable Rust, and this forces us to
    /// require that all implementations provide an implementation for this associated type as well.
    type ParameterStructure: Parameterized<Placeholder, Family = Self::Family, To<P> = Self>
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
    /// and path prefix matching is not being used. If there are missing parameters preventing reconstruction from
    /// being feasible, then this function will return a [`ParameterError::MissingParameters`] error. Furthermore,
    /// if extra paths remain after reconstruction, then it will return an [`ParameterError::UnusedParameters`] error.
    /// For fully worked examples, refer to the examples provided in the top-level documentation of the
    /// [`Parameterized`] trait.
    fn from_named_parameters<I: IntoIterator<Item = (ParameterPath, P)>>(
        structure: Self::ParameterStructure,
        parameters: I,
    ) -> Result<Self, ParameterError> {
        // We try to consume parameters in lockstep with `structure.named_parameters()` first. If we encounter any
        // out-of-order parameters, we lazily materialize the remaining entries into a hash map and continue with
        // keyed lookups.
        let expected_count = structure.parameter_count();
        let mut values = Vec::new();
        values.reserve_exact(expected_count);
        let mut matching_paths = Vec::new();
        matching_paths.reserve_exact(expected_count);
        let mut missing_paths = Vec::new();
        let mut deferred_parameters = None::<HashMap<_, _>>;
        let mut matching_path_indices = None::<HashMap<ParameterPath, usize>>;
        let mut parameters = parameters.into_iter();
        let mut next_parameter = parameters.next();

        for (expected_path, _) in structure.named_parameters() {
            if let Some(deferred_parameters) = deferred_parameters.as_mut() {
                // Once we fall back to deferred lookups, reconstruction is direct keyed removal
                // for the remaining expected paths.
                match deferred_parameters.remove(&expected_path) {
                    Some(parameter) => values.push(parameter),
                    None => missing_paths.push(expected_path.to_string()),
                }
                continue;
            }

            match next_parameter.take() {
                Some((provided_path, parameter)) if provided_path == expected_path => {
                    // As long as the provided parameter paths match the structure parameter paths that we are
                    // iterating over, we just consume the provided parameters without materializing a hash map.
                    values.push(parameter);
                    matching_paths.push(expected_path);
                    next_parameter = parameters.next();
                }
                Some((provided_path, parameter)) => {
                    // Once there is a mismatch, we materialize the hash map and continue via keyed lookup.
                    let mut deferred = HashMap::with_capacity(parameters.size_hint().0 + 1);
                    let matching_path_indices = matching_path_indices.get_or_insert_with(|| {
                        matching_paths
                            .iter()
                            .cloned()
                            .enumerate()
                            .map(|(index, path)| (path, index))
                            .collect::<HashMap<_, _>>()
                    });

                    // For duplicate parameter paths, the last provided value wins. Duplicates targeting
                    // already matched paths update `values` in place; all others remain deferred.
                    let parameters = std::iter::once((provided_path, parameter)).chain(parameters.by_ref());
                    for (provided_path, parameter) in parameters {
                        if let Some(index) = matching_path_indices.get(&provided_path).copied() {
                            values[index] = parameter;
                        } else {
                            deferred.insert(provided_path, parameter);
                        }
                    }

                    match deferred.remove(&expected_path) {
                        Some(parameter) => values.push(parameter),
                        None => missing_paths.push(expected_path.to_string()),
                    }

                    deferred_parameters = Some(deferred);
                }
                None => missing_paths.push(expected_path.to_string()),
            }
        }

        if deferred_parameters.is_none() {
            if let Some((provided_path, parameter)) = next_parameter.take() {
                // No mismatch occurred during traversal, but additional parameters remain. We need to materialize
                // the remaining parameters while still applying duplicate overrides for previously matched paths.
                let mut deferred = HashMap::with_capacity(parameters.size_hint().0 + 1);
                let matching_path_indices = matching_path_indices.get_or_insert_with(|| {
                    matching_paths
                        .iter()
                        .cloned()
                        .enumerate()
                        .map(|(index, path)| (path, index))
                        .collect::<HashMap<_, _>>()
                });

                let parameters = std::iter::once((provided_path, parameter)).chain(parameters);
                for (provided_path, parameter) in parameters {
                    if let Some(index) = matching_path_indices.get(&provided_path).copied() {
                        values[index] = parameter;
                    } else {
                        deferred.insert(provided_path, parameter);
                    }
                }

                deferred_parameters = Some(deferred);
            }
        }

        if !missing_paths.is_empty() {
            Err(ParameterError::MissingParameters { expected_count, paths: Some(missing_paths) })
        } else if deferred_parameters.is_some_and(|deferred_parameters| !deferred_parameters.is_empty()) {
            Err(ParameterError::UnusedParameters { paths: None })
        } else {
            Self::from_parameters(structure, values)
        }
    }

    /// Reconstructs a value of this [`Parameterized`] type having the provided `structure` and consuming named values
    /// from the provided `parameters` to populate its parameters, where keys in `parameters` are interpreted as path
    /// prefixes. Unlike [`Self::from_named_parameters`], this function does not require exact leaf paths, and each
    /// leaf path in `structure` receives the value from the most specific matching path prefix (i.e., the longest
    /// shared path prefix). If any leaf path is not covered by a provided prefix, then this function will return a
    /// [`ParameterError::MissingParameters`] error. Furthermore, if there are any remaining [`ParameterPath`]s with no
    /// match in `structure`, then it will return an [`ParameterError::UnusedParameters`] error. Note that since one
    /// prefix value may need to populate multiple leaves, this function requires `P: Clone`. For fully worked
    /// examples, refer to the examples provided in the top-level documentation of the [`Parameterized`] trait.
    fn from_broadcasted_named_parameters<I: IntoIterator<Item = (ParameterPath, P)>>(
        structure: Self::ParameterStructure,
        parameters: I,
    ) -> Result<Self, ParameterError>
    where
        P: Clone,
    {
        // This function uses a small local prefix trie to resolve each leaf parameter path to the most specific
        // matching prefix (i.e., the longest matching prefix) in path-depth time, instead of scanning all prefixes
        // for every leaf. That is usually much faster when many prefixes are provided, at the cost of the additional
        // trie construction/allocation overhead, which can be less favorable for very small prefix sets.

        #[derive(Default)]
        struct PrefixTrieNode {
            children: HashMap<ParameterPathSegment, usize>,
            selected_prefix_index: Option<usize>,
        }

        let paths = structure.named_parameters().map(|(path, _)| path).collect::<Vec<_>>();
        let expected_count = paths.len();
        let mut path_prefixes = parameters.into_iter().map(|(path, value)| (path, value, 0usize)).collect::<Vec<_>>();

        // Build a trie of provided path prefixes so that each lef parameter path can be resolved by walking segments
        // instead of scanning all prefixes. This turns per-path prefix selection into a path-depth walk.
        let mut prefix_trie = vec![PrefixTrieNode::default()];
        for (prefix_index, (path, _, _)) in path_prefixes.iter().enumerate() {
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

        // Keep values in `structure.named_parameters()` order so that reconstruction can use `from_parameters`.
        let mut values = Vec::with_capacity(expected_count);
        let mut missing_paths = Vec::new();
        for path in paths {
            let mut trie_node_index = 0usize;
            let mut selected_prefix_index = prefix_trie[trie_node_index].selected_prefix_index;
            // As we descend the trie, track the deepest prefix node encountered. That is the "most specific" match.
            for segment in path.segments() {
                let Some(next_trie_node_index) = prefix_trie[trie_node_index].children.get(segment).copied() else {
                    break;
                };
                trie_node_index = next_trie_node_index;
                if let Some(prefix_index) = prefix_trie[trie_node_index].selected_prefix_index {
                    selected_prefix_index = Some(prefix_index);
                }
            }
            if let Some(selected_prefix_index) = selected_prefix_index {
                let (_, value, matched_count) = &mut path_prefixes[selected_prefix_index];
                values.push(value.clone());
                *matched_count += 1;
            } else {
                missing_paths.push(path.to_string());
            }
        }

        let mut unused_prefix_paths = path_prefixes
            .into_iter()
            .filter_map(|(path, _, matched_count)| if matched_count == 0 { Some(path.to_string()) } else { None })
            .collect::<Vec<_>>();

        if !missing_paths.is_empty() {
            Err(ParameterError::MissingParameters { expected_count, paths: Some(missing_paths) })
        } else if !unused_prefix_paths.is_empty() {
            unused_prefix_paths.sort_unstable();
            Err(ParameterError::UnusedParameters { paths: Some(unused_prefix_paths) })
        } else {
            Self::from_parameters(structure, values)
        }
    }

    /// Broadcasts the named parameters from this [`Parameterized`] value into the provided target `structure`,
    /// returning a new value of type `T`. This is a convenience wrapper around [`T::from_broadcasted_named_parameters`]
    /// that pairs each current parameter with its current leaf [`ParameterPath`] and uses those paths as prefixes for
    /// filling the target structure. As a result, each leaf path in `structure` receives the value from the most
    /// specific matching current parameter path prefix (i.e., the longest shared prefix among this value's existing
    /// parameter paths). If the current parameter paths do not cover all target leaves, or if some current parameter
    /// paths match no target leaf, then this function returns the corresponding [`ParameterError::MissingParameters`]
    /// or [`ParameterError::UnusedParameters`] error from [`T::from_broadcasted_named_parameters`]. Since one current
    /// parameter may need to populate multiple leaves in `structure`, this function requires `P: Clone`.
    fn broadcast_to_parameter_structure<T: Parameterized<P>>(
        &self,
        structure: T::ParameterStructure,
    ) -> Result<T, ParameterError>
    where
        P: Clone,
    {
        T::from_broadcasted_named_parameters(
            structure,
            self.named_parameters().map(|(path, parameter)| (path, parameter.clone())),
        )
    }

    /// Maps each nested [`Parameter`] of type `P` in this value using the provided `map_fn` to a [`Parameter`] of type
    /// `T`, while preserving the [`Parameterized`] structure of this type. Nested parameters are visited in the same
    /// order as [`Self::parameters`], [`Self::parameters_mut`], [`Self::into_parameters`], [`Self::named_parameters`],
    /// [`Self::named_parameters_mut`], and [`Self::into_named_parameters`].
    fn map_parameters<T: Parameter, F: FnMut(P) -> T>(self, map_fn: F) -> Result<Self::To<T>, ParameterError>
    where
        Self::Family: ParameterizedFamily<T>,
    {
        Self::To::<T>::from_parameters(self.parameter_structure(), self.into_parameters().map(map_fn))
    }

    /// Maps each nested [`Parameter`] of type `P` in this value using the provided `map_fn`, which receives the
    /// [`ParameterPath`] for each [`Parameter`] along with its value, and returns a new [`Parameter`] value of type
    /// `T`, while preserving the [`Parameterized`] structure of this type. Nested parameters are visited in the
    /// same order as [`Self::parameters`], [`Self::parameters_mut`], [`Self::into_parameters`],
    /// [`Self::named_parameters`], [`Self::named_parameters_mut`], and [`Self::into_named_parameters`].
    fn map_named_parameters<T: Parameter, F: FnMut(&ParameterPath, P) -> T>(
        self,
        map_fn: F,
    ) -> Result<Self::To<T>, ParameterError>
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

    fn from_parameters_with_remainder<I: Iterator<Item = P>>(
        _structure: Self::ParameterStructure,
        _parameters: &mut I,
    ) -> Result<Self, ParameterError> {
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
                ) -> Result<Self, ParameterError> {
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
    ) -> Result<Self, ParameterError> {
        // TODO(eaplatanios): Make this more efficient by using [`std::array::try_from_fn`] once it becomes stable.
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
// The [`Parameterized`] implementation for [`HashMap`] is quite inefficient. That is because for most operations that
// require traversal of the [`HashMap`] we end up having to perform a sort of the results that we obtain (by key) to
// comply with the requirements of the [`Parameterized`].
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
        sorted_entries.sort_unstable_by(|(l, _), (r, _)| l.cmp(r));
        sorted_entries.into_iter().flat_map(|(k, v)| PathPrefixedParameterIterator {
            iterator: v.named_parameters(),
            segment: ParameterPathSegment::Key(format!("{k:?}")),
        })
    }

    fn named_parameters_mut(&mut self) -> Self::NamedParameterIteratorMut<'_, P> {
        // Note that each pointer below originates from a distinct `iter_mut()` item, and so the pointers hereare unique
        // and non-overlapping. We do not structurally modify the map after collecting the pointers and so they remain
        // valid for the duration of this traversal.
        let mut sorted_entries = self.iter_mut().map(|(k, v)| (k.clone(), v as *mut V)).collect::<Vec<_>>();
        sorted_entries.sort_unstable_by(|(l, _), (r, _)| l.cmp(r));
        sorted_entries.into_iter().flat_map(|(k, v)| PathPrefixedParameterIterator {
            iterator: (unsafe { &mut *v }).named_parameters_mut(),
            segment: ParameterPathSegment::Key(format!("{k:?}")),
        })
    }

    fn into_named_parameters(self) -> Self::NamedParameterIntoIterator<P> {
        let mut sorted_entries = self.into_iter().collect::<Vec<_>>();
        sorted_entries.sort_unstable_by(|(l, _), (r, _)| l.cmp(r));
        sorted_entries.into_iter().flat_map(|(k, v)| PathPrefixedParameterIterator {
            iterator: v.into_named_parameters(),
            segment: ParameterPathSegment::Key(format!("{k:?}")),
        })
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

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};

    /// Test-only shim to avoid taking a (circular) dependency on `ryft`.
    pub mod ryft {
        pub use ryft_macros::Parameterized;

        pub use crate::parameters::ParameterError;
        pub use crate::parameters::{
            Parameter, ParameterPath, ParameterPathSegment, Parameterized, ParameterizedFamily,
            PathPrefixedParameterIterator, Placeholder,
        };
    }

    use ryft::*;

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

        #[derive(Parameterized, Clone, Debug, PartialEq, Eq)]
        #[allow(dead_code)]
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

        // Test [`Parameterized::parameter_count`].
        assert_eq!(value.parameter_count(), 10);

        // Test [`Parameterized::parameter_structure`].
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

        // Test [`Parameterized::parameters`].
        assert_eq!(value.parameters().copied().collect::<Vec<_>>(), vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        // Test [`Parameterized::parameters_mut`].
        let mut value_clone = value.clone();
        for parameter in value_clone.parameters_mut() {
            *parameter += 100;
        }
        assert_eq!(
            value_clone.parameters().copied().collect::<Vec<_>>(),
            vec![101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
        );

        // Test [`Parameterized::into_parameters`].
        assert_eq!(value.clone().into_parameters().collect::<Vec<_>>(), vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        // Test [`Parameterized::named_parameters`].
        assert_eq!(
            value.named_parameters().map(|(path, parameter)| (path.to_string(), *parameter)).collect::<Vec<_>>(),
            expected_parameter_paths
                .iter()
                .cloned()
                .zip(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                .collect::<Vec<_>>()
        );

        // Test [`Parameterized::named_parameters_mut`].
        let mut value_clone = value.clone();
        let mut seen_parameter_paths = Vec::new();
        for (path, parameter) in value_clone.named_parameters_mut() {
            seen_parameter_paths.push(path.to_string());
            *parameter += path.len() as i32;
        }
        assert_eq!(seen_parameter_paths, expected_parameter_paths);
        assert_eq!(value_clone.parameters().copied().collect::<Vec<_>>(), vec![4, 5, 5, 7, 10, 10, 11, 11, 13, 13]);

        // Test [`Parameterized::into_named_parameters`].
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

        // Test [`Parameterized::parameter_paths`].
        assert_eq!(value.parameter_paths().map(|path| path.to_string()).collect::<Vec<_>>(), expected_parameter_paths);

        // Test [`Parameterized::from_parameters_with_remainder`].
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

        // Test [`Parameterized::from_parameters`].
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

        // Test [`Parameterized::from_named_parameters`].
        let mut parameters = value.clone().into_named_parameters().collect::<Vec<_>>();
        parameters.reverse();
        assert_eq!(Network::from_named_parameters(structure.clone(), parameters), Ok(value.clone()));

        // Test [`Parameterized::from_broadcasted_named_parameters`].
        assert_eq!(
            Network::from_broadcasted_named_parameters(
                structure.clone(),
                HashMap::from([
                    (ParameterPath::root(), 1),
                    (ParameterPath::root().field("controller"), 2),
                    (ParameterPath::root().field("controller").variant("Blend").field("branch"), 10),
                    (ParameterPath::root().field("controller").variant("Blend").field("branch").field("gain"), 20),
                    (ParameterPath::root().field("heads").index(1), 30),
                ])
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

        // Test [`Parameterized::broadcast_to_parameter_structure`].
        assert_eq!(
            42_i32.broadcast_to_parameter_structure::<(i32, (i32, i32))>((Placeholder, (Placeholder, Placeholder))),
            Ok((42, (42, 42))),
        );
        assert!(matches!(
            (1_i32, 2_i32).broadcast_to_parameter_structure::<i32>(Placeholder),
            Err(ParameterError::MissingParameters { .. }),
        ));

        // Test [`Parameterized::map_parameters`].
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

        // Test [`Parameterized::map_named_parameters`].
        assert_eq!(
            value
                .clone()
                .map_named_parameters(|path, parameter| i64::from(parameter) + (path.len() as i64))
                .unwrap()
                .into_parameters()
                .collect::<Vec<_>>(),
            vec![4i64, 5i64, 5i64, 7i64, 10i64, 10i64, 11i64, 11i64, 13i64, 13i64],
        );

        // Test [`Parameterized::filter_parameters`].
        assert_eq!(
            value
                .clone()
                .filter_parameters(|path, _| path.to_string().ends_with(".bias") || path.to_string().ends_with(".gain"))
                .unwrap()
                .into_parameters()
                .collect::<Vec<_>>(),
            vec![None, None, Some(3), None, None, Some(6), None, Some(8), None, Some(10)],
        );

        // Test [`Parameterized::partition_parameters`].
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

        // Test [`Parameterized::combine_parameters`].
        assert_eq!(Network::combine_parameters(structure.clone(), vec![partition_0, partition_1]), Ok(value.clone()));

        // Test [`Parameterized::replace_parameters`].
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
}
