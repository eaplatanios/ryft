use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;

use half::{bf16, f16};
use paste::paste;

use crate::errors::Error;

// TODO(eaplatanios): Add support for `named_parameters` which pairs each parameter with a path.
// TODO(eaplatanios): Support something like a `broadcast` operation (e.g., I want to use the same learning rate
//  for every sub-node from a specific point in the data structure). This is along the lines of what are called
//  PyTree prefixes in JAX. Related: https://jax.readthedocs.io/en/latest/pytrees.html#applying-optional-parameters-to-pytrees.
// TODO(eaplatanios): Borrow some of Equinox's tree manipulation capabilities.
//  Reference: https://docs.kidger.site/equinox/api/manipulation.

// For reference, in JAX, to register custom types as trees, we only need to implement these two functions:
// - flatten(tree) -> (children, aux_data)
// - unflatten(aux_data, children) -> tree

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

/// Placeholder leaf type for [`Parameterized`] trees that is used represent [`Parameterized::param_structure`].
/// That is, it is used to replace every parameter leaf in a [`Parameterized`] type yielding a _shape-only_
/// representation that can later be used with [`Parameterized::from_params`] to instantiate a [`Parameterized`]
/// value with the same shape but different types of leaves/parameters.
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
    type To: Parameterized<P, Family = Self, ParamStructure = <Self as ParameterizedFamily<Placeholder>>::To>
    where
        Self: ParameterizedFamily<Placeholder>;
}

// TODO(eaplatanios): Talk about the derive macro we have for this trait:
//    - We also provide a `#[derive(Parameter)]` macro for convenience.
//    - Supports both structs and enums already.
//    - The parameter type must be a generic type parameter bounded by [Parameter].
//    - There must be only one such generic type parameter. Not zero and not more than one.
//    - All fields that reference / depend on the parameter type are considered parameter fields.
//    - Attributes of generic parameters are not visited/transformed and they are always carried around as they are.
//    - We need a recursive helper in order to properly handle tuple types. Tuples are not [Parameterized]
//      themselves (that is done in order to avoid issues with blanket implementations since we only instantiate
//      [Parameterized] implementations using prespecified parameter types), but they are supported when nested
//      within other types, for which we are deriving [Parameterized] implementations.
//    - Configurable `macro_param_lifetime` and `macro_param_type`.
//    - `#[derive(Parameterized)]` provides support for custom structs and enums, which also support nested tuples
//      that mix [Parameterized] and non-[Parameterized] fields. However, they can only be nested within other tuples.
//      If, for example, they appear in e.g., `Vec<(P, usize)>`, then those tuples are not supported.

/// Recursively traversable data structure that contains nested [`Parameter`]s of type `P`. A [`Parameterized`] value
/// can be thought of as consisting of two parts:
/// 
///  1. Its _structure_ which can be obtained via [`Self::param_structure`].
///  2. The _parameters_ nested within that structure, which can be obtained via [`Self::params`], [`Self::params_mut`],
///     and [`Self::into_params`].
///
/// Given a [`Self::ParamStructure`] and an ordered collection of [`Parameter`]s, of potentially a different type than
/// `P`, new instances of this type can be constructed using [`Parameterized::from_params_with_remainder`]. Another way
/// to think of [`Parameterized`] types is as tree-structured containers of [`Parameter`]s.
///
/// This is analogous to a [JAX pytree](https://docs.jax.dev/en/latest/pytrees.html): JAX represents a pytree as
/// `leaves + treedef`, while this trait represents a value as parameters plus [`Self::ParamStructure`].
///
/// # Mapping To JAX Pytrees
///
/// - JAX `tree.flatten(x)` corresponds to [`params`](Self::params) (the leaves) plus
///   [`param_structure`](Self::param_structure) (the tree definition).
/// - JAX `tree.unflatten(treedef, leaves)` corresponds to [`from_params`](Self::from_params) or
///   [`from_params_with_remainder`](Self::from_params_with_remainder).
/// - JAX `tree.map(f, x)` corresponds to [`map_params`](Self::map_params).
///
/// # Examples
///
/// Flatten into leaves and structure:
///
/// ```rust
/// use ryft_core::parameters::{Parameterized, Placeholder};
///
/// let value = vec![(1.0_f32, 2.0_f32), (3.0_f32, 4.0_f32)];
/// assert_eq!(value.params().copied().collect::<Vec<_>>(), vec![1.0, 2.0, 3.0, 4.0]);
/// assert_eq!(value.param_structure(), vec![(Placeholder, Placeholder), (Placeholder, Placeholder)]);
/// ```
///
/// Rebuild the same structure with different parameter values:
///
/// ```rust
/// use ryft_core::parameters::{Parameterized, Placeholder};
///
/// let structure = vec![(Placeholder, Placeholder), (Placeholder, Placeholder)];
/// let rebuilt = <Vec<(f32, f32)> as Parameterized<f32>>::from_params(
///     structure,
///     vec![10.0_f32, 20.0_f32, 30.0_f32, 40.0_f32],
/// )?;
/// assert_eq!(rebuilt, vec![(10.0, 20.0), (30.0, 40.0)]);
/// # Ok::<(), ryft_core::errors::Error>(())
/// ```
///
/// Apply the same transformation to every leaf while preserving structure:
///
/// ```rust
/// use ryft_core::parameters::Parameterized;
///
/// let value = vec![(1_i32, 2_i32), (3_i32, 4_i32)];
/// let shifted: Vec<(i64, i64)> = value.map_params(|v| i64::from(v) + 100)?;
/// assert_eq!(shifted, vec![(101, 102), (103, 104)]);
/// # Ok::<(), ryft_core::errors::Error>(())
/// ```
///
/// For additional intuition and patterns, see JAX's docs on
/// [pytrees](https://docs.jax.dev/en/latest/pytrees.html) and
/// [custom pytree nodes](https://docs.jax.dev/en/latest/custom_pytrees.html).
///
/// # Implementations Provided In This Module
///
/// - Every `P: Parameter` is a leaf and therefore implements [`Parameterized<P>`].
/// - [`PhantomData<P>`] implements [`Parameterized<P>`] and contributes zero parameters.
/// - Tuples whose elements are all themselves [`Parameterized`] are supported for arities of 1 through 12.
/// - Arrays (`[T; N]`) and [`Vec<T>`] are supported when `T: Parameterized<P>`.
/// - [`HashMap<K, T, S>`] is supported when `K: Clone + Eq + std::hash::Hash`,
///   `S: std::hash::BuildHasher + Clone`, and `T: Parameterized<P>`.
/// - [`Box<T>`] is intentionally not supported (see the coherence note below).
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
/// parameters with [`params`](Self::params) and then rebuilding with [`from_params`](Self::from_params) must produce
/// the original value.
pub trait Parameterized<P: Parameter>: Sized {
    /// [`ParameterizedFamily`] that this type belongs to and which can be used to reparameterize it.
    type Family: ParameterizedFamily<P, To = Self> + ParameterizedFamily<Placeholder, To = Self::ParamStructure>;

    /// Reparameterized form of this [`Parameterized`] type with all of its nested `P` types replaced by `T`. This
    /// preserves the same [`Family`](Self::Family) and [`ParamStructure`](Self::ParamStructure), and is such that
    /// reparameterizing back to `P` recovers [`Self`].
    type To<T: Parameter>: Parameterized<T, Family = Self::Family, ParamStructure = Self::ParamStructure>
        + SameAs<<Self::Family as ParameterizedFamily<T>>::To>
    where
        Self::Family: ParameterizedFamily<T>;

    /// Shape-only representation of this [`Parameterized`] type with all parameter leaves replaced by [`Placeholder`].
    /// This is always set to `Self::To<Placeholder>`. The only reason this is not included here is that defaulted
    /// associated types are not supported in stable Rust.
    type ParamStructure: Parameterized<Placeholder, Family = Self::Family, To<P> = Self> + SameAs<Self::To<Placeholder>>;

    /// Iterator returned by [`params`](Self::params) for a borrow of the underlying [`Parameter`]s with lifetime `'t`.
    /// This is an associated type instead of an `impl Iterator` in the corresponding function signature, so that
    /// implementations can expose and reuse a concrete iterator type. In particular, `#[derive(Parameterized)]` for
    /// enums synthesizes concrete enum iterators here, avoiding an additional heap allocation and dynamic dispatch.
    type ParamIterator<'t, T: 't + Parameter>: 't + Iterator<Item = &'t T>
    where
        Self: 't;

    /// Iterator returned by [`params_mut`](Self::params_mut) for a mutable borrow of the underlying [`Parameter`]s with
    /// lifetime `'t`. Similar to [`ParamIterator`](Self::ParamIterator), this is an associated type instead of an
    /// `impl Iterator` in the corresponding function signature, so that implementations can expose and reuse a concrete
    /// iterator type, potentially avoiding additional heap allocations and dynamic dispatch.
    type ParamIteratorMut<'t, T: 't + Parameter>: 't + Iterator<Item = &'t mut T>
    where
        Self: 't;

    /// Iterator returned by [`into_params`](Self::into_params), consuming `self` and returning the underlying
    /// [`Parameter`]s. Similar to [`ParamIterator`](Self::ParamIterator), this is an associated type instead of
    /// an `impl Iterator` in the corresponding function signature, so that implementations can expose and reuse
    /// a concrete iterator type, potentially avoiding additional heap allocations and dynamic dispatch.
    type ParamIntoIterator<T: Parameter>: Iterator<Item = T>;

    /// Returns the number of parameters in this [Parameterized] instance.
    fn param_count(&self) -> usize;

    /// Returns the parameter structure of this value by replacing all leaves with [`Placeholder`]s.
    fn param_structure(&self) -> Self::ParamStructure;

    /// Returns an iterator over references to all parameters in this value.
    fn params(&self) -> Self::ParamIterator<'_, P>;

    /// Returns an iterator over mutable references to all parameters in this value.
    fn params_mut(&mut self) -> Self::ParamIteratorMut<'_, P>;

    /// Consumes this value and returns an iterator over all parameters.
    fn into_params(self) -> Self::ParamIntoIterator<P>;

    /// Reconstructs a value from `structure`, consuming parameters from `params` and leaving any remainder untouched.
    fn from_params_with_remainder<I: Iterator<Item = P>>(
        structure: Self::ParamStructure,
        params: &mut I,
    ) -> Result<Self, Error>;

    /// Reconstructs a value from `structure` using all provided parameters.
    ///
    /// Returns [`Error::UnusedParams`] if there are leftover parameters.
    fn from_params<I: IntoIterator<Item = P>>(structure: Self::ParamStructure, params: I) -> Result<Self, Error> {
        let mut params = params.into_iter();
        let parameterized = Self::from_params_with_remainder(structure, &mut params)?;
        params.next().map(|_| Err(Error::UnusedParams)).unwrap_or_else(|| Ok(parameterized))
    }

    /// Maps each nested [`Parameter`] of type `P` in this value using the provided `map_fn` to a [`Parameter`] of type
    /// `T`, while preserving the [`Parameterized`] tree structure of this type. Nested parameters are visited in the
    /// same order as [`Self::params`], [`Self::params_mut`], and [`Self::into_params`].
    fn map_params<T: Parameter, F: FnMut(P) -> T>(self, map_fn: F) -> Result<Self::To<T>, Error>
    where
        Self::Family: ParameterizedFamily<T>,
    {
        Self::To::<T>::from_params(self.param_structure(), self.into_params().map(map_fn))
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

    type ParamStructure = Self::To<Placeholder>;

    type ParamIterator<'t, T: 't + Parameter>
        = std::iter::Once<&'t T>
    where
        Self: 't;

    type ParamIteratorMut<'t, T: 't + Parameter>
        = std::iter::Once<&'t mut T>
    where
        Self: 't;

    type ParamIntoIterator<T: Parameter> = std::iter::Once<T>;

    fn param_count(&self) -> usize {
        1
    }

    fn param_structure(&self) -> Self::ParamStructure {
        Placeholder
    }

    fn params(&self) -> Self::ParamIterator<'_, P> {
        std::iter::once(self)
    }

    fn params_mut(&mut self) -> Self::ParamIteratorMut<'_, P> {
        std::iter::once(self)
    }

    fn into_params(self) -> Self::ParamIntoIterator<P> {
        std::iter::once(self)
    }

    fn from_params_with_remainder<I: Iterator<Item = P>>(
        _structure: Self::ParamStructure,
        params: &mut I,
    ) -> Result<Self, Error> {
        params.next().ok_or(Error::InsufficientParams { expected_count: 1 })
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

    type ParamStructure = Self::To<Placeholder>;

    type ParamIterator<'t, T: 't + Parameter>
        = std::iter::Empty<&'t T>
    where
        Self: 't;

    type ParamIteratorMut<'t, T: 't + Parameter>
        = std::iter::Empty<&'t mut T>
    where
        Self: 't;

    type ParamIntoIterator<T: Parameter> = std::iter::Empty<T>;

    fn param_count(&self) -> usize {
        0
    }

    fn param_structure(&self) -> Self::ParamStructure {
        PhantomData
    }

    fn params(&self) -> Self::ParamIterator<'_, P> {
        std::iter::empty()
    }

    fn params_mut(&mut self) -> Self::ParamIteratorMut<'_, P> {
        std::iter::empty()
    }

    fn into_params(self) -> Self::ParamIntoIterator<P> {
        std::iter::empty()
    }

    fn from_params_with_remainder<I: Iterator<Item = P>>(
        _structure: Self::ParamStructure,
        _params: &mut I,
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
    ($($T:ident),*) => {
        paste! {
            impl<P: Parameter$(, $T: Parameterized<P>)*> Parameterized<P> for ($($T,)*)
            {
                type Family = ($($T::Family,)*);

                type To<T: Parameter> = <Self::Family as ParameterizedFamily<T>>::To
                where
                    Self::Family: ParameterizedFamily<T>;

                type ParamStructure = Self::To<Placeholder>;

                type ParamIterator<'t, T: 't + Parameter> =
                    tuple_param_iterator_ty!('t, T, ($($T,)*))
                where Self: 't;

                type ParamIteratorMut<'t, T: 't + Parameter> =
                    tuple_param_iterator_mut_ty!('t, T, ($($T,)*))
                where Self: 't;

                type ParamIntoIterator<T: Parameter> = tuple_param_into_iterator_ty!(T, ($($T,)*));

                fn param_count(&self) -> usize {
                    let ($([<$T:lower>],)*) = &self;
                    $([<$T:lower>].param_count()+)* 0usize
                }

                fn param_structure(&self) -> Self::ParamStructure {
                    let ($([<$T:lower>],)*) = &self;
                    ($([<$T:lower>].param_structure(),)*)
                }

                fn params(&self) -> Self::ParamIterator<'_, P> {
                    let ($([<$T:lower>],)*) = self;
                    tuple_param_iterator!(P, ($([<$T:lower>],)*))
                }

                fn params_mut(&mut self) -> Self::ParamIteratorMut<'_, P> {
                    let ($([<$T:lower>],)*) = self;
                    tuple_param_iterator_mut!(P, ($([<$T:lower>],)*))
                }

                fn into_params(self) -> Self::ParamIntoIterator<P> {
                    let ($([<$T:lower>],)*) = self;
                    tuple_param_into_iterator!(P, ($([<$T:lower>],)*))
                }

                fn from_params_with_remainder<I: Iterator<Item = P>>(
                    structure: Self::ParamStructure,
                    params: &mut I,
                ) -> Result<Self, Error> {
                    let ($([<$T:lower _field>],)*) = structure;
                    $(let [<$T:lower>] = $T::from_params_with_remainder([<$T:lower _field>], params)?;)*
                    Ok(($([<$T:lower>],)*))
                }
            }
        }
    };
}

macro_rules! tuple_param_iterator_ty {
    ($t:lifetime, $T:ty, ()) => {
        std::iter::Empty<&$t $T>
    };

    ($t:lifetime, $T:ty, ($head:ident, $($tail:ident,)*)) => {
        std::iter::Chain<$head::ParamIterator<$t, $T>, tuple_param_iterator_ty!($t, $T, ($($tail,)*))>
    };
}

macro_rules! tuple_param_iterator_mut_ty {
    ($t:lifetime, $T:ty, ()) => {
        std::iter::Empty<&$t mut $T>
    };

    ($t:lifetime, $T:ty, ($head:ident, $($tail:ident,)*)) => {
        std::iter::Chain<$head::ParamIteratorMut<$t, $T>, tuple_param_iterator_mut_ty!($t, $T, ($($tail,)*))>
    };
}

macro_rules! tuple_param_into_iterator_ty {
    ($T:ty, ()) => {
        std::iter::Empty<$T>
    };

    ($T:ty, ($head:ident, $($tail:ident,)*)) => {
        std::iter::Chain<$head::ParamIntoIterator<$T>, tuple_param_into_iterator_ty!($T, ($($tail,)*))>
    };
}

macro_rules! tuple_param_iterator {
    ($T:tt, ()) => {
        std::iter::empty::<&'_ $T>()
    };

    ($T:tt, ($head:ident, $($tail:ident,)*)) => {
        $head.params().chain(tuple_param_iterator!($T, ($($tail,)*)))
    };
}

macro_rules! tuple_param_iterator_mut {
    ($T:tt, ()) => {
        std::iter::empty::<&'_ mut $T>()
    };

    ($T:tt, ($head:ident, $($tail:ident,)*)) => {
        $head.params_mut().chain(tuple_param_iterator_mut!($T, ($($tail,)*)))
    };
}

macro_rules! tuple_param_into_iterator {
    ($T:tt, ()) => {
        std::iter::empty::<$T>()
    };

    ($T:tt, ($head:ident, $($tail:ident,)*)) => {
        $head.into_params().chain(tuple_param_into_iterator!($T, ($($tail,)*)))
    };
}

tuple_parameterized_impl!(T0);
tuple_parameterized_impl!(T0, T1);
tuple_parameterized_impl!(T0, T1, T2);
tuple_parameterized_impl!(T0, T1, T2, T3);
tuple_parameterized_impl!(T0, T1, T2, T3, T4);
tuple_parameterized_impl!(T0, T1, T2, T3, T4, T5);
tuple_parameterized_impl!(T0, T1, T2, T3, T4, T5, T6);
tuple_parameterized_impl!(T0, T1, T2, T3, T4, T5, T6, T7);
tuple_parameterized_impl!(T0, T1, T2, T3, T4, T5, T6, T7, T8);
tuple_parameterized_impl!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9);
tuple_parameterized_impl!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10);
tuple_parameterized_impl!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11);

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

    type ParamStructure = Self::To<Placeholder>;

    type ParamIterator<'t, T: 't + Parameter>
        = std::iter::FlatMap<
        std::slice::Iter<'t, V>,
        <V as Parameterized<P>>::ParamIterator<'t, T>,
        fn(&'t V) -> <V as Parameterized<P>>::ParamIterator<'t, T>,
    >
    where
        Self: 't;

    type ParamIteratorMut<'t, T: 't + Parameter>
        = std::iter::FlatMap<
        std::slice::IterMut<'t, V>,
        <V as Parameterized<P>>::ParamIteratorMut<'t, T>,
        fn(&'t mut V) -> <V as Parameterized<P>>::ParamIteratorMut<'t, T>,
    >
    where
        Self: 't;

    type ParamIntoIterator<T: Parameter> = std::iter::FlatMap<
        std::array::IntoIter<V, N>,
        <V as Parameterized<P>>::ParamIntoIterator<T>,
        fn(V) -> <V as Parameterized<P>>::ParamIntoIterator<T>,
    >;

    fn param_count(&self) -> usize {
        self.iter().map(|value| value.param_count()).sum()
    }

    fn param_structure(&self) -> Self::ParamStructure {
        std::array::from_fn(|i| self[i].param_structure())
    }

    fn params(&self) -> Self::ParamIterator<'_, P> {
        self.iter().flat_map(V::params)
    }

    fn params_mut(&mut self) -> Self::ParamIteratorMut<'_, P> {
        self.iter_mut().flat_map(V::params_mut)
    }

    fn into_params(self) -> Self::ParamIntoIterator<P> {
        self.into_iter().flat_map(V::into_params)
    }

    fn from_params_with_remainder<I: Iterator<Item = P>>(
        structure: Self::ParamStructure,
        params: &mut I,
    ) -> Result<Self, Error> {
        // Make this more efficient by using [std::array::try_from_fn] once it becomes stable.
        // Tracking issue: https://github.com/rust-lang/rust/issues/89379.
        let values = structure
            .into_iter()
            .map(|value_structure| V::from_params_with_remainder(value_structure, params))
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

    type ParamStructure = Self::To<Placeholder>;

    type ParamIterator<'t, T: 't + Parameter>
        = std::iter::FlatMap<
        std::slice::Iter<'t, V>,
        <V as Parameterized<P>>::ParamIterator<'t, T>,
        fn(&'t V) -> <V as Parameterized<P>>::ParamIterator<'t, T>,
    >
    where
        Self: 't;

    type ParamIteratorMut<'t, T: 't + Parameter>
        = std::iter::FlatMap<
        std::slice::IterMut<'t, V>,
        <V as Parameterized<P>>::ParamIteratorMut<'t, T>,
        fn(&'t mut V) -> <V as Parameterized<P>>::ParamIteratorMut<'t, T>,
    >
    where
        Self: 't;

    type ParamIntoIterator<T: Parameter> = std::iter::FlatMap<
        std::vec::IntoIter<V>,
        <V as Parameterized<P>>::ParamIntoIterator<T>,
        fn(V) -> <V as Parameterized<P>>::ParamIntoIterator<T>,
    >;

    fn param_count(&self) -> usize {
        self.iter().map(|value| value.param_count()).sum()
    }

    fn param_structure(&self) -> Self::ParamStructure {
        self.iter().map(|value| value.param_structure()).collect()
    }

    fn params(&self) -> Self::ParamIterator<'_, P> {
        self.iter().flat_map(|value| value.params())
    }

    fn params_mut(&mut self) -> Self::ParamIteratorMut<'_, P> {
        self.iter_mut().flat_map(|value| value.params_mut())
    }

    fn into_params(self) -> Self::ParamIntoIterator<P> {
        self.into_iter().flat_map(|value| value.into_params())
    }

    fn from_params_with_remainder<I: Iterator<Item = P>>(
        structure: Self::ParamStructure,
        params: &mut I,
    ) -> Result<Self, Error> {
        let expected_count = structure.len();
        let mut values = Vec::new();
        values.reserve_exact(expected_count);
        for value_structure in structure {
            values.push(V::from_params_with_remainder(value_structure, params).map_err(|error| match error {
                Error::InsufficientParams { .. } => Error::InsufficientParams { expected_count },
                error => error,
            })?);
        }
        Ok(values)
    }
}

pub struct HashMapParameterizedFamily<K, F, S>(PhantomData<(K, F, S)>);

impl<
    P: Parameter,
    K: Clone + Eq + Hash,
    F: ParameterizedFamily<P> + ParameterizedFamily<Placeholder>,
    S: BuildHasher + Clone,
> ParameterizedFamily<P> for HashMapParameterizedFamily<K, F, S>
{
    type To = HashMap<K, <F as ParameterizedFamily<P>>::To, S>;
}

impl<P: Parameter, K: Clone + Eq + Hash, V: Parameterized<P>, S: BuildHasher + Clone> Parameterized<P>
    for HashMap<K, V, S>
{
    type Family = HashMapParameterizedFamily<K, V::Family, S>;

    type To<T: Parameter>
        = <Self::Family as ParameterizedFamily<T>>::To
    where
        Self::Family: ParameterizedFamily<T>;

    type ParamStructure = Self::To<Placeholder>;

    type ParamIterator<'t, T: 't + Parameter>
        = std::iter::FlatMap<
        std::collections::hash_map::Values<'t, K, V>,
        <V as Parameterized<P>>::ParamIterator<'t, T>,
        fn(&'t V) -> <V as Parameterized<P>>::ParamIterator<'t, T>,
    >
    where
        Self: 't;

    type ParamIteratorMut<'t, T: 't + Parameter>
        = std::iter::FlatMap<
        std::collections::hash_map::ValuesMut<'t, K, V>,
        <V as Parameterized<P>>::ParamIteratorMut<'t, T>,
        fn(&'t mut V) -> <V as Parameterized<P>>::ParamIteratorMut<'t, T>,
    >
    where
        Self: 't;

    type ParamIntoIterator<T: Parameter> = std::iter::FlatMap<
        std::collections::hash_map::IntoValues<K, V>,
        <V as Parameterized<P>>::ParamIntoIterator<T>,
        fn(V) -> <V as Parameterized<P>>::ParamIntoIterator<T>,
    >;

    fn param_count(&self) -> usize {
        self.values().map(|value| value.param_count()).sum()
    }

    fn param_structure(&self) -> Self::ParamStructure {
        let mut structure = HashMap::with_capacity_and_hasher(self.len(), self.hasher().clone());
        structure.extend(self.iter().map(|(key, value)| (key.clone(), value.param_structure())));
        structure
    }

    fn params(&self) -> Self::ParamIterator<'_, P> {
        self.values().flat_map(V::params)
    }

    fn params_mut(&mut self) -> Self::ParamIteratorMut<'_, P> {
        self.values_mut().flat_map(V::params_mut)
    }

    fn into_params(self) -> Self::ParamIntoIterator<P> {
        self.into_values().flat_map(V::into_params)
    }

    fn from_params_with_remainder<I: Iterator<Item = P>>(
        structure: Self::ParamStructure,
        params: &mut I,
    ) -> Result<Self, Error> {
        let expected_count = structure.len();
        let mut values = HashMap::with_capacity_and_hasher(expected_count, structure.hasher().clone());
        for (key, value_structure) in structure {
            values.insert(
                key,
                V::from_params_with_remainder(value_structure, params).map_err(|error| match error {
                    Error::InsufficientParams { .. } => Error::InsufficientParams { expected_count },
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

    use super::{Parameter, Parameterized, Placeholder};

    fn assert_roundtrip_parameterized<V>(value: V, expected_params: Vec<i32>)
    where
        V: Clone + Debug + PartialEq + Parameterized<i32>,
        V::ParamStructure: Clone + Debug + PartialEq,
    {
        assert_eq!(value.param_count(), expected_params.len());
        assert_eq!(value.params().copied().collect::<Vec<_>>(), expected_params);
        assert_eq!(value.clone().into_params().collect::<Vec<_>>(), expected_params);

        let structure = value.param_structure();
        assert_eq!(V::from_params(structure.clone(), expected_params.clone()), Ok(value.clone()));

        let mut params_with_remainder = expected_params.iter().copied().chain(std::iter::once(-1));
        assert_eq!(V::from_params_with_remainder(structure, &mut params_with_remainder), Ok(value));
        assert_eq!(params_with_remainder.collect::<Vec<_>>(), vec![-1]);
    }

    fn assert_params_mut_increments<V>(value: V, expected_before: Vec<i32>)
    where
        V: Clone + Debug + PartialEq + Parameterized<i32>,
        V::ParamStructure: Clone + Debug + PartialEq,
    {
        let mut mutable_value = value;
        for parameter in mutable_value.params_mut() {
            *parameter += 1;
        }
        let expected_after = expected_before.iter().map(|parameter| parameter + 1).collect::<Vec<_>>();
        assert_eq!(mutable_value.params().copied().collect::<Vec<_>>(), expected_after);
    }

    mod derive_support {
        pub use crate::errors::Error;
        pub use crate::parameters::{Parameter, Parameterized, ParameterizedFamily, Placeholder};
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

    macro_rules! assert_tuple_impl {
        (($($value:expr),+ $(,)?)) => {{
            assert_roundtrip_parameterized(($($value,)+), vec![$($value),+]);
            assert_params_mut_increments(($($value,)+), vec![$($value),+]);
        }};
    }

    #[test]
    fn test_leaf_parameterized_impl() {
        assert_roundtrip_parameterized(7, vec![7]);
        assert_params_mut_increments(7, vec![7]);
    }

    #[test]
    fn test_phantom_data_parameterized_impl() {
        assert_roundtrip_parameterized(PhantomData::<i32>, vec![]);
        assert_params_mut_increments(PhantomData::<i32>, vec![]);
        assert_eq!(PhantomData::<i32>.param_structure(), PhantomData::<Placeholder>);
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
        assert_params_mut_increments([1, 2, 3], vec![1, 2, 3]);
        assert_roundtrip_parameterized([(1, 2), (3, 4)], vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_vec_parameterized_impl() {
        assert_roundtrip_parameterized(vec![1, 2, 3], vec![1, 2, 3]);
        assert_params_mut_increments(vec![1, 2, 3], vec![1, 2, 3]);
        assert_roundtrip_parameterized(vec![(1, 2), (3, 4)], vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_hash_map_parameterized_impl() {
        let mut value = HashMap::new();
        value.insert("left", (1, 2));
        value.insert("right", (3, 4));

        let expected_params = value.params().copied().collect::<Vec<_>>();
        assert_roundtrip_parameterized(value.clone(), expected_params.clone());
        assert_params_mut_increments(value, expected_params);
    }

    #[test]
    fn test_derive_supports_additional_parameter_bounds() {
        let value = DomainRates { first: Rate32(3), second: Rate32(7) };
        assert_eq!(value.param_count(), 2);
        assert_eq!(value.params().copied().collect::<Vec<_>>(), vec![Rate32(3), Rate32(7)]);
        assert_eq!(value.param_structure(), DomainRates { first: Placeholder, second: Placeholder });

        let mapped: DomainRates<Rate64> = value.map_params(|rate| Rate64(i64::from(rate.0) * 10)).unwrap();
        assert_eq!(mapped, DomainRates { first: Rate64(30), second: Rate64(70) });
    }

    #[test]
    fn test_derive_supports_additional_parameter_bounds_in_where_clause() {
        let value = DomainRatesVec {
            values: vec![
                DomainRates { first: Rate32(1), second: Rate32(2) },
                DomainRates { first: Rate32(3), second: Rate32(4) },
            ],
        };
        assert_eq!(value.param_count(), 4);
        assert_eq!(
            value.param_structure(),
            DomainRatesVec {
                values: vec![
                    DomainRates { first: Placeholder, second: Placeholder },
                    DomainRates { first: Placeholder, second: Placeholder },
                ],
            }
        );
        assert_eq!(
            DomainRatesVec::from_params(value.param_structure(), vec![Rate32(1), Rate32(2), Rate32(3), Rate32(4)],),
            Ok(value.clone())
        );

        let mapped: DomainRatesVec<Rate64> = value.map_params(|rate| Rate64(i64::from(rate.0) + 5)).unwrap();
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
    fn test_from_params_reports_unused_params() {
        assert_eq!(<i32 as Parameterized<i32>>::from_params(Placeholder, vec![3, 4]), Err(Error::UnusedParams));
    }

    #[test]
    fn test_from_params_reports_insufficient_params_for_vec() {
        let structure = vec![Placeholder, Placeholder, Placeholder];
        let result = <Vec<i32> as Parameterized<i32>>::from_params(structure, vec![1, 2]);
        assert_eq!(result, Err(Error::InsufficientParams { expected_count: 3 }));
    }

    #[test]
    fn test_from_params_reports_insufficient_params_for_hash_map() {
        let mut structure = HashMap::new();
        structure.insert("left", Placeholder);
        structure.insert("right", Placeholder);
        structure.insert("middle", Placeholder);
        let result = <HashMap<&str, i32> as Parameterized<i32>>::from_params(structure, vec![1, 2]);
        assert_eq!(result, Err(Error::InsufficientParams { expected_count: 3 }));
    }
}
