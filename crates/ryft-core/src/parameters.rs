use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use half::{bf16, f16};
use paste::paste;

use crate::errors::Error;

// TODO(eaplatanios): Should `Parameter`s always have a static `Rank` and a static `DataType`?

// TODO(eaplatanios): Add support for `named_parameters` which pairs each parameter with a path.
// TODO(eaplatanios): Support something like a `broadcast` operation (e.g., I want to use the same learning rate
//  for every sub-node from a specific point in the data structure). This is along the lines of what are called
//  PyTree prefixes in JAX. Related: https://jax.readthedocs.io/en/latest/pytrees.html#applying-optional-parameters-to-pytrees.
// TODO(eaplatanios): Borrow some of Equinox's tree manipulation capabilities.
//  Reference: https://docs.kidger.site/equinox/api/manipulation.

// For reference, in JAX, to register custom types as trees, we only need to implement these two functions:
// - flatten(tree) -> (children, aux_data)
// - unflatten(aux_data, children) -> tree

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

// TODO(eaplatanios): `Vec<(P, non-P)>` is not supported.
// TODO(eaplatanios): Unit structs should be impossible.

// TODO(eaplatanios): Cover the following cases in the `Parameterized` documentation.
//  - HashMap<K, P> is not Parameterized<P>. HashMap<K, V: Parameterized<P>> is Parameterized<V>.
//  - Same goes for arrays and other collection types.

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
//    - Only the `: Parameter` bound is supported by the derive macro. No additional bounds are supported for `P`.

/// Recursively traversable parameter tree whose leaves are values that implement the [`Parameter`] marker trait.
///
/// A [`Parameterized`] value can be split into:
/// 1. Its shape-only representation via [`param_structure`](Self::param_structure), and
/// 2. An ordered stream of leaf parameters via [`params`](Self::params), [`params_mut`](Self::params_mut), or
///    [`into_params`](Self::into_params).
///
/// The same value can then be reconstructed using [`from_params`](Self::from_params) or
/// [`from_params_with_remainder`](Self::from_params_with_remainder).
///
/// # Implementations Provided In This Module
///
/// - Every `P: Parameter` is a leaf and therefore implements [`Parameterized<P>`].
/// - [`PhantomData<P>`] implements [`Parameterized<P>`] and contributes zero parameters.
/// - Tuples whose elements are all themselves [`Parameterized`] are supported for arities of 1 through 12.
/// - Arrays (`[T; N]`) and [`Vec<T>`] are supported when `T: Parameterized<P>`.
/// - Because containers are parameterized by element type, `Vec<P>` works whenever `P: Parameter` (as `P` itself is a
///   leaf that implements [`Parameterized<P>`]).
/// - [`std::collections::HashMap`] and [`Box`] are not supported yet.
///
/// # Derive Macros
///
/// `ryft-macros` provides two derives for working with this trait:
/// - `#[derive(Parameter)]` is a convenience derive for leaf types and expands to an empty [`Parameter`] `impl`.
/// - `#[derive(Parameterized)]` can generate implementations for structs and enums (unions are rejected).
///
/// `#[derive(Parameterized)]` currently assumes:
/// - Exactly one generic type parameter must be bounded by [`Parameter`].
/// - The parameter bound must be written as `Parameter` or `ryft::Parameter` (respecting any
///   `#[ryft(crate = "...")]` override).
/// - The parameter type cannot have additional bounds.
/// - All fields that reference the parameter type are treated as parameter fields.
/// - Parameter fields must be owned (references and pointers to the parameter type are rejected).
/// - Non-parameter fields must be [`Clone`] because [`param_structure`](Self::param_structure) clones them.
/// - The only supported macro attribute is container-level `#[ryft(crate = "...")]`; field/variant-level
///   `#[ryft(...)]` attributes are rejected.
/// - The names `__P` and `'__p` are reserved for generated code and cannot be used in the container generics.
/// - Nested tuples that mix parameterized and non-parameterized fields are supported within derived containers.
/// - Mixed tuples are not supported as direct items in generic containers covered by the blanket implementations in
///   this module (for example, `Vec<(P, usize)>`).
///
/// # Ordering Invariant
///
/// Implementations must preserve leaf order consistently across traversal and reconstruction. In other words, reading
/// parameters with [`params`](Self::params) and then rebuilding with [`from_params`](Self::from_params) must produce
/// the original value.
pub trait Parameterized<P: Parameter>: Sized {
    // TODO(eaplatanios): Can we enforce that `To<T>` is such that `To<T>::To<P> = Self` for any value of `T` and also
    //  `To<T>::To<R>` is equal to `To<R>` for any value of `R`.
    // TODO(eaplatanios): What if `P` has additional trait bounds? How can we represent `To` then?
    type To<T: Parameter>: Parameterized<T, To<P> = Self> + Parameterized<T, To<Placeholder> = Self::To<Placeholder>>;
    // + Parameterized<T, To<JvpTracer<P>> = Self::To<JvpTracer<P>>>;

    /// Shape-only representation of this [`Parameterized`] type with all parameter leaves replaced by [`Placeholder`].
    /// This is always set to `Self::To<Placeholder>`. The only reason this is not included here is that defaulted
    /// associated types are not supported in stable Rust.
    type ParamStructure: Parameterized<Placeholder, To<P> = Self>;

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
    fn param_structure(&self) -> Self::To<Placeholder>;

    /// Returns an iterator over references to all parameters in this value.
    fn params(&self) -> Self::ParamIterator<'_, P>;
    
    /// Returns an iterator over mutable references to all parameters in this value.
    fn params_mut(&mut self) -> Self::ParamIteratorMut<'_, P>;
    
    /// Consumes this value and returns an iterator over all parameters.
    fn into_params(self) -> Self::ParamIntoIterator<P>;

    /// Reconstructs a value from `structure`, consuming parameters from `params` and leaving any remainder untouched.
    fn from_params_with_remainder<I: Iterator<Item = P>>(
        structure: Self::To<Placeholder>,
        params: &mut I,
    ) -> Result<Self, Error>;

    /// Reconstructs a value from `structure` using all provided parameters.
    ///
    /// Returns [`Error::UnusedParams`] if there are leftover parameters.
    fn from_params<I: IntoIterator<Item = P>>(structure: Self::To<Placeholder>, params: I) -> Result<Self, Error> {
        let mut params = params.into_iter();
        let parameterized = Self::from_params_with_remainder(structure, &mut params)?;
        params.next().map(|_| Err(Error::UnusedParams)).unwrap_or_else(|| Ok(parameterized))
    }

    // TODO(eaplatanios): Document that this maps the parameters in this type.
    fn map_params<T: Parameter, F: FnMut(P) -> T>(self, map_fn: F) -> Result<Self::To<T>, Error> {
        Self::To::<T>::from_params(self.param_structure(), self.into_params().map(map_fn))
    }
}

impl<P: Parameter> Parameterized<P> for P {
    type To<T: Parameter> = T;
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

    fn param_structure(&self) -> Self::To<Placeholder> {
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
        _structure: Self::To<Placeholder>,
        params: &mut I,
    ) -> Result<Self, Error> {
        params.next().ok_or(Error::InsufficientParams { expected_count: 1 })
    }
}

impl<P: Parameter> Parameterized<P> for PhantomData<P> {
    type To<T: Parameter> = PhantomData<T>;
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

    fn param_structure(&self) -> Self::To<Placeholder> {
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
        _structure: Self::To<Placeholder>,
        _params: &mut I,
    ) -> Result<Self, Error> {
        Ok(PhantomData)
    }
}

// TODO(eaplatanios): Add implementation for [Box].

// Use declarative macros to provide implementations for tuples of [Parameterized] items. Note that if a tuple contains
// a mix of [Parameterized] and non-[Parameterized] items, then the generated implementations here will not cover it.
// Instead, such tuples are supported when nested within `struct`s or `enum`s tagged with `#[derive(Parameterized)]`
// as the `derive` macro for [Parameterized] provides special treatment for them.

macro_rules! tuple_parameterized_impl {
    ($($T:ident),*) => {
        paste! {
            impl<P: Parameter$(, $T: Parameterized<P>)*> Parameterized<P> for ($($T,)*) {
                type To<T: Parameter> = ($($T::To<T>,)*);
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

                fn param_structure(&self) -> Self::To<Placeholder> {
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
                    structure: Self::To<Placeholder>,
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

impl<P: Parameter, V: Parameterized<P>, const N: usize> Parameterized<P> for [V; N] {
    type To<T: Parameter> = [V::To<T>; N];
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

    fn param_structure(&self) -> Self::To<Placeholder> {
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
        structure: Self::To<Placeholder>,
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

impl<P: Parameter, V: Parameterized<P>> Parameterized<P> for Vec<V> {
    type To<T: Parameter> = Vec<V::To<T>>;
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

    fn param_structure(&self) -> Self::To<Placeholder> {
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
        structure: Self::To<Placeholder>,
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

// TODO(eaplatanios): Implement this for arrays, HashMap<K, _>, etc.
// TODO(eaplatanios): Add tests for each of the [Parameterized] implementations included in this file.

#[cfg(test)]
mod tests {
    use std::fmt::Debug;
    use std::marker::PhantomData;

    use crate::errors::Error;

    use super::{Parameterized, Placeholder};

    fn assert_roundtrip_parameterized<V>(value: V, expected_params: Vec<i32>)
    where
        V: Clone + Debug + PartialEq + Parameterized<i32>,
        V::To<Placeholder>: Clone + Debug + PartialEq,
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
        V::To<Placeholder>: Clone + Debug + PartialEq,
    {
        let mut mutable_value = value;
        for parameter in mutable_value.params_mut() {
            *parameter += 1;
        }
        let expected_after = expected_before.iter().map(|parameter| parameter + 1).collect::<Vec<_>>();
        assert_eq!(mutable_value.params().copied().collect::<Vec<_>>(), expected_after);
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
    fn test_from_params_reports_unused_params() {
        assert_eq!(<i32 as Parameterized<i32>>::from_params(Placeholder, vec![3, 4]), Err(Error::UnusedParams));
    }

    #[test]
    fn test_from_params_reports_insufficient_params_for_vec() {
        let structure = vec![Placeholder, Placeholder, Placeholder];
        let result = <Vec<i32> as Parameterized<i32>>::from_params(structure, vec![1, 2]);
        assert_eq!(result, Err(Error::InsufficientParams { expected_count: 3 }));
    }
}
