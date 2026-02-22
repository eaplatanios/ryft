use std::{fmt::Debug, marker::PhantomData};

use half::{bf16, f16};
use paste::paste;

use crate::errors::Error;

// TODO(eaplatanios): Add thorough documentation for [Parameterized].
//  - [Parameter]s are the leafs and all `P: Parameter` are [Parameterized].
//  - `PhantomData<P: Parameterized>` is [Parameterized].
//  - `(P: Parameterized)`, `(P: Parameterized, P: Parameterized)`, ..., up to sized 12, are all [Parameterized].
//  - `[P: Parameterized; N]` is [Parameterized].
//  - `Vec<P: Parameterized>` is [Parameterized].
//  - TODO(eaplatanios): [HashMap]s.
//  - TODO(eaplatanios): [Box]s.
//  - `#[derive(Parameterized)]` provides support for custom structs and enums, which also support nested tuples
//    that mix [Parameterized] and non-[Parameterized] fields. However, they can only be nested within other tuples.
//    If, for example, they appear in e.g., `Vec<(P, usize)>`, then those tuples are not supported.
//  - Only the `: Parameter` bound is supported by the derive macro. No additional bounds are supported for `P`.
//
// TODO(eaplatanios): Add tests for each of the [Parameterized] implementations included in this file.

// TODO(eaplatanios): Add support for `named_parameters` which pairs each parameter with a path.
// TODO(eaplatanios): Support something like a `broadcast` operation (e.g., I want to use the same learning rate
//  for every sub-node from a specific point in the data structure). This is along the lines of what are called
//  PyTree prefixes in JAX. Related: https://jax.readthedocs.io/en/latest/pytrees.html#applying-optional-parameters-to-pytrees.
// TODO(eaplatanios): Borrow some of Equinox's tree manipulation capabilities.
//  Reference: https://docs.kidger.site/equinox/api/manipulation.

// For reference, in JAX, to register custom types as trees, we only need to implement these two functions:
// - flatten(tree) -> (children, aux_data)
// - unflatten(aux_data, children) -> tree

// TODO(eaplatanios): Document that this this an empty parameter acting as a placeholder for when we want to manipulate
//  parameter structures without having to worry about specific parameter types.
// TODO(eaplatanios): Document that this is a marker trait for parameter types (i.e., leaf nodes).
//  Furthermore, explain why we need this. Provide `Vec<P>` as a motivating example along with an explanation
//  for why something like specialization would need to be stable for us to support this.
// TODO(eaplatanios): Should `Parameter`s always have a static `Rank` and a static `DataType`?
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

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Placeholder;

impl Debug for Placeholder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<Parameter>")
    }
}

impl Parameter for Placeholder {}

// TODO(eaplatanios): `Vec<(P, non-P)>` is not supported.
// TODO(eaplatanios): Unit structs should be impossible.
// TODO(eaplatanios): Talk about the derive macro we have for this trait:
//  - We also provide a `#[derive(Parameter)]` macro for convenience.
//  - Supports both structs and enums already.
//  - The parameter type must be a generic type parameter bounded by [Parameter].
//  - There must be only one such generic type parameter. Not zero and not more than one.
//  - All fields that reference / depend on the parameter type are considered parameter fields.
//  - Attributes of generic parameters are not visited/transformed and they are always carried around as they are.
//  - We need a recursive helper in order to properly handle tuple types. Tuples are not [Parameterized]
//    themselves (that is done in order to avoid issues with blanket implementations since we only instantiate
//    [Parameterized] implementations using prespecified parameter types), but they are supported when nested
//    within other types, for which we are deriving [Parameterized] implementations.
//  - Configurable `macro_param_lifetime` and `macro_param_type`.
// TODO(eaplatanios): Document the following:
//  - Vec<P> is not Parameterized<P>. Vec<T: Parameterized<P>> is Parameterized<P>.
//  - HashMap<K, P> is not Parameterized<P>. HashMap<K, V: Parameterized<P>> is Parameterized<V>.
//  - Same goes for arrays and other collection types.
pub trait Parameterized<P: Parameter>: Sized {
    // TODO(eaplatanios): We need to prove that `Self::To<P> = Self`.
    // TODO(eaplatanios): What if `P` has additional trait bounds?
    type To<T: Parameter>: Parameterized<T, To<P> = Self> + Parameterized<T, To<Placeholder> = Self::To<Placeholder>>;
    // + Parameterized<T, To<JvpTracer<P>> = Self::To<JvpTracer<P>>>;

    // #![feature(associated_type_defaults)]
    // type ParamStructure = Self::To<ParamPlaceholder>;

    // TODO(eaplatanios): Explain that we use associated types instead of `RPITIT` in order to support
    //  deriving [Parameterized] for enums without the need to do any boxing. Though, is that really true?
    //  I mean the wrapping enum would have to box anyway...hmm...maybe enums should always use `Box<dyn Iterator>`.
    type ParamIterator<'t, T: 't + Parameter>: 't + Iterator<Item = &'t T>
    where
        // TODO(eaplatanios): Configure rustfmt to put these in the same line when possible.
        Self: 't;

    type ParamIteratorMut<'t, T: 't + Parameter>: 't + Iterator<Item = &'t mut T>
    where
        Self: 't;

    type ParamIntoIterator<T: Parameter>: Iterator<Item = T>;

    /// Returns the number of parameters in this [Parameterized] instance.
    fn param_count(&self) -> usize;

    fn param_structure(&self) -> Self::To<Placeholder>;

    fn params(&self) -> Self::ParamIterator<'_, P>;
    fn params_mut(&mut self) -> Self::ParamIteratorMut<'_, P>;
    fn into_params(self) -> Self::ParamIntoIterator<P>;

    fn from_params_with_remainder<I: Iterator<Item = P>>(
        structure: Self::To<Placeholder>,
        params: &mut I,
    ) -> Result<Self, Error>;

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
