//! Declarative macros that are shared by the modules of this crate.

// TODO(eaplatanios): Review this module.

/// Defines an optimizer state struct whose leaves are named by the paths of a parameter structure, together with the
/// [`Parameterized`](ryft_core::Parameterized) implementation that lets it cross compiled function boundaries and be
/// checkpointed leaf by leaf.
///
/// The first type parameter of the struct, `P`, is the leaf type (e.g., an array value), and every other type parameter
/// (usually just `S`) is a structure of leaves of type `P` (e.g., the parameter structure, whose shape per-leaf
/// statistics take, or the state of an inner optimizer). Every field must have a type that is
/// [`Parameterized<P>`](ryft_core::Parameterized) whenever the structure parameters are, such as `P` itself for a
/// rank-0 step count, `S` for per-leaf statistics, or another state type of this crate. States that hold both per-leaf
/// statistics and the state of an inner optimizer declare two structure parameters (e.g., `<P, S, I>`). Structure
/// parameters must not be named `T` or `Remainder`, which the generated implementations use. The derive macro
/// of [`Parameterized`](ryft_core::Parameterized) cannot be used for such structs, because it recognizes parameter
/// fields only by whether their types mention `P`, and so it would treat a field of type `S` as non-parameter data.
///
/// The generated implementation follows the one that `ryft-core` provides for [`Vec`]. Its family is the zero sized
/// marker struct that the invocation names after `with family`, whose reparameterization to a leaf type `T` replaces
/// `P` with `T` and every structure parameter with its reparameterization to `T`. Leaves are visited field by field in
/// declaration order, and the path of every leaf starts with the name of its field (e.g., `$.first_moments.weights`).
/// The fields stay private, so the module that invokes the macro provides the constructors and accessors.
///
/// A state whose fields do not mention `P` and every structure parameter (e.g., one that holds only per-leaf statistics
/// of type `S`, or only states that do not depend on the parameter structure) names a marker field after `and marker`,
/// which the macro declares with type `PhantomData<fn() -> (P, (S,))>` (with every structure parameter in the inner
/// tuple) and which holds no leaves. Such states are constructed with
/// `PhantomData` as the value of that field.
///
/// # Example
///
/// ```ignore
/// parameterized_state! {
///     /// State of a transformation that tracks a step count and one statistic per leaf.
///     #[derive(Clone, Debug, PartialEq, Eq, Hash)]
///     pub struct ExampleState<P, S> with family ExampleStateFamily {
///         /// Rank-0 `i32` array that holds the number of updates applied so far.
///         count: P,
///
///         /// Statistics with the structure of the parameters.
///         statistics: S,
///     }
/// }
/// ```
macro_rules! parameterized_state {
    // Definition form, which receives the structure parameters once more as the parenthesized group `$structures`
    // (because the marker field cannot repeat them inside its own optional repetition), and which defines the state
    // struct, its family marker, and their implementations.
    (
        @define $structures:tt
        $(#[$attribute:meta])*
        $visibility:vis struct $name:ident<P, $($structure:ident),+> with family $family:ident
            $(and marker $marker:ident)? {
            $($(#[$field_attribute:meta])* $field:ident: $field_type:ty),+
        }
    ) => {
        $(#[$attribute])*
        $visibility struct $name<P: ::ryft_core::Parameter, $($structure),+> {
            $($(#[$field_attribute])* $field: $field_type,)+
            $(
                /// Marker of the leaf type `P` and the structure parameters, which the other fields of this state do
                /// not all mention.
                $marker: ::std::marker::PhantomData<fn() -> (P, $structures)>,
            )?
        }

        #[doc = concat!(
            "[`ParameterizedFamily`](::ryft_core::ParameterizedFamily) of [`",
            stringify!($name),
            "`], which is public only because it is the family of its public ",
            "[`Parameterized`](::ryft_core::Parameterized) implementation.",
        )]
        #[doc(hidden)]
        $visibility struct $family<$($structure),+>(::std::marker::PhantomData<fn() -> ($($structure,)+)>);

        impl<T, $($structure),+> ::ryft_core::ParameterizedFamily<T> for $family<$($structure),+>
        where
            T: ::ryft_core::Parameter,
            $(
                $structure: ::ryft_core::ParameterizedFamily<T>
                    + ::ryft_core::ParameterizedFamily<::ryft_core::Placeholder>,
            )+
        {
            type To = $name<T, $(<$structure as ::ryft_core::ParameterizedFamily<T>>::To),+>;
        }

        impl<P, $($structure),+> ::ryft_core::Parameterized<P> for $name<P, $($structure),+>
        where
            P: ::ryft_core::Parameter,
            $($structure: ::ryft_core::Parameterized<P>,)+
        {
            type Family = $family<$(<$structure as ::ryft_core::Parameterized<P>>::Family),+>;

            type To<T: ::ryft_core::Parameter>
                = <Self::Family as ::ryft_core::ParameterizedFamily<T>>::To
            where
                Self::Family: ::ryft_core::ParameterizedFamily<T>;

            type ParameterStructure = Self::To<::ryft_core::Placeholder>;

            type ParameterIterator<'t, T: 't + ::ryft_core::Parameter>
                = $crate::macros::parameterized_state!(@parameter_iterator_type 't, T; $($field_type),+)
            where
                Self: 't;

            type ParameterIteratorMut<'t, T: 't + ::ryft_core::Parameter>
                = $crate::macros::parameterized_state!(@parameter_iterator_mut_type 't, T; $($field_type),+)
            where
                Self: 't;

            type ParameterIntoIterator<T: ::ryft_core::Parameter>
                = $crate::macros::parameterized_state!(@parameter_into_iterator_type T; $($field_type),+);

            type NamedParameterIterator<'t, T: 't + ::ryft_core::Parameter>
                = $crate::macros::parameterized_state!(@named_parameter_iterator_type 't, T; $($field_type),+)
            where
                Self: 't;

            type NamedParameterIteratorMut<'t, T: 't + ::ryft_core::Parameter>
                = $crate::macros::parameterized_state!(@named_parameter_iterator_mut_type 't, T; $($field_type),+)
            where
                Self: 't;

            type NamedParameterIntoIterator<T: ::ryft_core::Parameter>
                = $crate::macros::parameterized_state!(@named_parameter_into_iterator_type T; $($field_type),+);

            fn parameter_count(&self) -> usize {
                let Self { $($field),+, .. } = self;
                0usize $(+ <$field_type as ::ryft_core::Parameterized<P>>::parameter_count($field))+
            }

            fn parameter_structure(&self) -> Self::ParameterStructure {
                let Self { $($field),+, .. } = self;
                $name {
                    $($field: <$field_type as ::ryft_core::Parameterized<P>>::parameter_structure($field),)+
                    $($marker: ::std::marker::PhantomData,)?
                }
            }

            fn parameters(&self) -> Self::ParameterIterator<'_, P> {
                let Self { $($field),+, .. } = self;
                $crate::macros::parameterized_state!(@iterator parameters; $($field: $field_type),+)
            }

            fn parameters_mut(&mut self) -> Self::ParameterIteratorMut<'_, P> {
                let Self { $($field),+, .. } = self;
                $crate::macros::parameterized_state!(@iterator parameters_mut; $($field: $field_type),+)
            }

            fn into_parameters(self) -> Self::ParameterIntoIterator<P> {
                let Self { $($field),+, .. } = self;
                $crate::macros::parameterized_state!(@iterator into_parameters; $($field: $field_type),+)
            }

            fn named_parameters(&self) -> Self::NamedParameterIterator<'_, P> {
                let Self { $($field),+, .. } = self;
                $crate::macros::parameterized_state!(@named_iterator named_parameters; $($field: $field_type),+)
            }

            fn named_parameters_mut(&mut self) -> Self::NamedParameterIteratorMut<'_, P> {
                let Self { $($field),+, .. } = self;
                $crate::macros::parameterized_state!(@named_iterator named_parameters_mut; $($field: $field_type),+)
            }

            fn into_named_parameters(self) -> Self::NamedParameterIntoIterator<P> {
                let Self { $($field),+, .. } = self;
                $crate::macros::parameterized_state!(@named_iterator into_named_parameters; $($field: $field_type),+)
            }

            fn paths(&self) -> ::std::vec::Vec<::ryft_core::ParameterPath> {
                let Self { $($field),+, .. } = self;
                let mut paths = vec![::ryft_core::ParameterPath::root()];
                $(paths.extend(
                    ::ryft_core::PathPrefixedParameterIterator::new(
                        <$field_type as ::ryft_core::Parameterized<P>>::paths($field)
                            .into_iter()
                            .map(|path| (path, ())),
                        ::ryft_core::ParameterPathSegment::Field(stringify!($field)),
                    )
                    .map(|(path, _)| path),
                );)+
                paths
            }

            fn from_parameters_with_remainder<Remainder: Iterator<Item = P>>(
                structure: Self::ParameterStructure,
                parameters: &mut Remainder,
            ) -> Result<Self, ::ryft_core::ParameterError> {
                let $name { $($field),+, .. } = structure;
                $(let $field = <$field_type as ::ryft_core::Parameterized<P>>::from_parameters_with_remainder(
                    $field,
                    parameters,
                )?;)+
                Ok(Self { $($field,)+ $($marker: ::std::marker::PhantomData,)? })
            }
        }
    };

    // Public form, which forwards to the definition form with the structure parameters as one parenthesized group.
    (
        $(#[$attribute:meta])*
        $visibility:vis struct $name:ident<P, $($structure:ident),+> with family $family:ident
            $(and marker $marker:ident)? {
            $($(#[$field_attribute:meta])* $field:ident: $field_type:ty),+ $(,)?
        }
    ) => {
        $crate::macros::parameterized_state! {
            @define ($($structure,)+)
            $(#[$attribute])*
            $visibility struct $name<P, $($structure),+> with family $family $(and marker $marker)? {
                $($(#[$field_attribute])* $field: $field_type),+
            }
        }
    };

    // Type of the chain of the borrowing leaf iterators of the fields, which ends with an empty iterator.
    (@parameter_iterator_type $lifetime:lifetime, $leaf:ident;) => { ::std::iter::Empty<&$lifetime $leaf> };
    (@parameter_iterator_type $lifetime:lifetime, $leaf:ident; $head:ty $(, $tail:ty)*) => {
        ::std::iter::Chain<
            <$head as ::ryft_core::Parameterized<P>>::ParameterIterator<$lifetime, $leaf>,
            $crate::macros::parameterized_state!(@parameter_iterator_type $lifetime, $leaf; $($tail),*),
        >
    };

    // Type of the chain of the mutably borrowing leaf iterators of the fields, which ends with an empty iterator.
    (@parameter_iterator_mut_type $lifetime:lifetime, $leaf:ident;) => { ::std::iter::Empty<&$lifetime mut $leaf> };
    (@parameter_iterator_mut_type $lifetime:lifetime, $leaf:ident; $head:ty $(, $tail:ty)*) => {
        ::std::iter::Chain<
            <$head as ::ryft_core::Parameterized<P>>::ParameterIteratorMut<$lifetime, $leaf>,
            $crate::macros::parameterized_state!(@parameter_iterator_mut_type $lifetime, $leaf; $($tail),*),
        >
    };

    // Type of the chain of the consuming leaf iterators of the fields, which ends with an empty iterator.
    (@parameter_into_iterator_type $leaf:ident;) => { ::std::iter::Empty<$leaf> };
    (@parameter_into_iterator_type $leaf:ident; $head:ty $(, $tail:ty)*) => {
        ::std::iter::Chain<
            <$head as ::ryft_core::Parameterized<P>>::ParameterIntoIterator<$leaf>,
            $crate::macros::parameterized_state!(@parameter_into_iterator_type $leaf; $($tail),*),
        >
    };

    // Type of the chain of the borrowing named leaf iterators of the fields, whose paths are prefixed with the names of
    // the fields, which ends with an empty iterator.
    (@named_parameter_iterator_type $lifetime:lifetime, $leaf:ident;) => {
        ::std::iter::Empty<(::ryft_core::ParameterPath, &$lifetime $leaf)>
    };
    (@named_parameter_iterator_type $lifetime:lifetime, $leaf:ident; $head:ty $(, $tail:ty)*) => {
        ::std::iter::Chain<
            ::ryft_core::PathPrefixedParameterIterator<
                &$lifetime $leaf,
                <$head as ::ryft_core::Parameterized<P>>::NamedParameterIterator<$lifetime, $leaf>,
            >,
            $crate::macros::parameterized_state!(@named_parameter_iterator_type $lifetime, $leaf; $($tail),*),
        >
    };

    // Type of the chain of the mutably borrowing named leaf iterators of the fields, whose paths are prefixed with the
    // names of the fields, which ends with an empty iterator.
    (@named_parameter_iterator_mut_type $lifetime:lifetime, $leaf:ident;) => {
        ::std::iter::Empty<(::ryft_core::ParameterPath, &$lifetime mut $leaf)>
    };
    (@named_parameter_iterator_mut_type $lifetime:lifetime, $leaf:ident; $head:ty $(, $tail:ty)*) => {
        ::std::iter::Chain<
            ::ryft_core::PathPrefixedParameterIterator<
                &$lifetime mut $leaf,
                <$head as ::ryft_core::Parameterized<P>>::NamedParameterIteratorMut<$lifetime, $leaf>,
            >,
            $crate::macros::parameterized_state!(@named_parameter_iterator_mut_type $lifetime, $leaf; $($tail),*),
        >
    };

    // Type of the chain of the consuming named leaf iterators of the fields, whose paths are prefixed with the names of
    // the fields, which ends with an empty iterator.
    (@named_parameter_into_iterator_type $leaf:ident;) => {
        ::std::iter::Empty<(::ryft_core::ParameterPath, $leaf)>
    };
    (@named_parameter_into_iterator_type $leaf:ident; $head:ty $(, $tail:ty)*) => {
        ::std::iter::Chain<
            ::ryft_core::PathPrefixedParameterIterator<
                $leaf,
                <$head as ::ryft_core::Parameterized<P>>::NamedParameterIntoIterator<$leaf>,
            >,
            $crate::macros::parameterized_state!(@named_parameter_into_iterator_type $leaf; $($tail),*),
        >
    };

    // Chain of the leaf iterators that `$function` returns for the bound fields, in the nesting of the types above.
    (@iterator $function:ident;) => { ::std::iter::empty() };
    (@iterator $function:ident; $head:ident: $head_type:ty $(, $tail:ident: $tail_type:ty)*) => {
        <$head_type as ::ryft_core::Parameterized<P>>::$function($head)
            .chain($crate::macros::parameterized_state!(@iterator $function; $($tail: $tail_type),*))
    };

    // Chain of the named leaf iterators that `$function` returns for the bound fields, whose paths are prefixed with
    // the names of the fields, in the nesting of the types above.
    (@named_iterator $function:ident;) => { ::std::iter::empty() };
    (@named_iterator $function:ident; $head:ident: $head_type:ty $(, $tail:ident: $tail_type:ty)*) => {
        ::ryft_core::PathPrefixedParameterIterator::new(
            <$head_type as ::ryft_core::Parameterized<P>>::$function($head),
            ::ryft_core::ParameterPathSegment::Field(stringify!($head)),
        )
        .chain($crate::macros::parameterized_state!(@named_iterator $function; $($tail: $tail_type),*))
    };
}

pub(crate) use parameterized_state;
