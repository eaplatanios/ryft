// When to use `#[inline]`?
//
// #[inline] is very different than simply just an inline hint. As I mentioned before, there's no equivalent in C++ for what #[inline] does. In debug mode rustc basically ignores #[inline], pretending you didn't even write it. In release mode the compiler will, by default, codegen an #[inline] function into every single referencing codegen unit, and then it will also add inlinehint. This means that if you have 16 CGUs and they all reference an item, every single one is getting the entire item's implementation inlined into it.
//
// You can add #[inline]:
//   - To public, small, non-generic functions.
// You shouldn't need #[inline]:
//   - On methods that have any generics in scope.
//   - On methods on traits that don't have a default implementation.
// #[inline] can always be introduced later, so if you're in doubt they can just be removed.
//
// What about `#[inline(always)]`?
//
// You should just about never need #[inline(always)]. It may be beneficial for private helper methods that are used
// in a limited number of places or for trivial operators. A micro benchmark should justify the attribute.

#![feature(min_specialization)]

impl<T: Clone> From<&[T]> for Rc<[T]> {
    #[inline]
    fn from(v: &[T]) -> Rc<[T]> {
        <Self as RcFromSlice<T>>::from_slice(v)
    }
}

/// Specialization trait used for `From<&[T]>`.
trait RcFromSlice<T> {
    fn from_slice(slice: &[T]) -> Self;
}

impl<T: Clone> RcFromSlice<T> for Rc<[T]> {
    #[inline]
    default fn from_slice(v: &[T]) -> Self {
        unsafe { Self::from_iter_exact(v.iter().cloned(), v.len()) }
    }
}

impl<T: Copy> RcFromSlice<T> for Rc<[T]> {
    #[inline]
    fn from_slice(v: &[T]) -> Self {
        unsafe { Self::copy_from_slice(v) }
    }
}

// rustc_specialization_trait restricts the implementations of a trait to be "always applicable". Implementing traits annotated with rustc_specialization_trait is unstable, so this should not be used on any stable traits exported from the standard library. Sized is an exception, and can have this attribute because it already cannot be implemented by an impl block. Note: rustc_specialization_trait only prevents incorrect monomorphizations, it does not prevent a type from being coerced between specialized and unspecialized types which can be important when specialization must be applied consistently. See rust-lang/rust#85863 for more details.

// rustc_unsafe_specialization_marker allows specializing on a trait with no associated items. The attribute is unsafe because lifetime constraints from the implementations of the trait are not considered when specializing. The following example demonstrates a limitation of rustc_unsafe_specialization_marker, the specialized implementation is used for all shared reference types, not just those with 'static lifetime. Because of this, new uses of rustc_unsafe_specialization_marker should be avoided.
//
// #[rustc_unsafe_specialization_marker]
// trait StaticRef {}
//
// impl<T> StaticRef for &'static T {}
//
// trait DoThing: Sized {
//     fn do_thing(self);
// }
//
// impl<T> DoThing for T {
//     default fn do_thing(self) {
//         // slow impl
//     }
// }
//
// impl<T: StaticRef> DoThing for T {
//     fn do_thing(self) {
//         // fast impl
//     }
// }
//
// rustc_unsafe_specialization_marker exists to allow existing specializations that are based on marker traits exported from std, such as Copy, FusedIterator or Eq.
