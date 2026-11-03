//! Checks that downstream crates can implement value capabilities directly through the public API. Whether a downstream
//! capability implementation is coherent with the blanket implementations of `ryft-core` depends on which crate defines
//! the value, because the coherence rules reason differently about local and foreign types. A unit test inside
//! `ryft-core` cannot observe this, so this test lives in its own crate.

use std::borrow::Cow;
use std::fmt::{Display, Formatter};

use pretty_assertions::assert_eq;

use ryft_core::{
    Accuracy, Array, ArrayType, Context, EagerContext, LogicalMesh, ManualVariationAlignment, ParallelVary, Parameter,
    ProgramError, Sin, SinOperation, Typed, Value, ValueDirectDispatch,
};

#[test]
fn test_value_direct_dispatch() {
    /// Value that implements capabilities directly even though its domain can bind the operations that the
    /// domain-binding blanket capability implementations would bind. Only the [`ValueDirectDispatch`] marker
    /// keeps these direct implementations disjoint from those blanket implementations.
    #[derive(Clone, Debug, PartialEq)]
    struct DirectArray {
        /// Host array that this value wraps.
        array: Array,
    }

    impl Display for DirectArray {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
            Display::fmt(&self.array, formatter)
        }
    }

    impl Parameter for DirectArray {}

    impl Typed for DirectArray {
        type Type = ArrayType;

        fn r#type(&self) -> Cow<'_, ArrayType> {
            self.array.r#type()
        }
    }

    /// Operation family of the [`DirectArray`] domain. It rejects guards, which eager replay never requests.
    #[derive(Clone, Debug, ryft_macros::Operation)]
    #[ryft(crate = "ryft_core", type(ArrayType), constant(DirectArray))]
    enum DirectOperation {
        Sin(SinOperation<ArrayType>),
    }

    impl Value for DirectArray {
        type Dispatch = ValueDirectDispatch;
        type Domain = EagerContext<Self, DirectOperation>;

        fn domain(&self) -> Self::Domain {
            EagerContext::new()
        }
    }

    // The domain's operation family converts from `SinOperation`, so the blanket `Sin` implementation would also apply
    // to this value (i.e., a conflicting implementation, E0119) if the value did not use `ValueDirectDispatch`.
    impl Sin for DirectArray {
        fn sin_with_accuracy(&self, accuracy: Accuracy) -> Result<Self, ProgramError> {
            Ok(Self { array: self.array.sin_with_accuracy(accuracy)? })
        }
    }

    // Composition blanket implementations, such as `ManualVariationAlignment`, still apply to direct values.
    // This one requires a direct `ParallelVary` implementation, because the domain-binding `ParallelVary` blanket
    // does not apply.
    impl ParallelVary for DirectArray {
        fn parallel_vary(&self, axis_name: &str) -> Result<Self, ProgramError> {
            Err(ProgramError::UnsupportedOperation {
                message: format!("`parallel_vary` requires an active manual mesh axis `{axis_name}`"),
            })
        }

        fn parallel_vary_on_mesh(&self, axis_name: &str, _mesh: &LogicalMesh) -> Result<Self, ProgramError> {
            self.parallel_vary(axis_name)
        }
    }

    let value = DirectArray { array: Array::scalar(0.5f64).unwrap() };
    let expected = DirectArray { array: Array::scalar(0.5f64.sin()).unwrap() };

    // Capability calls use the direct implementation.
    assert_eq!(value.sin(), Ok(expected.clone()));

    // The domain binds the same operation, and its interpretation bottoms out in the direct implementation.
    assert_eq!(
        value.domain().bind(SinOperation::<ArrayType>::new(), Vec::new(), std::slice::from_ref(&value)),
        Ok(vec![expected]),
    );

    // Composition blanket implementations still apply, and values without manual variation are already aligned.
    let other = DirectArray { array: Array::scalar(2.0f64).unwrap() };
    assert_eq!(
        ManualVariationAlignment::align_manual_variation(&[value.clone(), other.clone()]),
        Ok(vec![value, other]),
    );
}
