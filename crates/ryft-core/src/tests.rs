//! Contains test support utilities shared by `ryft` unit tests, doctests, and downstream crates, such as the
//! region-carrying [`TestRegionOperation`]. Concrete reference values live in the [`backends`](crate::backends)
//! module instead ([`Scalar`](crate::backends::scalars::Scalar) for the scalar universe and
//! [`Array`](crate::backends::arrays::Array) for the array universe), and the
//! [`check_gradient!`](crate::check_gradient) finite-difference gradient oracle lives in the
//! [`macros`](crate::macros) module.
//!
//! The module is part of `ryft-core`'s public API so downstream tests and documentation examples can use it without
//! feature configuration, but its contents exist only for tests and documentation examples.

use crate::macros::check_count;
use crate::programs::effects::{Effect, Effects};
use crate::programs::operations::Operation;
use crate::programs::regions::RegionInterface;
use crate::programs::types::TypeError;
use crate::types::DataType;

/// Test [`Operation`] with declared attached-region slots, used to exercise the region-carrying construction,
/// inference, validation, effects, rendering, and rebuild paths before any production operation family migrates onto
/// attached regions. Like the rest of this module, it exists only for tests and documentation examples.
#[derive(Clone, Debug, PartialEq)]
pub enum TestRegionOperation {
    /// Region-free binary addition stand-in used inside region bodies.
    Add,

    /// Region-free unary identity stand-in with an observable ordered-IO effect.
    Effectful,

    /// Region-carrying operation declaring the provided region slot names. Its inferred output types are the first
    /// attached region's output types, which pins that region interfaces are derived and delivered during inference.
    WithRegions(&'static [&'static str]),
}

impl Operation<DataType> for TestRegionOperation {
    fn name(&self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Effectful => "effectful",
            Self::WithRegions(_) => "with_regions",
        }
    }

    fn infer_output_types(
        &self,
        input_types: &[DataType],
        region_interfaces: &[RegionInterface<DataType>],
    ) -> Result<Vec<DataType>, TypeError> {
        match self {
            Self::Add => {
                check_count!("input", input_types, 2, TypeError);
                Ok(vec![input_types[0]])
            }
            Self::Effectful => {
                check_count!("input", input_types, 1, TypeError);
                Ok(vec![input_types[0]])
            }
            Self::WithRegions(names) => {
                check_count!("input", input_types, 1, TypeError);
                if region_interfaces.len() != names.len() {
                    return Err(TypeError {
                        message: format!(
                            "expected {} region interfaces but got {}",
                            names.len(),
                            region_interfaces.len(),
                        ),
                    });
                }
                Ok(region_interfaces[0].output_types().to_vec())
            }
        }
    }

    fn region_names(&self) -> &'static [&'static str] {
        match self {
            Self::Add | Self::Effectful => &[],
            Self::WithRegions(names) => names,
        }
    }

    fn effects(&self) -> Effects {
        match self {
            Self::Add | Self::WithRegions(_) => Effects::PURE,
            Self::Effectful => Effects::single(Effect::OrderedIo),
        }
    }
}
