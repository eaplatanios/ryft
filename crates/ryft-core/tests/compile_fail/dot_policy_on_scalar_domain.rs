//! `DotsSaveable` requires the operation family to contain `DotOperation`, which `ScalarOperation` does not, so
//! configuring it for a scalar domain must fail to compile at `with_policy`.

use ryft_core::backends::scalars::{Scalar, ScalarOperation};
use ryft_core::contexts::EagerContext;
use ryft_core::tracing::DomainTracer;
use ryft_core::tracing_v2::{DotsSaveable, rematerialize};

fn main() {
    let _ = rematerialize::<EagerContext<Scalar, ScalarOperation<Scalar>>, _, _, _>(
        |x: DomainTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>| Ok(x.clone() * x),
    )
    .with_policy(DotsSaveable);
}
