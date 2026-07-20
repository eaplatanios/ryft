use std::fmt::{Debug, Display};

use crate::contexts::Domain;
use crate::parameters::Parameter;
use crate::programs::atoms::AtomId;
use crate::programs::regions::RegionId;
use crate::programs::types::Typed;

/// Location of one Single Static Assignment (SSA) value in a multi-region [`Program`](crate::Program), identified by
/// its containing [`Region`](crate::Region) and its region-local [`AtomId`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueId {
    /// [`Region`](crate::Region) containing the atom.
    region: RegionId,

    /// Region-local [`AtomId`] of the value.
    atom: AtomId,
}

impl ValueId {
    /// Creates a new [`ValueId`] from the provided containing region and region-local atom identifier.
    #[inline]
    pub fn new(region: RegionId, atom: AtomId) -> Self {
        Self { region, atom }
    }

    /// Returns the [`RegionId`] of the [`Region`](crate::Region) containing the atom.
    #[inline]
    pub fn region(self) -> RegionId {
        self.region
    }

    /// Returns the region-local [`AtomId`] of the value.
    #[inline]
    pub fn atom(self) -> AtomId {
        self.atom
    }
}

/// Represents leaf values that can participate in traced [`Program`](crate::Program)s. [`Value`] is implemented by
/// every type that can appear as a leaf in a staged [`Program`](crate::Program): both concrete data types such as
/// `f32`, `f64`, and backend arrays, and tracing wrappers such as [`Tracer`](crate::Tracer). It inherits its associated
/// [`Type`](crate::Type) from [`Typed`], so generic code recovers the type as `V::Type` and pinning sites write
/// `V: Value<Type = ArrayType>`. It additionally requires [`Debug`] and [`Display`] so that diagnostics, constants,
/// and [`Operation`](crate::Operation) metadata can render their carried values directly.
pub trait Value: Clone + Debug + Display + Parameter + Typed + Sized {
    /// [`Domain`] that operations involving this [`Value`] *dispatch* through. Every value names two domains:
    /// capability function calls dispatch through the [`DispatchDomain`](Self::DispatchDomain), while transform work
    /// executes in the [`ExecutionDomain`](Self::ExecutionDomain). The two domains coincide for every transform and
    /// staged value (e.g., a staged [`Tracer`](crate::Tracer)'s trace, a [`BatchingTracer`](crate::BatchingTracer)'s
    /// batching level, etc.): dispatch and execution both happen in the live context such a value flows through.
    /// However, they become separate for concrete backend values (e.g., concrete arrays). In those cases, the
    /// [`DispatchDomain`](Self::DispatchDomain) is the constant-only [`EagerContext`](crate::EagerContext) such that
    /// capability calls dispatch to direct implementations instead of a context, while the
    /// [`ExecutionDomain`](Self::ExecutionDomain) names the backend's *rich*, operation-executing eager domain. Backend
    /// values whose rich domain requires state or defaults that cannot be derived from a value (e.g., a client handle)
    /// keep the constant-only domain here too, which simply means free transform entry points do not serve them and an
    /// explicit context must be used instead.
    ///
    /// Blanket capability implementations (e.g., the value-level arithmetic sugar) bind through this domain and use its
    /// operation universe as their coherence discriminator: the sugar applies when `V::DispatchDomain::Operation` can
    /// accept the operation being bound. A staged [`Tracer`](crate::Tracer)'s dispatch domain is its live trace, so the
    /// sugar records instructions there. A concrete backend value's dispatch domain is the constant-only
    /// [`EagerContext`](crate::EagerContext), whose [`ConstantOperation`](crate::ConstantOperation) universe accepts
    /// nothing. This is precisely what keeps the blanket implementations coherent with (i.e., disjoint from) the direct
    /// capability implementations that concrete values provide instead.
    type DispatchDomain: Domain<Type = Self::Type, Value = Self>;

    /// [`Domain`] that transform work involving this [`Value`] *executes* in. Refer to the documentation of
    /// [`DispatchDomain`](Self::DispatchDomain) for information on the two types of [`Domain`]s that each value
    /// provides.
    type ExecutionDomain: Domain<Type = Self::Type, Value = Self>;

    /// Returns the [`Domain`] that operations involving this [`Value`] *dispatch* through. Refer to the
    /// documentation of [`DispatchDomain`](Self::DispatchDomain) for more information.
    fn dispatch_domain(&self) -> Self::DispatchDomain;

    /// Returns the [`Domain`] that transform work involving this [`Value`] *executes* in. Refer to the
    /// documentation of [`ExecutionDomain`](Self::ExecutionDomain) for more information.
    fn execution_domain(&self) -> Self::ExecutionDomain;
}
