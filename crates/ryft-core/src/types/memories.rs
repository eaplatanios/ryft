use std::fmt::Display;

/// Represents the memory space in which the values described by an [`ArrayType`](crate::ArrayType) reside. [`Memory`]
/// represents the abstract *placement* information carried by staged types: it names the memory *tier* holding the
/// underlying data, while any [`Sharding`](crate::Sharding) on the same type names the devices holding the shards.
/// The two are orthogonal, and for sharded arrays the memory applies uniformly to every shard. Each shard resides
/// in its own device's memory of this kind. Note that *placement* information is metadata about *where* values live,
/// not about their contents: it never affects shapes, data types, or the numerical semantics of operations. Values
/// residing in different memory spaces never combine directly (i.e., moving a value between memories requires staging
/// an explicit transfer), and eager domains, which have no memory hierarchy, treat all placements alike.
///
/// The [`Display`] implementation renders `Device`, `Host[Pinned]`, or `Host[Unpinned]`. This rendering is only
/// meant for diagnostics and rendered types/programs; backends that need their own placement vocabulary (e.g., the
/// XLA device placement annotation values or PJRT memory-kind names) own explicit conversions in their lowering
/// code instead of relying on this rendering.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Memory {
    /// Default device-resident memory (e.g., HBM on a GPU or a TPU).
    #[default]
    Device,

    /// Host (i.e., CPU) memory.
    Host {
        /// Boolean value indicating whether the memory is pinned (i.e., page-locked). Pinned host memory supports
        /// the asynchronous device-to-host and host-to-device transfers that XLA's host-offloading pipeline relies
        /// on (which may also be true for similar functionality in other backends). On the other hand, unpinned host
        /// memory avoids consuming page-locked pages at the cost of slower, synchronous, staging.
        pinned: bool,
    },
}

impl Display for Memory {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Device => formatter.write_str("Device"),
            Self::Host { pinned: true } => formatter.write_str("Host[Pinned]"),
            Self::Host { pinned: false } => formatter.write_str("Host[Unpinned]"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_memory() {
        // The default placement is device-resident memory.
        assert_eq!(Memory::default(), Memory::Device);

        // `Display` renders the diagnostic placement names.
        assert_eq!(Memory::Device.to_string(), "Device");
        assert_eq!(Memory::Host { pinned: true }.to_string(), "Host[Pinned]");
        assert_eq!(Memory::Host { pinned: false }.to_string(), "Host[Unpinned]");

        // Equality distinguishes the device/host split and the pinned flag.
        assert_eq!(Memory::Host { pinned: true }, Memory::Host { pinned: true });
        assert_ne!(Memory::Device, Memory::Host { pinned: true });
        assert_ne!(Memory::Host { pinned: true }, Memory::Host { pinned: false });

        // `Memory` is usable as a hash-map key.
        let map = HashMap::from([(Memory::Device, 0), (Memory::Host { pinned: true }, 1)]);
        assert_eq!(map[&Memory::Device], 0);
        assert_eq!(map[&Memory::Host { pinned: true }], 1);
    }
}
