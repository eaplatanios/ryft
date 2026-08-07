use std::fmt::Display;

/// Represents the memory space in which the values described by an [`ArrayType`](crate::arrays::ArrayType) reside.
/// [`Memory`] represents the abstract *placement* information carried by staged types: it names the memory *tier* that
/// holds the underlying data within its owning device's memory hierarchy. Host memories are not a placement target
/// separate from the devices. For example, in the PJRT runtime model every device exposes its own set of addressable
/// memories (i.e., its default device-resident memory plus pinned and unpinned host memories), and a host-placed buffer
/// remains owned by its device (i.e., it is allocated against that device, resides on the physical machine that hosts
/// it, and moves back through that device's copy engines).
///
/// Placement therefore composes with, rather than competes with, any [`Sharding`](crate::arrays::Sharding) carried by
/// the same type. The sharding information determines how the array is partitioned and which device *owns* each shard,
/// while the memory names the tier of the owner's hierarchy that holds the shard's bytes, uniformly for every shard.
/// For a mesh spanning several processes, offloading a sharded array to host memory scatters the shards across the
/// participating machines. Each shard lands in the host memory of its owner device's machine, so the sharding is what
/// decides which machine holds which shard. Host placement parks each shard's bytes off its device between uses
/// without changing which device owns it. Running on CPU *devices* (i.e., a different platform rather than a different
/// tier) is unrelated to [`Memory`] spaces and flows through device meshes and shardings instead.
///
/// Note that *placement* information is metadata about *where* values live, not about their contents. It never affects
/// shapes, data types, or the numerical semantics of operations. Values residing in different memory spaces never
/// combine directly (i.e., moving a value between memories requires staging an explicit transfer), and eager domains,
/// which have no memory hierarchy, treat all placements alike.
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
