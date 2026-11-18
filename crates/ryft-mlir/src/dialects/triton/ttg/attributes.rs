use std::ops::BitOr;

/// Memory accesses ordered by a TritonGPU barrier. Combine flags to order more than one address space.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BarrierAddressSpaces(u32);

impl BarrierAddressSpaces {
    /// Orders no memory accesses and synchronizes control flow only.
    pub const NONE: Self = Self(0);

    /// Orders shared-memory accesses.
    pub const LOCAL: Self = Self(1);

    /// Orders reads of global memory.
    pub const GLOBAL_READ: Self = Self(2);

    /// Orders writes of global memory.
    pub const GLOBAL_WRITE: Self = Self(4);

    /// Orders reads of tensor memory.
    pub const TENSOR_READ: Self = Self(8);

    /// Orders writes of tensor memory.
    pub const TENSOR_WRITE: Self = Self(16);

    /// Orders all memory accesses supported by the barrier.
    pub const ALL: Self = Self(31);

    /// Returns the bitmask represented by these address spaces.
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Constructs an address-space mask if every bit belongs to a supported address space.
    pub const fn from_bits(bits: u32) -> Option<Self> {
        if bits & !Self::ALL.0 == 0 { Some(Self(bits)) } else { None }
    }
}

impl BitOr for BarrierAddressSpaces {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

/// Threads that participate in a TritonGPU barrier.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum BarrierScope {
    /// All threads of the workgroup (i.e., the CTA).
    Cta = 0,

    /// All lanes of each participating warp, with only shared-memory ordering.
    Warp = 1,
}
