//! CUDA version encoding, validation, and presentation.

use std::fmt::{Display, Formatter};

use crate::{Error, ffi};

/// CUDA version required by a [`CudaKernelLauncher`](crate::CudaKernelLauncher).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CudaVersion {
    /// CUDA's integer encoding: `1000 * major + 10 * minor`.
    encoded: i32,
}

impl CudaVersion {
    /// Creates a CUDA version from CUDA's integer encoding: `1000 * major + 10 * minor`.
    pub fn from_encoded(encoded: u32) -> Result<Self, Error> {
        let encoded = i32::try_from(encoded).map_err(|_| {
            Error::invalid_argument("the encoded cuda version cannot be represented by the CUDA Driver API")
        })?;
        if encoded < ffi::ENTRY_POINT_ABI_VERSION || encoded.rem_euclid(10) != 0 {
            return Err(Error::invalid_argument(format!(
                "invalid encoded cuda version `{encoded}`; expected CUDA 12.0 or newer",
            )));
        }
        Ok(Self { encoded })
    }

    /// Returns the CUDA major version.
    pub fn major(self) -> u32 {
        (self.encoded / 1000) as u32
    }

    /// Returns the CUDA minor version.
    pub fn minor(self) -> u32 {
        (self.encoded.rem_euclid(1000) / 10) as u32
    }

    /// Returns CUDA's integer version encoding: `1000 * major + 10 * minor`.
    pub fn encoded(self) -> u32 {
        self.encoded as u32
    }
}

impl Display for CudaVersion {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}.{}", self.major(), self.minor())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_cuda_version_from_encoded() {
        assert_eq!(CudaVersion::from_encoded(12_000), Ok(CudaVersion { encoded: 12_000 }));
        assert_eq!(CudaVersion::from_encoded(13_000), Ok(CudaVersion { encoded: 13_000 }));
    }

    #[test]
    fn test_cuda_version_from_encoded_invalid() {
        assert!(matches!(
            CudaVersion::from_encoded(11_080),
            Err(Error::InvalidArgument { message, .. })
                if message == "invalid encoded cuda version `11080`; expected CUDA 12.0 or newer",
        ));
        assert!(matches!(
            CudaVersion::from_encoded(12_091),
            Err(Error::InvalidArgument { message, .. })
                if message == "invalid encoded cuda version `12091`; expected CUDA 12.0 or newer",
        ));
        assert!(matches!(
            CudaVersion::from_encoded(u32::MAX),
            Err(Error::InvalidArgument { message, .. })
                if message == "the encoded cuda version cannot be represented by the CUDA Driver API",
        ));
    }

    #[test]
    fn test_cuda_version_major() {
        assert_eq!(CudaVersion::from_encoded(12_090).unwrap().major(), 12);
        assert_eq!(CudaVersion::from_encoded(13_000).unwrap().major(), 13);
    }

    #[test]
    fn test_cuda_version_minor() {
        assert_eq!(CudaVersion::from_encoded(12_090).unwrap().minor(), 9);
        assert_eq!(CudaVersion::from_encoded(13_000).unwrap().minor(), 0);
    }

    #[test]
    fn test_cuda_version_encoded() {
        assert_eq!(CudaVersion::from_encoded(12_090).unwrap().encoded(), 12_090);
    }

    #[test]
    fn test_cuda_version_display_and_debug() {
        let version = CudaVersion::from_encoded(12_090).unwrap();
        assert_eq!(version.to_string(), "12.9");
        assert_eq!(format!("{version:?}"), "CudaVersion { encoded: 12090 }");
    }

    #[test]
    fn test_cuda_version_equality_ordering_and_hash() {
        let version = CudaVersion::from_encoded(12_090).unwrap();
        let newer = CudaVersion::from_encoded(13_000).unwrap();
        assert_eq!(version, version);
        assert_ne!(version, newer);
        assert!(version < newer);
        let versions = HashMap::from([(version, "cuda 12.9")]);
        assert_eq!(versions.get(&CudaVersion::from_encoded(12_090).unwrap()), Some(&"cuda 12.9"));
        assert_eq!(versions.get(&newer), None);
    }
}
