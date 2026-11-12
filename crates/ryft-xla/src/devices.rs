use ryft_core::Device;
use ryft_pjrt::Device as PjrtDevice;

use crate::FromPjrt;

impl FromPjrt<PjrtDevice<'_>> for Device {
    type Output = Result<Self, ryft_pjrt::Error>;

    #[inline]
    fn from_pjrt(value: PjrtDevice<'_>) -> Self::Output {
        Self::from_pjrt(&value)
    }
}

impl FromPjrt<&PjrtDevice<'_>> for Device {
    type Output = Result<Self, ryft_pjrt::Error>;

    #[inline]
    fn from_pjrt(value: &PjrtDevice<'_>) -> Self::Output {
        Ok(Self::new(value.id()?, value.process_index()?))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::tests::execution_client;

    use super::*;

    #[test]
    fn test_device_from_pjrt() {
        let client = execution_client();
        let devices = client.addressable_devices().unwrap();
        let expected = Device::new(devices[0].id().unwrap(), devices[0].process_index().unwrap());
        assert_eq!(Device::from_pjrt(&devices[0]), Ok(expected));
        assert_eq!(Device::from_pjrt(devices[0].clone()), Ok(expected));
    }
}
