//! JAX-style orchestration helpers on top of PJRT extensions.
//!
//! This module mirrors the registration pattern used by JAX:
//! - register platform-scoped custom call/type handlers,
//! - allow registrations to happen before handlers are installed,
//! - flush queued registrations when handlers become available.
//!
//! [`ffi`] maps JAX custom call and custom type registration onto PJRT FFI and
//! legacy GPU custom call extensions.
//! [`triton`] maps JAX Triton compilation handlers onto the PJRT Triton extension.

use std::borrow::Cow;

use crate::{Client, Error};

/// JAX-style custom call and custom type registration orchestration.
pub mod ffi;
/// JAX-style Triton compilation-handler orchestration.
pub mod triton;

#[allow(dead_code)]
pub(crate) mod gpu_runtime;

/// Canonical platform identifier used by [`crate::jax`] helpers to emulate JAX-style
/// per-platform registration behavior.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum JaxPlatform {
    Cpu,
    Cuda,
    Rocm,
    Tpu,
    Neuron,
    Metal,
    Other(String),
}

impl JaxPlatform {
    /// Returns the canonical uppercase registration key used by handler maps.
    pub fn canonical_handler_name(&self) -> Cow<'_, str> {
        match self {
            Self::Cpu => Cow::Borrowed("CPU"),
            Self::Cuda => Cow::Borrowed("CUDA"),
            Self::Rocm => Cow::Borrowed("ROCM"),
            Self::Tpu => Cow::Borrowed("TPU"),
            Self::Neuron => Cow::Borrowed("NEURON"),
            Self::Metal => Cow::Borrowed("METAL"),
            Self::Other(name) => Cow::Owned(name.to_ascii_uppercase()),
        }
    }

    /// Returns platform names to try when invoking backend APIs.
    pub(crate) fn registration_name_candidates(&self) -> Vec<String> {
        let canonical = self.canonical_handler_name().into_owned();
        let lowercase = canonical.to_ascii_lowercase();
        if canonical == lowercase { vec![canonical] } else { vec![canonical, lowercase] }
    }

    /// Returns `true` iff this platform is a GPU backend.
    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::Cuda | Self::Rocm)
    }

    /// Constructs a canonical platform enum from a runtime platform string.
    pub fn from_platform_name(name: &str) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "cpu" => Self::Cpu,
            "cuda" | "gpu" => Self::Cuda,
            "rocm" => Self::Rocm,
            "tpu" => Self::Tpu,
            "neuron" => Self::Neuron,
            "metal" => Self::Metal,
            _ => Self::Other(name.trim().to_ascii_uppercase()),
        }
    }

    /// Reads and normalizes the platform from the provided PJRT [`Client`].
    pub fn from_client(client: &Client<'_>) -> Result<Self, Error> {
        client.platform_name().map(|name| Self::from_platform_name(name.as_ref()))
    }
}

impl From<&str> for JaxPlatform {
    fn from(value: &str) -> Self {
        Self::from_platform_name(value)
    }
}

impl From<String> for JaxPlatform {
    fn from(value: String) -> Self {
        Self::from_platform_name(value.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jax_platform_normalization() {
        assert_eq!(JaxPlatform::from("cpu"), JaxPlatform::Cpu);
        assert_eq!(JaxPlatform::from("CUDA"), JaxPlatform::Cuda);
        assert_eq!(JaxPlatform::from("rocm"), JaxPlatform::Rocm);
        assert_eq!(JaxPlatform::from("tpu"), JaxPlatform::Tpu);
        assert_eq!(JaxPlatform::from("NEURON"), JaxPlatform::Neuron);
        assert_eq!(JaxPlatform::from("METAL"), JaxPlatform::Metal);
        assert_eq!(JaxPlatform::from("xpu"), JaxPlatform::Other("XPU".to_string()));
    }

    #[test]
    fn test_jax_platform_handler_names() {
        assert_eq!(JaxPlatform::Cpu.canonical_handler_name(), "CPU");
        assert_eq!(JaxPlatform::Cuda.canonical_handler_name(), "CUDA");
        assert_eq!(JaxPlatform::Rocm.canonical_handler_name(), "ROCM");
        assert_eq!(JaxPlatform::Tpu.canonical_handler_name(), "TPU");
        assert_eq!(JaxPlatform::Neuron.canonical_handler_name(), "NEURON");
        assert_eq!(JaxPlatform::Metal.canonical_handler_name(), "METAL");
        assert_eq!(JaxPlatform::Other("foo".to_string()).canonical_handler_name(), "FOO");
    }
}
