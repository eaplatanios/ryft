//! XLA compilation options and specialization policies.

use std::hash::Hash;
use std::path::Path;
use std::sync::Arc;

use ryft_core::{DeviceMesh, Parameterized, Sharding};
use sha2::{Digest, Sha256};

use crate::{Error, RaggedDotLoweringStrategy};

// TODO(eaplatanios): Review this module.

/// Opaque XLA feedback-directed optimization profile used for profile-guided latency estimation.
///
/// XLA defines the binary payload consumed by `ExecutableCompilationOptions::fdo_profile`; Ryft intentionally does
/// not reinterpret it. Profiles are compiler- and program-specific. Supplying one changes exact, persistent, and
/// distributed compilation identity because its bytes are encoded in the canonical compilation options.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FeedbackDirectedProfile {
    bytes: Arc<[u8]>,
    version: i64,
}

impl FeedbackDirectedProfile {
    /// Creates a profile with XLA profile version zero.
    #[inline]
    pub fn new(bytes: impl Into<Vec<u8>>) -> Self {
        Self { bytes: bytes.into().into(), version: 0 }
    }

    /// Loads opaque profile bytes from `path`.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Error> {
        Ok(Self::new(std::fs::read(path)?))
    }

    /// Sets the XLA compilation profile version associated with these bytes.
    #[inline]
    pub fn with_version(mut self, version: i64) -> Self {
        self.version = version;
        self
    }

    /// Returns the opaque profile bytes passed to XLA.
    #[inline]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns the XLA compilation profile version.
    #[inline]
    pub fn version(&self) -> i64 {
        self.version
    }

    /// Returns a stable SHA-256 digest suitable for diagnostics and provenance.
    #[inline]
    pub fn digest(&self) -> [u8; 32] {
        Sha256::digest(self.bytes()).into()
    }

    /// Writes the opaque profile bytes to `path`.
    #[inline]
    pub fn write_to_file(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        std::fs::write(path, self.bytes())?;
        Ok(())
    }
}

/// Host-side policy for grouping concrete input extents into bounded retained-JIT specializations.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum InputBoundBucketing {
    /// Rounds each host-observed positive axis extent up to the next power of two. The resulting dynamic type admits
    /// extents through that bucket, bounding padding overhead below two times the logical extent while retaining only
    /// logarithmically many compiled specializations.
    PowerOfTwo,
}

/// Backend-specific per-call options for the [`CompilationDomain`](ryft_core::CompilationDomain) implementation on
/// [`Domain`](crate::Domain). Carries the device mesh, optional sharding overrides for inputs/outputs, optional
/// per-input donation flags, backend lowering choices, and retained-dispatch policies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Options {
    /// Explicit compiler and typed embedding used to select retained portable kernel calls.
    pub kernel_compiler: Option<crate::kernels::KernelCompilerBinding>,

    /// Concrete device mesh the compiled program runs against.
    pub mesh: DeviceMesh,

    /// Optional override for per-input shardings.
    pub input_shardings: Option<Vec<Sharding>>,

    /// Optional override for per-output shardings.
    pub output_shardings: Option<Vec<Sharding>>,

    /// Optional flat per-input donation flags. `None` donates nothing; a present vector must match the program's
    /// flat public input arity exactly, which lowering validates.
    pub donation_flags: Option<Vec<bool>>,

    /// Optional opaque feedback-directed optimization profile forwarded to XLA.
    pub feedback_directed_profile: Option<FeedbackDirectedProfile>,

    /// Optional host-side bound bucketing policy for retained-JIT input signatures.
    pub input_bound_bucketing: Option<InputBoundBucketing>,

    /// Backend representation used to lower ragged dots. Defaults to [`RaggedDotLoweringStrategy::Decomposition`].
    pub ragged_dot_lowering_strategy: RaggedDotLoweringStrategy,
}

impl Options {
    /// Constructs options for `mesh` with no sharding overrides or donation and with portable grouped-dot
    /// decomposition.
    #[inline]
    pub fn new(mesh: DeviceMesh) -> Self {
        Self {
            kernel_compiler: None,
            mesh,
            input_shardings: None,
            output_shardings: None,
            donation_flags: None,
            feedback_directed_profile: None,
            input_bound_bucketing: None,
            ragged_dot_lowering_strategy: RaggedDotLoweringStrategy::default(),
        }
    }

    /// Selects one immutable compiler and typed embedding for portable kernel calls in this compilation.
    pub fn with_kernel_compiler(mut self, compiler: crate::kernels::KernelCompilerBinding) -> Self {
        self.kernel_compiler = Some(compiler);
        self
    }

    /// Sets the per-input sharding overrides that the SPMD partitioner reads at lowering time.
    /// Length must equal the flat input arity once the program is traced; mismatches surface as
    /// [`CompilationError::ShardingCountMismatch`](crate::CompilationError::ShardingCountMismatch) at compile time.
    #[inline]
    pub fn with_input_shardings(mut self, input_shardings: Vec<Sharding>) -> Self {
        self.input_shardings = Some(input_shardings);
        self
    }

    /// Sets the per-output sharding overrides for SPMD partitioning. Length must equal the flat
    /// output arity once the program is traced.
    #[inline]
    pub fn with_output_shardings(mut self, output_shardings: Vec<Sharding>) -> Self {
        self.output_shardings = Some(output_shardings);
        self
    }

    /// Sets the per-input donation flags from a [`Parameterized`] tree of `bool`s whose leaf
    /// shape matches the function's flat input layout.
    ///
    /// Each `true` leaf marks the corresponding input as donatable: the executor may reuse its
    /// device buffer for an output, leaving the caller's runtime value in a donated state after
    /// the call returns. The tree is flattened into [`Self::donation_flags`]; the resulting
    /// vector length is validated against the function's flat public input arity at lowering time.
    ///
    /// Typical leaf-shaped inputs lower to a single `bool` (e.g. `with_donate(true)` for a
    /// single-argument closure), while nested tuple / struct inputs accept the matching nested
    /// tuple / struct of `bool`s.
    #[inline]
    pub fn with_donate<P: Parameterized<bool>>(mut self, donate: P) -> Self {
        self.donation_flags = Some(donate.into_parameters().collect());
        self
    }

    /// Supplies an opaque feedback-directed optimization profile for manual profile-guided recompilation.
    ///
    /// The profile must have been produced for compatible StableHLO, topology, and compiler settings. XLA performs
    /// backend-specific validation according to its debug options; Ryft includes the exact bytes in every cache key.
    #[inline]
    pub fn with_feedback_directed_profile(mut self, profile: FeedbackDirectedProfile) -> Self {
        self.feedback_directed_profile = Some(profile);
        self
    }

    /// Enables host-side retained-JIT input-bound bucketing.
    ///
    /// Static runtime array axes become bounded-dynamic axes in the staged signature, with bounds selected by
    /// `bucketing`. The observed extent remains entirely host-side, participates in dispatch identity only through its
    /// bucket, and is transported through the existing bounded-input ABI. Data-derived extents created inside the
    /// program are deliberately unaffected because observing them here would require a device-to-host synchronization
    /// and a split compilation boundary. Bucketing currently accepts only array inputs on a single-device mesh:
    /// bucketed signatures are dynamic and therefore require disabling the Shardy partitioner, which Ryft does not do
    /// implicitly for a multi-device request.
    #[inline]
    pub fn with_input_bound_bucketing(mut self, bucketing: InputBoundBucketing) -> Self {
        self.input_bound_bucketing = Some(bucketing);
        self
    }

    /// Selects the backend representation used to lower ragged dots.
    #[inline]
    pub fn with_ragged_dot_lowering_strategy(mut self, strategy: RaggedDotLoweringStrategy) -> Self {
        self.ragged_dot_lowering_strategy = strategy;
        self
    }
}

impl std::hash::Hash for Options {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // `DeviceMesh` does not derive `Hash`; hash its (logical mesh, device order) pair
        // manually instead. Everything else here derives `Hash` already.
        self.mesh.logical_mesh().hash(state);
        self.mesh.devices().hash(state);
        self.input_shardings.hash(state);
        self.output_shardings.hash(state);
        self.donation_flags.hash(state);
        self.feedback_directed_profile.hash(state);
        self.input_bound_bucketing.hash(state);
        self.ragged_dot_lowering_strategy.hash(state);
        self.kernel_compiler.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_pjrt::protos::CompilationOptions;

    use crate::domains::compilation::jit_compilation_options;

    use super::*;

    #[test]
    fn test_feedback_directed_profile_round_trips_and_changes_compilation_options() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("profile.pb");
        let profile = FeedbackDirectedProfile::new(vec![1, 2, 3, 4]).with_version(7);
        profile.write_to_file(&path).unwrap();
        let restored = FeedbackDirectedProfile::from_file(&path).unwrap().with_version(7);
        assert_eq!(restored, profile);
        assert_eq!(restored.digest(), profile.digest());

        let baseline = jit_compilation_options(&CompilationOptions::default(), 2, true, None);
        let profiled = jit_compilation_options(&CompilationOptions::default(), 2, true, Some(&profile));
        assert!(baseline.executable_build_options.unwrap().fdo_profile.is_empty());
        assert_eq!(profiled.executable_build_options.unwrap().fdo_profile, vec![1, 2, 3, 4]);
        assert_eq!(profiled.profile_version, 7);
    }
}
