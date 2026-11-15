//! Compilation options of the XLA [`Domain`](crate::Domain).
//!
//! [`Options`] configures one compilation. It carries the device mesh, sharding overrides, donation flags, backend
//! lowering choices, and policies of retained dispatch, and it is retained by staged functions for their later lowering
//! and compilation. [`FeedbackDirectedProfile`] holds an opaque XLA profile for profile-guided recompilation, and
//! [`InputBoundBucketing`] groups concrete input extents into bounded specializations of retained JIT dispatchers.

use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Arc;

use sha2::{Digest, Sha256};

use ryft_core::{DeviceMesh, Parameterized, Sharding};

use crate::kernels::KernelCompilerBinding;
use crate::{Error, RaggedDotLoweringStrategy};

// TODO(eaplatanios): Review this module.

/// Opaque XLA feedback-directed optimization profile used for profile-guided latency estimation.
///
/// XLA defines the binary payload consumed by `ExecutableCompilationOptions::fdo_profile`, and Ryft intentionally does
/// not reinterpret it. Profiles are specific to the compiler and the program. Supplying one changes exact, persistent,
/// and distributed compilation identity because its bytes are encoded in the canonical compilation options.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FeedbackDirectedProfile {
    /// Opaque compiler profile payload passed to XLA.
    bytes: Arc<[u8]>,

    /// XLA profile format version associated with the payload.
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

/// Policy on the host for grouping concrete input extents into bounded specializations of retained JIT dispatchers.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum InputBoundBucketing {
    /// Rounds each positive axis extent observed on the host up to the next power of two. The resulting dynamic type
    /// admits extents through that bucket, bounding padding overhead below two times the logical extent while retaining
    /// only logarithmically many compiled specializations.
    PowerOfTwo,
}

/// Backend options of one compilation for the [`CompilationDomain`](ryft_core::CompilationDomain) implementation on
/// [`Domain`](crate::Domain). Carries the device mesh, optional sharding overrides for inputs and outputs, optional
/// donation flags for each input, backend lowering choices, and policies of retained dispatch.
///
/// These settings supplement the PJRT options configured through
/// [`Domain::compilation_options`](crate::Domain::compilation_options). Settings that affect compilation participate
/// in its cache identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Options {
    /// Explicit compiler and typed embedding used to select retained portable kernel calls.
    kernel_compiler: Option<KernelCompilerBinding>,

    /// Concrete device mesh the compiled program runs against.
    mesh: DeviceMesh,

    /// Optional override for the shardings of the inputs. An override may place an input over auto mesh axes, which
    /// the staged input type does not record (refer to
    /// [`ArrayType::with_sharding`](ryft_core::ArrayType::with_sharding)). The type carries the override without its
    /// auto mesh axes, and the executable argument is placed as the override requests.
    input_shardings: Option<Vec<Sharding>>,

    /// Optional override for the shardings of the outputs. Like [`Self::input_shardings`], an override may place an
    /// output over auto mesh axes, which only its runtime placement records.
    output_shardings: Option<Vec<Sharding>>,

    /// Runtime placements of the flat public inputs of one executable specialization, which retained dispatch derives
    /// from the concrete input arrays (refer to
    /// [`CompilationDomain::dispatch_signature`](ryft_core::CompilationDomain::dispatch_signature)).
    /// They place the executable arguments of inputs without an override over the auto mesh axes that input types
    /// never record, like JAX compiles for the shardings of committed input arrays. Entries of inputs that are not
    /// arrays are [`None`].
    input_placements: Option<Arc<[Option<Sharding>]>>,

    /// Runtime placements of the captured arrays of a staged program, which lowering derives from its capture table,
    /// with the same role as [`Self::input_placements`] for the leading capture arguments.
    capture_placements: Option<Arc<[Option<Sharding>]>>,

    /// Optional flat donation flags, one for each input. `None` donates nothing, and a present vector must match the
    /// program's flat public input arity exactly, which lowering validates.
    donation_flags: Option<Vec<bool>>,

    /// Optional opaque feedback-directed optimization profile forwarded to XLA.
    feedback_directed_profile: Option<FeedbackDirectedProfile>,

    /// Optional policy on the host for bucketing the input bounds of retained JIT signatures.
    input_bound_bucketing: Option<InputBoundBucketing>,

    /// Backend representation used to lower ragged dots. Defaults to [`RaggedDotLoweringStrategy::Decomposition`].
    ragged_dot_lowering_strategy: RaggedDotLoweringStrategy,
}

impl Options {
    /// Constructs options for `mesh` with no sharding overrides or donation and with the portable decomposition of
    /// grouped dots.
    #[inline]
    pub fn new(mesh: DeviceMesh) -> Self {
        Self {
            kernel_compiler: None,
            mesh,
            input_shardings: None,
            output_shardings: None,
            input_placements: None,
            capture_placements: None,
            donation_flags: None,
            feedback_directed_profile: None,
            input_bound_bucketing: None,
            ragged_dot_lowering_strategy: RaggedDotLoweringStrategy::default(),
        }
    }

    /// Selects one immutable compiler and typed embedding for portable kernel calls in this compilation.
    pub fn with_kernel_compiler(mut self, compiler: KernelCompilerBinding) -> Self {
        self.kernel_compiler = Some(compiler);
        self
    }

    /// Sets the sharding overrides of the inputs, which replace the shardings of the flat input types before tracing.
    /// Executions reshard concrete inputs to them, and calls of the staged function bring their inputs to them in the
    /// calling context (refer to the [`functions`](crate::compilation::functions) module documentation). Their number
    /// must equal the flat input arity, and a mismatch surfaces as
    /// [`CompilationError::ShardingCountMismatch`](crate::CompilationError::ShardingCountMismatch).
    #[inline]
    pub fn with_input_shardings(mut self, input_shardings: Vec<Sharding>) -> Self {
        self.input_shardings = Some(input_shardings);
        self
    }

    /// Sets the sharding overrides of the outputs. The staging entry points of the
    /// [`functions`](crate::compilation::functions) module stage them on the outputs of the traced function, through a
    /// `reshard` over explicit mesh axes and a `constrain_sharding` over auto mesh axes, so that calls of the staged
    /// function return the output types that its executions return, and lowering places the executable outputs
    /// accordingly. Their number must equal the flat output arity, and a mismatch surfaces as
    /// [`CompilationError::ShardingCountMismatch`](crate::CompilationError::ShardingCountMismatch).
    #[inline]
    pub fn with_output_shardings(mut self, output_shardings: Vec<Sharding>) -> Self {
        self.output_shardings = Some(output_shardings);
        self
    }

    /// Sets the donation flags of the inputs from a [`Parameterized`] tree of `bool`s whose leaf structure matches the
    /// function's flat input layout.
    ///
    /// Each `true` leaf marks the corresponding input as donatable, which means that the executor may reuse its device
    /// buffer for an output, leaving the caller's runtime value in a donated state after the call returns. The tree is
    /// flattened into [`Self::donation_flags`], and the length of the resulting vector is validated against the
    /// function's flat public input arity at lowering time.
    ///
    /// A single input leaf takes a single `bool` (e.g., `with_donate(true)` for a closure with one argument), while
    /// nested tuple or struct inputs take the matching nested tuple or struct of `bool`s.
    #[inline]
    pub fn with_donate<P: Parameterized<bool>>(mut self, donate: P) -> Self {
        self.donation_flags = Some(donate.into_parameters().collect());
        self
    }

    /// Supplies an opaque feedback-directed optimization profile for manual profile-guided recompilation.
    ///
    /// The profile must have been produced for compatible StableHLO, topology, and compiler settings. XLA performs
    /// backend-specific validation according to its debug options, and Ryft includes the exact bytes in every cache
    /// key.
    #[inline]
    pub fn with_feedback_directed_profile(mut self, profile: FeedbackDirectedProfile) -> Self {
        self.feedback_directed_profile = Some(profile);
        self
    }

    /// Enables bucketing of the input bounds of retained JIT dispatch on the host.
    ///
    /// Static runtime array axes become bounded dynamic axes in the staged signature, with bounds selected by
    /// `bucketing`. The observed extent remains entirely on the host, participates in dispatch identity only through
    /// its bucket, and is transported through the existing ABI for bounded inputs. Extents that the program derives
    /// from data are deliberately unaffected because observing them here would require a device-to-host
    /// synchronization and a split compilation boundary. Bucketing currently accepts only array inputs on a mesh with a
    /// single device, because bucketed signatures are dynamic and therefore require disabling the Shardy partitioner,
    /// which Ryft does not do implicitly for a request over multiple devices.
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

    /// Records the runtime placements of the inputs over all mesh axes, including the auto mesh axes that their types
    /// do not record, for the retained dispatch and eager paths that compile for the arrays that they receive.
    #[inline]
    pub(crate) fn with_input_placements(mut self, input_placements: Arc<[Option<Sharding>]>) -> Self {
        self.input_placements = Some(input_placements);
        self
    }

    /// Records the runtime placements of the captured arrays over all mesh axes, including the auto mesh axes that
    /// their types do not record.
    #[inline]
    pub(crate) fn with_capture_placements(mut self, capture_placements: Arc<[Option<Sharding>]>) -> Self {
        self.capture_placements = Some(capture_placements);
        self
    }

    /// Returns the explicit compiler and typed embedding used to select retained portable kernel calls.
    #[inline]
    pub fn kernel_compiler(&self) -> Option<&KernelCompilerBinding> {
        self.kernel_compiler.as_ref()
    }

    /// Returns the concrete device mesh the compiled program runs against.
    #[inline]
    pub fn mesh(&self) -> &DeviceMesh {
        &self.mesh
    }

    /// Returns the optional override for the shardings of the inputs. An override may place an input over auto mesh
    /// axes, which the staged input type does not record (refer to
    /// [`ArrayType::with_sharding`](ryft_core::ArrayType::with_sharding)). The type carries the override without its
    /// auto mesh axes, and the executable argument is placed as the override requests.
    #[inline]
    pub fn input_shardings(&self) -> Option<&[Sharding]> {
        self.input_shardings.as_deref()
    }

    /// Returns the optional override for the shardings of the outputs. Like [`Self::input_shardings`], an override may
    /// place an output over auto mesh axes, which only its runtime placement records.
    #[inline]
    pub fn output_shardings(&self) -> Option<&[Sharding]> {
        self.output_shardings.as_deref()
    }

    /// Returns the runtime placements of the flat public inputs of one executable specialization, which retained
    /// dispatch derives from the concrete input arrays (refer to
    /// [`CompilationDomain::dispatch_signature`](ryft_core::CompilationDomain::dispatch_signature)). They place the
    /// executable arguments of inputs without an override over the auto mesh axes that input types never record, like
    /// JAX compiles for the shardings of committed input arrays. Entries of inputs that are not arrays are [`None`].
    #[inline]
    pub(crate) fn input_placements(&self) -> Option<&[Option<Sharding>]> {
        self.input_placements.as_deref()
    }

    /// Returns the runtime placements of the captured arrays of a staged program, which lowering derives from its
    /// capture table, with the same role as [`Self::input_placements`] for the leading capture arguments.
    #[inline]
    pub(crate) fn capture_placements(&self) -> Option<&[Option<Sharding>]> {
        self.capture_placements.as_deref()
    }

    /// Returns the optional flat donation flags, one for each input. `None` donates nothing, and a present vector must
    /// match the program's flat public input arity exactly, which lowering validates.
    #[inline]
    pub fn donation_flags(&self) -> Option<&[bool]> {
        self.donation_flags.as_deref()
    }

    /// Returns the optional opaque feedback-directed optimization profile forwarded to XLA.
    #[inline]
    pub fn feedback_directed_profile(&self) -> Option<&FeedbackDirectedProfile> {
        self.feedback_directed_profile.as_ref()
    }

    /// Returns the optional policy on the host for bucketing the input bounds of retained JIT signatures.
    #[inline]
    pub fn input_bound_bucketing(&self) -> Option<InputBoundBucketing> {
        self.input_bound_bucketing
    }

    /// Returns the backend representation used to lower ragged dots. Defaults to
    /// [`RaggedDotLoweringStrategy::Decomposition`].
    #[inline]
    pub fn ragged_dot_lowering_strategy(&self) -> RaggedDotLoweringStrategy {
        self.ragged_dot_lowering_strategy
    }
}

impl Hash for Options {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // `DeviceMesh` does not derive `Hash`, so hash its logical mesh and its device order manually instead.
        // Everything else here derives `Hash` already.
        self.mesh.logical_mesh().hash(state);
        self.mesh.devices().hash(state);
        self.input_shardings.hash(state);
        self.output_shardings.hash(state);
        self.input_placements.hash(state);
        self.capture_placements.hash(state);
        self.donation_flags.hash(state);
        self.feedback_directed_profile.hash(state);
        self.input_bound_bucketing.hash(state);
        self.ragged_dot_lowering_strategy.hash(state);
        self.kernel_compiler.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use std::io::ErrorKind;

    use pretty_assertions::assert_eq;
    use ryft_core::ShardingDimension;
    use ryft_pjrt::protos::CompilationOptions;

    use crate::compilation::domains::jit_compilation_options;
    use crate::tests::{device_mesh_2x2, hash_of};

    use super::*;

    #[test]
    fn test_feedback_directed_profile_new() {
        let profile = FeedbackDirectedProfile::new(vec![1, 2, 3, 4]);
        assert_eq!(profile.bytes(), &[1, 2, 3, 4]);
        assert_eq!(profile.version(), 0);
        assert_eq!(profile.clone().with_version(7).version(), 7);
        assert_eq!(
            profile.digest(),
            [
                159, 100, 167, 71, 225, 185, 127, 19, 31, 171, 182, 180, 71, 41, 108, 155, 111, 2, 1, 231, 159, 179,
                197, 53, 110, 108, 119, 232, 155, 106, 128, 106,
            ],
        );
    }

    #[test]
    fn test_feedback_directed_profile_from_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("profile.pb");
        let profile = FeedbackDirectedProfile::new(vec![1, 2, 3, 4]).with_version(7);
        profile.write_to_file(&path).unwrap();
        let restored = FeedbackDirectedProfile::from_file(&path).unwrap().with_version(7);
        assert_eq!(restored, profile);
        assert_eq!(restored.digest(), profile.digest());

        let baseline = jit_compilation_options(&CompilationOptions::default(), 2, true, None);
        let profiled = jit_compilation_options(&CompilationOptions::default(), 2, true, Some(&profile));
        assert_eq!(baseline.executable_build_options.unwrap().fdo_profile, Vec::<u8>::new());
        assert_eq!(profiled.executable_build_options.unwrap().fdo_profile, vec![1, 2, 3, 4]);
        assert_eq!(profiled.profile_version, 7);
    }

    #[test]
    fn test_feedback_directed_profile_from_file_missing() {
        let directory = tempfile::tempdir().unwrap();
        assert!(matches!(
            FeedbackDirectedProfile::from_file(directory.path().join("missing.pb")),
            Err(Error::Io(error)) if error.kind() == ErrorKind::NotFound,
        ));
    }

    #[test]
    fn test_options_new() {
        let mesh = device_mesh_2x2();
        let options = Options::new(mesh.clone());
        assert_eq!(options.kernel_compiler(), None);
        assert_eq!(options.mesh(), &mesh);
        assert_eq!(options.input_shardings(), None);
        assert_eq!(options.output_shardings(), None);
        assert_eq!(options.input_placements(), None);
        assert_eq!(options.capture_placements(), None);
        assert_eq!(options.donation_flags(), None);
        assert_eq!(options.feedback_directed_profile(), None);
        assert_eq!(options.input_bound_bucketing(), None);
        assert_eq!(options.ragged_dot_lowering_strategy(), RaggedDotLoweringStrategy::Decomposition);
    }

    #[test]
    fn test_options_with_input_shardings() {
        let mesh = device_mesh_2x2();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let options = Options::new(mesh).with_input_shardings(vec![sharding.clone()]);
        assert_eq!(options.input_shardings(), Some([sharding].as_slice()));
        assert_eq!(options.output_shardings(), None);
    }

    #[test]
    fn test_options_with_output_shardings() {
        let mesh = device_mesh_2x2();
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let options = Options::new(mesh).with_output_shardings(vec![sharding.clone()]);
        assert_eq!(options.output_shardings(), Some([sharding].as_slice()));
        assert_eq!(options.input_shardings(), None);
    }

    #[test]
    fn test_options_with_donate() {
        let options = Options::new(device_mesh_2x2()).with_donate((true, (false, true)));
        assert_eq!(options.donation_flags(), Some([true, false, true].as_slice()));
    }

    #[test]
    fn test_options_with_feedback_directed_profile() {
        let profile = FeedbackDirectedProfile::new(vec![1, 2, 3]);
        let options = Options::new(device_mesh_2x2()).with_feedback_directed_profile(profile.clone());
        assert_eq!(options.feedback_directed_profile(), Some(&profile));
    }

    #[test]
    fn test_options_with_input_bound_bucketing() {
        let options = Options::new(device_mesh_2x2()).with_input_bound_bucketing(InputBoundBucketing::PowerOfTwo);
        assert_eq!(options.input_bound_bucketing(), Some(InputBoundBucketing::PowerOfTwo));
    }

    #[test]
    fn test_options_with_ragged_dot_lowering_strategy() {
        let options =
            Options::new(device_mesh_2x2()).with_ragged_dot_lowering_strategy(RaggedDotLoweringStrategy::Instruction);
        assert_eq!(options.ragged_dot_lowering_strategy(), RaggedDotLoweringStrategy::Instruction);
    }

    #[test]
    fn test_options_hash() {
        let options = Options::new(device_mesh_2x2());
        let donated = options.clone().with_donate(true);
        assert_eq!(hash_of(&options), hash_of(&options.clone()));
        assert_eq!(hash_of(&donated), hash_of(&options.clone().with_donate(true)));
        assert_ne!(donated, options);
        assert_ne!(hash_of(&donated), hash_of(&options));
    }
}
