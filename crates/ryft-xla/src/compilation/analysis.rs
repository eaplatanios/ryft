//! Compiler diagnostics of compiled XLA programs.
//!
//! [`CompilationAnalysis`] normalizes the cost and memory properties that PJRT reports for one [`CompiledProgram`]
//! into typed fields, retaining every raw property as an [`AnalysisValue`], and [`MemoryAnalysis`] holds the executable
//! and allocator memory statistics. The [`AnalyzableCompilationDomain`] implementation of [`Domain`] caches each
//! analysis on its program. [`OptimizedProgram`] holds the optimized program that the backend returns on explicit
//! inspection.

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::time::Duration;

use ryft_core::{AnalyzableCompilationDomain, ArrayIrType, ExecutableFunction, Parameterized};
use ryft_pjrt::Value as PjrtValue;
use serde::Serialize;

use crate::compilation::CompilationError;
use crate::compilation::programs::CompiledProgram;
use crate::{Domain, Error};

// TODO(eaplatanios): Review this module.

/// Typed value of a PJRT cost analysis property that is specific to a plugin.
#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum AnalysisValue {
    /// Boolean property.
    Boolean(bool),

    /// Signed integer property.
    Integer(i64),

    /// Property that holds a list of signed integers.
    IntegerList(Vec<i64>),

    /// Floating-point property.
    Float(f64),

    /// String property.
    String(String),
}

impl From<&PjrtValue> for AnalysisValue {
    fn from(value: &PjrtValue) -> Self {
        match value {
            PjrtValue::Bool(value) => Self::Boolean(*value),
            PjrtValue::I64(value) => Self::Integer(*value),
            PjrtValue::I64List(value) => Self::IntegerList(value.clone()),
            PjrtValue::F32(value) => Self::Float(*value as f64),
            PjrtValue::String(value) => Self::String(value.clone()),
        }
    }
}

/// Executable and allocator memory statistics reported by PJRT.
#[derive(Copy, Clone, Debug, Serialize, PartialEq, Eq)]
pub struct MemoryAnalysis {
    /// Bytes occupied by generated code in device memory.
    device_generated_code_size_in_bytes: usize,

    /// Bytes occupied by executable input buffers in device memory.
    device_input_size_in_bytes: usize,

    /// Bytes occupied by executable output buffers in device memory.
    device_output_size_in_bytes: usize,

    /// Device bytes shared by aliased input and output buffers.
    device_alias_size_in_bytes: usize,

    /// Bytes required for temporary buffers in device memory.
    device_temporary_size_in_bytes: usize,

    /// Peak device memory use reported by the backend.
    device_peak_memory_in_bytes: usize,

    /// Total device memory capacity reported by the backend.
    device_total_memory_in_bytes: usize,

    /// Bytes held by the device allocator, including retained memory pools.
    device_total_allocation_bytes: usize,

    /// Number of device allocations without a bounded interval between allocation and deallocation in the backend
    /// schedule.
    device_indefinite_allocations: usize,

    /// Peak device heap use before allocator padding, in bytes.
    device_peak_unpadded_heap_bytes: usize,

    /// Bytes occupied by generated code in host memory.
    host_generated_code_size_in_bytes: usize,

    /// Bytes occupied by executable input buffers in host memory.
    host_input_size_in_bytes: usize,

    /// Bytes occupied by executable output buffers in host memory.
    host_output_size_in_bytes: usize,

    /// Host bytes shared by aliased input and output buffers.
    host_alias_size_in_bytes: usize,

    /// Bytes required for temporary buffers in host memory.
    host_temporary_size_in_bytes: usize,
}

impl MemoryAnalysis {
    /// Returns the bytes occupied by generated code in device memory.
    #[inline]
    pub fn device_generated_code_size_in_bytes(&self) -> usize {
        self.device_generated_code_size_in_bytes
    }

    /// Returns the bytes occupied by executable input buffers in device memory.
    #[inline]
    pub fn device_input_size_in_bytes(&self) -> usize {
        self.device_input_size_in_bytes
    }

    /// Returns the bytes occupied by executable output buffers in device memory.
    #[inline]
    pub fn device_output_size_in_bytes(&self) -> usize {
        self.device_output_size_in_bytes
    }

    /// Returns the device bytes shared by aliased input and output buffers.
    #[inline]
    pub fn device_alias_size_in_bytes(&self) -> usize {
        self.device_alias_size_in_bytes
    }

    /// Returns the bytes required for temporary buffers in device memory.
    #[inline]
    pub fn device_temporary_size_in_bytes(&self) -> usize {
        self.device_temporary_size_in_bytes
    }

    /// Returns the peak device memory use reported by the backend.
    #[inline]
    pub fn device_peak_memory_in_bytes(&self) -> usize {
        self.device_peak_memory_in_bytes
    }

    /// Returns the total device memory capacity reported by the backend.
    #[inline]
    pub fn device_total_memory_in_bytes(&self) -> usize {
        self.device_total_memory_in_bytes
    }

    /// Returns the bytes held by the device allocator, including retained memory pools.
    #[inline]
    pub fn device_total_allocation_bytes(&self) -> usize {
        self.device_total_allocation_bytes
    }

    /// Returns the number of device allocations without a bounded interval between allocation and deallocation in the
    /// backend schedule.
    #[inline]
    pub fn device_indefinite_allocations(&self) -> usize {
        self.device_indefinite_allocations
    }

    /// Returns the peak device heap use before allocator padding, in bytes.
    #[inline]
    pub fn device_peak_unpadded_heap_bytes(&self) -> usize {
        self.device_peak_unpadded_heap_bytes
    }

    /// Returns the bytes occupied by generated code in host memory.
    #[inline]
    pub fn host_generated_code_size_in_bytes(&self) -> usize {
        self.host_generated_code_size_in_bytes
    }

    /// Returns the bytes occupied by executable input buffers in host memory.
    #[inline]
    pub fn host_input_size_in_bytes(&self) -> usize {
        self.host_input_size_in_bytes
    }

    /// Returns the bytes occupied by executable output buffers in host memory.
    #[inline]
    pub fn host_output_size_in_bytes(&self) -> usize {
        self.host_output_size_in_bytes
    }

    /// Returns the host bytes shared by aliased input and output buffers.
    #[inline]
    pub fn host_alias_size_in_bytes(&self) -> usize {
        self.host_alias_size_in_bytes
    }

    /// Returns the bytes required for temporary buffers in host memory.
    #[inline]
    pub fn host_temporary_size_in_bytes(&self) -> usize {
        self.host_temporary_size_in_bytes
    }
}

/// Typed cost, memory, and executable metadata for one compiled XLA program.
///
/// PJRT plugins expose different property sets. Common properties are normalized into optional fields and every raw
/// property is retained in [`Self::properties`]. Unsupported optional queries remain `None`.
///
/// [`AnalyzableCompilationDomain::analyze`] on a [`Domain`] caches both successful analyses and failures on the
/// corresponding [`CompiledProgram`], so repeated inspection does not issue new backend queries. Optimized compiler
/// representations are inspected separately through [`CompiledProgram::optimized_program`].
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct CompilationAnalysis {
    /// Estimated floating-point operation count from the plugin cost analysis, when available.
    floating_point_operations: Option<f64>,

    /// Estimated transcendental operation count from the plugin cost analysis, when available.
    transcendental_operations: Option<f64>,

    /// Estimated number of bytes accessed according to the plugin cost analysis, when available.
    bytes_accessed: Option<f64>,

    /// Executable memory statistics, when supported by the plugin.
    memory: Option<MemoryAnalysis>,

    /// Size of generated executable code in bytes, when supported by the plugin.
    generated_code_size_in_bytes: Option<usize>,

    /// Number of executable replicas, when supported by the plugin.
    replica_count: Option<usize>,

    /// Number of executable computation partitions, when supported by the plugin.
    partition_count: Option<usize>,

    /// Executable fingerprint provided by the plugin, when supported.
    executable_fingerprint: Option<String>,

    /// Wall-clock duration of the PJRT compilation request, retained across persistent transport when present.
    compilation_duration: Option<Duration>,

    /// All cost analysis properties of the plugin, including those normalized into the fields above.
    properties: BTreeMap<String, AnalysisValue>,
}

impl CompilationAnalysis {
    /// Returns a deterministic JSON representation suitable for machine-readable diagnostics.
    pub fn to_json(&self) -> Result<String, Error> {
        serde_json::to_string(self)
            .map_err(|error| Error::Compilation(CompilationError::InvalidAnalysis { reason: error.to_string() }))
    }

    /// Returns the estimated floating-point operation count from the plugin cost analysis, when available.
    #[inline]
    pub fn floating_point_operations(&self) -> Option<f64> {
        self.floating_point_operations
    }

    /// Returns the estimated transcendental operation count from the plugin cost analysis, when available.
    #[inline]
    pub fn transcendental_operations(&self) -> Option<f64> {
        self.transcendental_operations
    }

    /// Returns the estimated number of bytes accessed according to the plugin cost analysis, when available.
    #[inline]
    pub fn bytes_accessed(&self) -> Option<f64> {
        self.bytes_accessed
    }

    /// Returns the executable memory statistics, when supported by the plugin.
    #[inline]
    pub fn memory(&self) -> Option<&MemoryAnalysis> {
        self.memory.as_ref()
    }

    /// Returns the size of generated executable code in bytes, when supported by the plugin.
    #[inline]
    pub fn generated_code_size_in_bytes(&self) -> Option<usize> {
        self.generated_code_size_in_bytes
    }

    /// Returns the number of executable replicas, when supported by the plugin.
    #[inline]
    pub fn replica_count(&self) -> Option<usize> {
        self.replica_count
    }

    /// Returns the number of executable computation partitions, when supported by the plugin.
    #[inline]
    pub fn partition_count(&self) -> Option<usize> {
        self.partition_count
    }

    /// Returns the executable fingerprint provided by the plugin, when supported.
    #[inline]
    pub fn executable_fingerprint(&self) -> Option<&str> {
        self.executable_fingerprint.as_deref()
    }

    /// Returns the wall-clock duration of the PJRT compilation request, retained across persistent transport when
    /// present.
    #[inline]
    pub fn compilation_duration(&self) -> Option<Duration> {
        self.compilation_duration
    }

    /// Returns the all cost analysis properties of the plugin, including those normalized into the fields above.
    #[inline]
    pub fn properties(&self) -> &BTreeMap<String, AnalysisValue> {
        &self.properties
    }
}

impl Display for CompilationAnalysis {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "XLA compilation analysis [flops={}, bytes accessed={}, generated code={} bytes, replicas={}, \
             partitions={}]",
            self.floating_point_operations
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unsupported".into()),
            self.bytes_accessed.map(|value| value.to_string()).unwrap_or_else(|| "unsupported".into()),
            self.generated_code_size_in_bytes
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unsupported".into()),
            self.replica_count.map(|value| value.to_string()).unwrap_or_else(|| "unsupported".into()),
            self.partition_count.map(|value| value.to_string()).unwrap_or_else(|| "unsupported".into()),
        )
    }
}

/// Optimized compiler program returned on explicit inspection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OptimizedProgram {
    /// PJRT program format (i.e., `mlir`, `hlo`, or `hlo_with_config`).
    format: &'static str,

    /// Optimized program bytes owned by the backend.
    bytes: Vec<u8>,
}

impl OptimizedProgram {
    /// Creates an [`OptimizedProgram`] from the program that a PJRT executable reports as its optimized form.
    pub(crate) fn new(program: ryft_pjrt::Program) -> Self {
        match program {
            ryft_pjrt::Program::Mlir { bytecode } => Self { format: "mlir", bytes: bytecode },
            ryft_pjrt::Program::Hlo { proto } => Self { format: "hlo", bytes: proto },
            ryft_pjrt::Program::HloWithConfig { proto } => Self { format: "hlo_with_config", bytes: proto },
        }
    }

    /// Returns the PJRT program format (i.e., `mlir`, `hlo`, or `hlo_with_config`).
    #[inline]
    pub fn format(&self) -> &'static str {
        self.format
    }

    /// Returns the optimized program bytes owned by the backend.
    #[inline]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl AnalyzableCompilationDomain for Domain<'_> {
    type Analysis = CompilationAnalysis;

    fn analyze<Input: Parameterized<ArrayIrType>, Output: Parameterized<ArrayIrType>>(
        &self,
        executable: &ExecutableFunction<Self, Input, Output>,
    ) -> Result<Self::Analysis, Self::Error> {
        let program = executable.compiled_program();
        program.analysis().get_or_init(|| program.analyze()).clone()
    }
}

impl CompiledProgram<'_> {
    /// Queries PJRT analysis without updating the shared analysis cache.
    pub(super) fn analyze(&self) -> Result<CompilationAnalysis, Error> {
        let executable = self.executable().executable()?;
        let properties = match executable.cost_analysis() {
            Ok(properties) => properties
                .iter()
                .map(|(name, value)| (name.clone(), AnalysisValue::from(value)))
                .collect::<BTreeMap<_, _>>(),
            Err(ryft_pjrt::Error::Unimplemented { .. } | ryft_pjrt::Error::Unavailable { .. }) => BTreeMap::new(),
            Err(error) => return Err(error.into()),
        };
        let memory = match executable.memory_statistics() {
            Ok(statistics) => Some(MemoryAnalysis {
                device_generated_code_size_in_bytes: statistics.device_generated_code_size_in_bytes,
                device_input_size_in_bytes: statistics.device_input_size_in_bytes,
                device_output_size_in_bytes: statistics.device_output_size_in_bytes,
                device_alias_size_in_bytes: statistics.device_alias_size_in_bytes,
                device_temporary_size_in_bytes: statistics.device_temporary_size_in_bytes,
                device_peak_memory_in_bytes: statistics.device_peak_memory_in_bytes,
                device_total_memory_in_bytes: statistics.device_total_memory_in_bytes,
                device_total_allocation_bytes: statistics.device_total_allocation_bytes,
                device_indefinite_allocations: statistics.device_indefinite_allocations,
                device_peak_unpadded_heap_bytes: statistics.device_peak_unpadded_heap_bytes,
                host_generated_code_size_in_bytes: statistics.host_generated_code_size_in_bytes,
                host_input_size_in_bytes: statistics.host_input_size_in_bytes,
                host_output_size_in_bytes: statistics.host_output_size_in_bytes,
                host_alias_size_in_bytes: statistics.host_alias_size_in_bytes,
                host_temporary_size_in_bytes: statistics.host_temporary_size_in_bytes,
            }),
            Err(ryft_pjrt::Error::Unimplemented { .. } | ryft_pjrt::Error::Unavailable { .. }) => None,
            Err(error) => return Err(error.into()),
        };
        Ok(CompilationAnalysis {
            floating_point_operations: numeric_analysis_property(&properties, &["flops", "floating_point_operations"])?,
            transcendental_operations: numeric_analysis_property(
                &properties,
                &["transcendentals", "transcendental_operations"],
            )?,
            bytes_accessed: numeric_analysis_property(&properties, &["bytes accessed", "bytes_accessed"])?,
            memory,
            generated_code_size_in_bytes: optional_pjrt_analysis(executable.generated_code_size_in_bytes())?,
            replica_count: optional_pjrt_analysis(executable.replica_count())?,
            partition_count: optional_pjrt_analysis(executable.computation_count())?,
            executable_fingerprint: optional_pjrt_analysis(executable.fingerprint().map(|value| value.into_owned()))?,
            compilation_duration: self.compilation_duration(),
            properties,
        })
    }
}

/// Returns the first named numeric cost property, or `None` if every name is absent. A present property of another
/// type is invalid analysis data.
fn numeric_analysis_property(
    properties: &BTreeMap<String, AnalysisValue>,
    names: &[&str],
) -> Result<Option<f64>, Error> {
    let Some((name, value)) = names.iter().find_map(|name| properties.get(*name).map(|value| (*name, value))) else {
        return Ok(None);
    };
    match value {
        AnalysisValue::Integer(value) => Ok(Some(*value as f64)),
        AnalysisValue::Float(value) => Ok(Some(*value)),
        _ => Err(Error::Compilation(CompilationError::InvalidAnalysis {
            reason: format!("property `{name}` is not numeric"),
        })),
    }
}

/// Maps unsupported or unavailable PJRT queries to `None`, preserving other backend errors.
fn optional_pjrt_analysis<T>(result: Result<T, ryft_pjrt::Error>) -> Result<Option<T>, Error> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(ryft_pjrt::Error::Unimplemented { .. } | ryft_pjrt::Error::Unavailable { .. }) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use pretty_assertions::assert_eq;
    use ryft_core::{ArrayIrType, ArrayType, DataType, StagedFunction as CoreStagedFunction};

    use crate::compilation::options::Options;
    use crate::domains::tests::domain_mesh;
    use crate::tests::execution_client;
    use crate::{Error, Session};

    use super::*;

    #[test]
    fn test_analysis_value_from() {
        assert_eq!(AnalysisValue::from(&PjrtValue::Bool(true)), AnalysisValue::Boolean(true));
        assert_eq!(AnalysisValue::from(&PjrtValue::I64(-3)), AnalysisValue::Integer(-3));
        assert_eq!(AnalysisValue::from(&PjrtValue::I64List(vec![1, 2])), AnalysisValue::IntegerList(vec![1, 2]));
        assert_eq!(AnalysisValue::from(&PjrtValue::F32(0.5)), AnalysisValue::Float(0.5));
        assert_eq!(
            AnalysisValue::from(&PjrtValue::String("fingerprint".to_string())),
            AnalysisValue::String("fingerprint".to_string()),
        );
    }

    #[test]
    fn test_compilation_analysis_to_json() {
        let analysis = CompilationAnalysis {
            floating_point_operations: Some(12.0),
            transcendental_operations: None,
            bytes_accessed: Some(3.5),
            memory: None,
            generated_code_size_in_bytes: Some(64),
            replica_count: Some(1),
            partition_count: Some(2),
            executable_fingerprint: Some("abc".to_string()),
            compilation_duration: None,
            properties: BTreeMap::from([("flops".to_string(), AnalysisValue::Integer(12))]),
        };
        assert_eq!(
            analysis.to_json().unwrap(),
            "{\"floating_point_operations\":12.0,\"transcendental_operations\":null,\"bytes_accessed\":3.5,\
             \"memory\":null,\"generated_code_size_in_bytes\":64,\"replica_count\":1,\"partition_count\":2,\
             \"executable_fingerprint\":\"abc\",\"compilation_duration\":null,\
             \"properties\":{\"flops\":{\"type\":\"integer\",\"value\":12}}}",
        );
    }

    #[test]
    fn test_compilation_analysis_display() {
        let analysis = CompilationAnalysis {
            floating_point_operations: Some(12.0),
            transcendental_operations: None,
            bytes_accessed: None,
            memory: None,
            generated_code_size_in_bytes: Some(64),
            replica_count: Some(1),
            partition_count: None,
            executable_fingerprint: None,
            compilation_duration: None,
            properties: BTreeMap::new(),
        };
        assert_eq!(
            analysis.to_string(),
            "XLA compilation analysis [flops=12, bytes accessed=unsupported, generated code=64 bytes, replicas=1, \
             partitions=unsupported]",
        );
    }

    #[test]
    fn test_optimized_program_new() {
        let program = OptimizedProgram::new(ryft_pjrt::Program::Mlir { bytecode: vec![1, 2] });
        assert_eq!(program.format(), "mlir");
        assert_eq!(program.bytes(), &[1, 2]);
        let program = OptimizedProgram::new(ryft_pjrt::Program::Hlo { proto: vec![3] });
        assert_eq!(program.format(), "hlo");
        assert_eq!(program.bytes(), &[3]);
        let program = OptimizedProgram::new(ryft_pjrt::Program::HloWithConfig { proto: vec![4] });
        assert_eq!(program.format(), "hlo_with_config");
        assert_eq!(program.bytes(), &[4]);
    }

    #[test]
    fn test_domain_analyze_preserves_cached_errors() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let staged = CoreStagedFunction::<Domain<'_>, ArrayIrType, ArrayIrType>::stage(
            |_, _, input| Ok(input),
            Vec::new(),
            ArrayType::scalar(DataType::F32).into(),
            Options::new(mesh),
        )
        .unwrap();
        let compiled = staged.lower(&domain).unwrap().compile(&domain).unwrap();
        compiled
            .compiled_program()
            .analysis()
            .set(Err(ryft_pjrt::Error::invalid_argument("analysis query failed").into()))
            .unwrap();

        // Repeated access preserves the runtime's category and diagnostic rather than relabeling cached failures
        // as invalid analysis data or adding another display prefix.
        for _ in 0..2 {
            let error = domain.analyze(compiled.executable()).unwrap_err();
            assert!(matches!(
                &error,
                Error::Pjrt(ryft_pjrt::Error::InvalidArgument { message, .. }) if message == "analysis query failed",
            ));
            assert_eq!(error.to_string(), "analysis query failed");
        }
    }

    #[test]
    fn test_numeric_analysis_property() {
        let properties = BTreeMap::from([
            ("flops".to_string(), AnalysisValue::Integer(12)),
            ("bytes_accessed".to_string(), AnalysisValue::Boolean(true)),
        ]);
        assert_eq!(numeric_analysis_property(&properties, &["flops"]).unwrap(), Some(12.0));
        assert_eq!(numeric_analysis_property(&properties, &["missing"]).unwrap(), None);
        assert!(matches!(
            numeric_analysis_property(&properties, &["bytes_accessed"]),
            Err(Error::Compilation(CompilationError::InvalidAnalysis { reason }))
                if reason == "property `bytes_accessed` is not numeric",
        ));
    }

    #[test]
    fn test_optional_pjrt_analysis() {
        assert_eq!(optional_pjrt_analysis(Ok(3usize)).unwrap(), Some(3));
        assert_eq!(optional_pjrt_analysis::<usize>(Err(ryft_pjrt::Error::unimplemented("cost"))).unwrap(), None);
        assert_eq!(optional_pjrt_analysis::<usize>(Err(ryft_pjrt::Error::unavailable("cost"))).unwrap(), None);
        assert!(matches!(
            optional_pjrt_analysis::<usize>(Err(ryft_pjrt::Error::invalid_argument("cost failed"))),
            Err(Error::Pjrt(ryft_pjrt::Error::InvalidArgument { message, .. })) if message == "cost failed",
        ));
    }
}
