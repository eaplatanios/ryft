use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::time::Duration;

use ryft_core::{AnalyzableCompilationDomain, ArrayIrType, ExecutableFunction, Parameterized};
use ryft_pjrt::Value as PjrtValue;
use serde::Serialize;

use crate::Error;
use crate::domains::Domain;
use crate::domains::compilation::CompilationError;
use crate::domains::programs::CompiledProgram;

// TODO(eaplatanios): Review this module.

/// Typed value returned for a plugin-specific PJRT cost-analysis property.
#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum AnalysisValue {
    /// Boolean property.
    Boolean(bool),

    /// Signed integer property.
    Integer(i64),

    /// Signed integer-list property.
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
    pub device_generated_code_size_in_bytes: usize,

    /// Bytes occupied by executable input buffers in device memory.
    pub device_input_size_in_bytes: usize,

    /// Bytes occupied by executable output buffers in device memory.
    pub device_output_size_in_bytes: usize,

    /// Device bytes shared by aliased input and output buffers.
    pub device_alias_size_in_bytes: usize,

    /// Bytes required for temporary buffers in device memory.
    pub device_temporary_size_in_bytes: usize,

    /// Peak device memory use reported by the backend.
    pub device_peak_memory_in_bytes: usize,

    /// Total device memory capacity reported by the backend.
    pub device_total_memory_in_bytes: usize,

    /// Bytes held by the device allocator, including retained memory pools.
    pub device_total_allocation_bytes: usize,

    /// Number of device allocations without a bounded allocation/deallocation interval in the backend schedule.
    pub device_indefinite_allocations: usize,

    /// Peak device heap use before allocator padding, in bytes.
    pub device_peak_unpadded_heap_bytes: usize,

    /// Bytes occupied by generated code in host memory.
    pub host_generated_code_size_in_bytes: usize,

    /// Bytes occupied by executable input buffers in host memory.
    pub host_input_size_in_bytes: usize,

    /// Bytes occupied by executable output buffers in host memory.
    pub host_output_size_in_bytes: usize,

    /// Host bytes shared by aliased input and output buffers.
    pub host_alias_size_in_bytes: usize,

    /// Bytes required for temporary buffers in host memory.
    pub host_temporary_size_in_bytes: usize,
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
    pub floating_point_operations: Option<f64>,

    /// Estimated transcendental operation count from the plugin cost analysis, when available.
    pub transcendental_operations: Option<f64>,

    /// Estimated number of bytes accessed according to the plugin cost analysis, when available.
    pub bytes_accessed: Option<f64>,

    /// Executable memory statistics, when supported by the plugin.
    pub memory: Option<MemoryAnalysis>,

    /// Size of generated executable code in bytes, when supported by the plugin.
    pub generated_code_size_in_bytes: Option<usize>,

    /// Number of executable replicas, when supported by the plugin.
    pub replica_count: Option<usize>,

    /// Number of executable computation partitions, when supported by the plugin.
    pub partition_count: Option<usize>,

    /// Plugin-provided executable fingerprint, when supported.
    pub executable_fingerprint: Option<String>,

    /// Wall-clock duration of the PJRT compilation request, retained across persistent transport when present.
    pub compilation_duration: Option<Duration>,

    /// All plugin-specific cost-analysis properties, including those normalized into the fields above.
    pub properties: BTreeMap<String, AnalysisValue>,
}

impl CompilationAnalysis {
    /// Returns a deterministic JSON representation suitable for machine-readable diagnostics.
    pub fn to_json(&self) -> Result<String, Error> {
        serde_json::to_string(self)
            .map_err(|error| Error::Compilation(CompilationError::InvalidAnalysis { reason: error.to_string() }))
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
    /// PJRT program format (`mlir`, `hlo`, or `hlo_with_config`).
    pub format: &'static str,

    /// Backend-owned optimized program bytes.
    pub bytes: Vec<u8>,
}

impl<'c> AnalyzableCompilationDomain for Domain<'c> {
    type Analysis = CompilationAnalysis;

    fn analyze<Input: Parameterized<ArrayIrType>, Output: Parameterized<ArrayIrType>>(
        &self,
        executable: &ExecutableFunction<Self, Input, Output>,
    ) -> Result<Self::Analysis, Self::Error> {
        let program = executable.compiled_program();
        program.analysis.get_or_init(|| program.analyze()).clone()
    }
}

impl CompiledProgram<'_> {
    /// Queries PJRT analysis without updating the shared analysis cache.
    pub(super) fn analyze(&self) -> Result<CompilationAnalysis, Error> {
        let executable = self.executable.executable()?;
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
            compilation_duration: self.compilation_duration,
            properties,
        })
    }
}

/// Returns the first named numeric cost property, or `None` if every name is absent. A present property of another type
/// is invalid analysis data.
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

    use crate::Error;
    use crate::domains::options::Options;
    use crate::domains::sessions::Session;
    use crate::domains::tests::*;
    use crate::tests::execution_client;

    use super::*;

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
            .analysis
            .set(Err(ryft_pjrt::Error::invalid_argument("analysis query failed").into()))
            .unwrap();

        // Repeated access preserves the runtime's category and diagnostic rather than relabeling cached failures
        // as invalid analysis data or adding another display prefix.
        for _ in 0..2 {
            let error = domain.analyze(compiled.executable()).unwrap_err();
            assert!(matches!(&error,
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
}
