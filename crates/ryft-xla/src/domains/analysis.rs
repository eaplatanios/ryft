//! Compiled XLA program inspection and analysis.

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::time::Duration;

use ryft_core::{AnalyzableCompilationDomain, ArrayIrType, Parameterized};
use ryft_pjrt::Value as PjrtValue;
use serde::Serialize;

use crate::Error;
use crate::domains::Domain;
use crate::domains::compilation::CompilationError;
use crate::domains::programs::CompiledProgram;

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

/// Normalized memory requirements reported by PJRT for one XLA executable.
#[derive(Copy, Clone, Debug, Serialize, PartialEq, Eq)]
pub struct MemoryAnalysis {
    pub device_generated_code_size_in_bytes: usize,
    pub device_input_size_in_bytes: usize,
    pub device_output_size_in_bytes: usize,
    pub device_alias_size_in_bytes: usize,
    pub device_temporary_size_in_bytes: usize,
    pub device_peak_memory_in_bytes: usize,
    pub device_total_memory_in_bytes: usize,
    pub device_total_allocation_bytes: usize,
    pub device_indefinite_allocations: usize,
    pub device_peak_unpadded_heap_bytes: usize,
    pub host_generated_code_size_in_bytes: usize,
    pub host_input_size_in_bytes: usize,
    pub host_output_size_in_bytes: usize,
    pub host_alias_size_in_bytes: usize,
    pub host_temporary_size_in_bytes: usize,
}

/// Backend-neutral-facing, typed analysis of one compiled XLA executable.
///
/// PJRT plugins expose different property sets. Common properties are normalized into optional fields and every raw
/// property is retained in [`Self::properties`]. Unsupported optional queries remain `None`.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct CompilationAnalysis {
    pub floating_point_operations: Option<f64>,
    pub transcendental_operations: Option<f64>,
    pub bytes_accessed: Option<f64>,
    pub memory: Option<MemoryAnalysis>,
    pub generated_code_size_in_bytes: Option<usize>,
    pub replica_count: Option<usize>,
    pub partition_count: Option<usize>,
    pub executable_fingerprint: Option<String>,
    pub compilation_duration: Option<Duration>,
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
        executable: &ryft_core::compilation::ExecutableFunction<Self, Input, Output>,
    ) -> Result<Self::Analysis, Self::Error> {
        let program = executable.compiled_program();
        program.analysis.get_or_init(|| analyze_program(program)).clone()
    }
}
pub(super) fn analyze_program(program: &CompiledProgram<'_>) -> Result<CompilationAnalysis, Error> {
    let executable = program.executable.executable()?;
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
        compilation_duration: program.compilation_duration,
        properties,
    })
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
