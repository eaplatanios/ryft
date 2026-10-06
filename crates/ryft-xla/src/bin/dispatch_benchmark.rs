//! Per-call dispatch latency benchmark for small eager and jitted operations, for comparison with JAX.
//!
//! Every case applies one tiny operation (an `f32[8]` add, a `condition` over such adds, a first-class dimension add,
//! or an `f32[8]` reference read) on a single CPU device, so the measured time is dispatch overhead rather than
//! computation. The dimension and reference cases come in pairs that compare dispatch through the session-carrying
//! [`XlaValue`] members with the direct host implementations of the session-free `ArrayIrValue` members. Each case
//! runs warm-up calls and then several timed rounds without per-call synchronization (the same methodology as
//! `python/scripts/compare_dispatch_performance_with_jax.py`), and reports the per-call mean of every round.

use std::collections::HashMap;
use std::env;
use std::sync::Arc;
use std::time::Instant;

use ryft_core::{
    Add, AddOperation, ArrayIrType, ArrayIrValue, ArrayType, ConditionOperation, Context, DataType, Device, DeviceMesh,
    Dimension, DimensionValue, LogicalMesh, MeshAxis, MeshAxisType, MulOperation, Placeholder, ReferenceNew,
    ReferenceRead, Shape, Sharding,
};
use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
use ryft_pjrt::{ClientOptions, CpuClientOptions, ExecutionDeviceInputs, ExecutionInput, Program, load_cpu_plugin};
use ryft_xla::experimental::ops::{XlaConstant, XlaOperation, XlaProgramBuilder};
use ryft_xla::{FromPjrt, JittedXlaFunction, XlaArray, XlaCompileTracer, XlaDimension, XlaSession, XlaValue, jitted};
use serde_json::json;

/// Command-line arguments of this benchmark.
#[derive(Debug)]
struct Arguments {
    /// Number of timed calls per round.
    iterations: usize,

    /// Number of timed rounds per case.
    rounds: usize,

    /// Name of the only case to run, if one was selected (e.g., to attach a sampling profiler to a single case).
    case: Option<String>,
}

impl Arguments {
    /// Returns whether the case named `case` should run.
    fn selects(&self, case: &str) -> bool {
        self.case.as_deref().is_none_or(|selected| selected == case)
    }
}

/// Returns the usage message of this benchmark.
fn usage() -> &'static str {
    "Usage: dispatch_benchmark [OPTIONS]\n\
     \n\
     Options:\n\
       --iterations N     Timed calls per round (default: 10000)\n\
       --rounds N         Timed rounds per case (default: 5)\n\
       --case NAME        Only run the named case (`eager_add`, `eager_condition`, `xla_dimension_add`,\n\
                          `host_dimension_add`, `xla_reference_read`, `host_reference_read`, `jit_add`, or\n\
                          `pjrt_add`)\n\
       --smoke            Use small defaults\n\
       -h, --help         Print this help"
}

/// Parses the command-line arguments, returning [`None`] when only the usage message was requested.
fn parse_arguments() -> Result<Option<Arguments>, String> {
    let mut arguments = Arguments { iterations: 10_000, rounds: 5, case: None };
    let mut values = env::args().skip(1);
    while let Some(argument) = values.next() {
        let mut value = |name: &str| -> Result<usize, String> {
            values
                .next()
                .ok_or(format!("expected a value after `{name}`"))?
                .parse()
                .map_err(|error| format!("invalid `{name}` value: {error}"))
        };
        match argument.as_str() {
            "-h" | "--help" => {
                println!("{}", usage());
                return Ok(None);
            }
            "--iterations" => arguments.iterations = value("--iterations")?,
            "--rounds" => arguments.rounds = value("--rounds")?,
            "--case" => arguments.case = Some(values.next().ok_or("expected a value after `--case`")?),
            "--smoke" => {
                arguments.iterations = 100;
                arguments.rounds = 1;
            }
            other => return Err(format!("unknown argument `{other}`\n\n{}", usage())),
        }
    }
    if arguments.iterations == 0 || arguments.rounds == 0 {
        return Err("`--iterations` and `--rounds` must both be greater than zero".into());
    }
    Ok(Some(arguments))
}

/// Runs `call` for 100 warm-up calls and then `rounds` timed rounds of `iterations` calls, returning the per-call mean
/// of each round in nanoseconds.
fn measure(arguments: &Arguments, mut call: impl FnMut()) -> Vec<f64> {
    for _ in 0..100 {
        call();
    }
    (0..arguments.rounds)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..arguments.iterations {
                call();
            }
            start.elapsed().as_nanos() as f64 / arguments.iterations as f64
        })
        .collect()
}

/// StableHLO module that adds its two `f32[8]` inputs, used to measure the raw PJRT execution floor.
const ADD_MODULE: &str = "module {
  func.func @main(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(arguments) = parse_arguments().map_err(std::io::Error::other)? else {
        return Ok(());
    };

    let plugin = load_cpu_plugin()?;
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))?;
    let device = Device::from_pjrt(&client.addressable_devices()?.remove(0))?;
    let mesh = DeviceMesh::new(LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto)?])?, vec![device])?;
    let domain = XlaSession::new(&client).domain();
    let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
        .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))?;
    let bytes = (0..8).flat_map(|index| (index as f32).to_ne_bytes()).collect::<Vec<_>>();
    let input = XlaArray::from_host_buffer(&domain, input_type.clone(), mesh.clone(), bytes.as_slice())?;
    input.block_until_ready()?;

    let mut results = HashMap::new();

    // Eager `add` through the receiver-based capability, which dispatches through the array's domain.
    if arguments.selects("eager_add") {
        results.insert("eager_add", measure(&arguments, || drop(Add::add(&input, &input).unwrap())));
    }

    // First-class dimension addition through a session-carrying `XlaDimension`, which binds the operation through its
    // domain, and through a session-free `DimensionValue`, which adds the extents directly on the host.
    let extent = DimensionValue::constant(8)?;
    let xla_extent = XlaDimension::new(extent.clone(), domain.clone());
    if arguments.selects("xla_dimension_add") {
        results.insert("xla_dimension_add", measure(&arguments, || drop(Add::add(&xla_extent, &xla_extent).unwrap())));
    }
    if arguments.selects("host_dimension_add") {
        results.insert("host_dimension_add", measure(&arguments, || drop(Add::add(&extent, &extent).unwrap())));
    }

    // Reference reads through a session-carrying `XlaValue`, which binds the read through its domain, and through a
    // session-free `ArrayIrValue`, which reads the handle directly.
    let xla_reference = XlaValue::Array(input.clone()).reference_new()?;
    if arguments.selects("xla_reference_read") {
        results.insert("xla_reference_read", measure(&arguments, || drop(xla_reference.read().unwrap())));
    }
    let host_reference = ArrayIrValue::Array(input.clone()).reference_new()?;
    if arguments.selects("host_reference_read") {
        results.insert("host_reference_read", measure(&arguments, || drop(host_reference.read().unwrap())));
    }

    // Eager `condition` whose branch programs are rebuilt for every application, like closures traced per call.
    let branch_type = ArrayIrType::Array(input_type.clone());
    let branch = |operation: XlaOperation| {
        let mut builder = XlaProgramBuilder::new();
        let value = builder.add_input(branch_type.clone());
        let output = builder.add_instruction(operation, Vec::new(), vec![value, value], None).unwrap()[0];
        builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
    };
    let predicate_type = ArrayType::scalar(DataType::Boolean).replicated(&mesh)?;
    let predicate = XlaArray::from_host_buffer(&domain, predicate_type, mesh.clone(), [1u8])?;
    let condition_inputs = [XlaValue::Array(predicate), XlaValue::Array(input.clone())];
    if arguments.selects("eager_condition") {
        results.insert(
            "eager_condition",
            measure(&arguments, || {
                let branches = [
                    branch(XlaOperation::Array(AddOperation::new().into())),
                    branch(XlaOperation::Array(MulOperation::new().into())),
                ];
                let operation = XlaOperation::Condition(ConditionOperation::new());
                drop(domain.bind(operation, branches, &condition_inputs).unwrap());
            }),
        );
    }

    // Warm jitted call of `x + x`.
    let function: JittedXlaFunction<'_, _, (), ArrayType, ArrayType> =
        jitted(|_, value: XlaCompileTracer<'_>| Add::add(&value, &value).unwrap(), &domain, mesh.clone());
    if arguments.selects("jit_add") {
        results.insert("jit_add", measure(&arguments, || drop(function.call((), input.clone()).unwrap())));
    }

    // Raw PJRT execution floor: a precompiled add executed directly through `ryft-pjrt`.
    let options = CompilationOptions {
        executable_build_options: Some(ExecutableCompilationOptions {
            device_ordinal: -1,
            replica_count: 1,
            partition_count: 1,
            ..Default::default()
        }),
        matrix_unit_operand_precision: Precision::Default as i32,
        ..Default::default()
    };
    let executable = client.compile(&Program::Mlir { bytecode: ADD_MODULE.as_bytes().to_vec() }, &options)?;
    let buffer = Arc::clone(input.addressable_shards().next().unwrap().buffer().unwrap());
    let execution_inputs =
        [ExecutionInput { buffer: Arc::clone(&buffer), donatable: false }, ExecutionInput { buffer, donatable: false }];
    if arguments.selects("pjrt_add") {
        results.insert(
            "pjrt_add",
            measure(&arguments, || {
                let inputs = vec![ExecutionDeviceInputs::from(&execution_inputs[..])];
                drop(executable.execute(inputs, Vec::new(), 0, None, None, None, None).unwrap());
            }),
        );
    }
    input.block_until_ready()?;

    let report = results
        .into_iter()
        .map(|(case, rounds)| {
            let mut sorted = rounds.clone();
            sorted.sort_by(f64::total_cmp);
            let median = sorted[sorted.len() / 2];
            (case.to_string(), json!({ "round_means_ns": rounds, "minimum_ns": sorted[0], "median_ns": median }))
        })
        .collect::<serde_json::Map<_, _>>();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "iterations": arguments.iterations,
            "rounds": arguments.rounds,
            "platform": client.platform_name()?.into_owned(),
            "cases": report,
        }))?,
    );
    Ok(())
}
