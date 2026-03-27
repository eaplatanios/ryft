//! CLI that emits Rust-side IR benchmark artifacts as JSON.

use std::env;

use ryft_core::tracing_v2::benchmarking::{benchmark_case_ids, collect_ir_benchmark_records};

/// Runs the Rust-side IR benchmark emitter.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut case_ids = Vec::new();
    let mut list_cases = false;

    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--list" => list_cases = true,
            "--case" => {
                let case_id = arguments.next().ok_or("expected a case ID after --case")?;
                case_ids.push(case_id);
            }
            other => {
                return Err(format!("unknown argument '{other}'").into());
            }
        }
    }

    if list_cases {
        println!("{}", serde_json::to_string_pretty(&benchmark_case_ids())?);
        return Ok(());
    }

    let records = collect_ir_benchmark_records(case_ids.as_slice())?;
    println!("{}", serde_json::to_string_pretty(&records)?);
    Ok(())
}
