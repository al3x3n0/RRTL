use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

use rrtl_gpu_sim::GpuMemoryLayout;
use rrtl_pyrtl::{
    bench_backends_trace, bench_gpu_combined, bench_gpu_options, bench_gpu_trace,
    bench_packed_trace, bench_single_trace, bench_threaded_trace, bench_trace, compare_trace,
    emit_compiled_json, emit_systemverilog, import_export, plan_backends, profile_replay,
    read_export, read_lane_trace, read_trace, validate_runtime_profile, BenchBackendsTraceOptions,
    BenchGpuTraceOptions, BenchPackedTraceOptions, BenchSingleTraceOptions,
    BenchThreadedTraceOptions, BenchTraceOptions, PlanBackendsOptions, PlannerCalibration,
    ProfileReplayOptions, PyrtlBenchBackendKind, RuntimeProfile,
};
use rrtl_runtime::{
    RuntimeSurrogateAttachment, RuntimeSurrogateExecutionReport, RuntimeTopology, RuntimeWorker,
};
use rrtl_sim_ir::{SimBackendKind, ThreadedReplayWorkerOptions};
use rrtl_surrogate::{
    emit_event_corpus, emit_instrumented_event_corpus, export_dataset, infer_model_fast_op_kind,
    inspect_event_corpus, inspect_rrtl_instrumentation_trace, plan_runtime_events,
    plan_runtime_events_for_workers, plan_runtime_gemm, plan_runtime_gemm_for_workers,
    policy_event_corpus, policy_gemm_batch, read_event_corpus, read_event_emitter_config,
    read_event_policy_report, read_event_runtime_plan, read_gemm_batch, read_gemm_policy_report,
    read_gemm_transaction, read_manifest, read_model_fast_plan, read_rrtl_instrumentation_trace,
    render_validation_markdown, run_fast_event_corpus_with_options,
    run_fast_gemm_batch_with_options, run_gemm_batch, run_gemm_transaction, run_model_fast_plan,
    shadow_event_corpus, validate_event_corpus, validate_manifest_path, validate_surrogate,
    EventFastRunOptions, EventRuntimePlan, GemmFastRunOptions, GemmRuntimePlan,
    GemmRuntimeWorkerSpec, ModelFastOp, ModelFastPlan, ModelFastThresholds, MODEL_FAST_PLAN_SCHEMA,
};

fn main() {
    if let Err(err) = run() {
        eprintln!("pyrtl2rrtl: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() || args.iter().any(|arg| arg == "-h" || arg == "--help") {
        usage(&mut io::stdout())?;
        return if args.is_empty() {
            Ok(())
        } else {
            Err("invalid arguments".into())
        };
    }

    let command = args.remove(0);
    if command == "surrogate" {
        return run_surrogate(args);
    }
    if args.is_empty() {
        usage(&mut io::stderr())?;
        return Err(format!("command `{command}` requires an export path").into());
    }
    let path = args.remove(0);
    let export = read_export(File::open(path)?)?;
    match command.as_str() {
        "check" => {
            import_export(&export)?;
            println!("ok");
        }
        "sv" => {
            print!("{}", emit_systemverilog(&export)?);
        }
        "json" => {
            println!("{}", emit_compiled_json(&export)?);
        }
        "compare" => {
            let trace_path = args.first().ok_or("compare requires a trace JSON path")?;
            let trace = read_trace(File::open(trace_path)?)?;
            let mismatches = compare_trace(&export, &trace)?;
            if mismatches.is_empty() {
                println!("ok");
            } else {
                for mismatch in &mismatches {
                    eprintln!(
                        "step {} `{}`: expected {}, actual {}",
                        mismatch.step, mismatch.signal, mismatch.expected, mismatch.actual
                    );
                }
                return Err(format!("{} trace mismatches", mismatches.len()).into());
            }
        }
        "bench-trace" => {
            let trace_path = args
                .first()
                .ok_or("bench-trace requires a trace JSON path")?
                .clone();
            let options = parse_bench_trace_options(&args[1..])?;
            let trace = read_trace(File::open(trace_path)?)?;
            let report = bench_trace(&export, &trace, options)?;
            if report.mismatch_count != 0 {
                for mismatch in &report.mismatches {
                    eprintln!(
                        "step {} `{}`: expected {}, actual {}",
                        mismatch.step, mismatch.signal, mismatch.expected, mismatch.actual
                    );
                }
                return Err(format!("{} trace mismatches", report.mismatch_count).into());
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        "bench-packed-trace" => {
            let trace_path = args
                .first()
                .ok_or("bench-packed-trace requires a trace JSON path")?
                .clone();
            let options = parse_bench_packed_trace_options(&args[1..])?;
            let trace = read_trace(File::open(trace_path)?)?;
            let report = bench_packed_trace(&export, &trace, options)?;
            if report.mismatch_count != 0 {
                for mismatch in &report.mismatches {
                    let lane = mismatch
                        .lane
                        .map(|lane| format!(" lane {lane}"))
                        .unwrap_or_default();
                    eprintln!(
                        "step {}{} `{}`: expected {}, actual {}",
                        mismatch.step, lane, mismatch.signal, mismatch.expected, mismatch.actual
                    );
                }
                return Err(format!("{} trace mismatches", report.mismatch_count).into());
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        "bench-single-trace" => {
            let trace_path = args
                .first()
                .ok_or("bench-single-trace requires a trace JSON path")?
                .clone();
            let options = parse_bench_single_trace_options(&args[1..])?;
            let trace = read_trace(File::open(trace_path)?)?;
            let report = bench_single_trace(&export, &trace, options)?;
            if report.mismatch_count != 0 {
                for mismatch in &report.mismatches {
                    eprintln!(
                        "step {} `{}`: expected {}, actual {}",
                        mismatch.step, mismatch.signal, mismatch.expected, mismatch.actual
                    );
                }
                return Err(format!("{} trace mismatches", report.mismatch_count).into());
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        "bench-backends" => {
            let trace_path = args
                .first()
                .ok_or("bench-backends requires a trace JSON path")?
                .clone();
            let options = parse_bench_backends_trace_options(&args[1..])?;
            let trace = read_trace(File::open(trace_path)?)?;
            let report = bench_backends_trace(&export, &trace, options)?;
            for backend in &report.backends {
                if backend.mismatch_count != 0 {
                    for mismatch in &backend.mismatches {
                        let lane = mismatch
                            .lane
                            .map(|lane| format!(" lane {lane}"))
                            .unwrap_or_default();
                        eprintln!(
                            "{} step {}{} `{}`: expected {}, actual {}",
                            backend.backend,
                            mismatch.step,
                            lane,
                            mismatch.signal,
                            mismatch.expected,
                            mismatch.actual
                        );
                    }
                    return Err(format!(
                        "{} trace mismatches in backend `{}`",
                        backend.mismatch_count, backend.backend
                    )
                    .into());
                }
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        "bench-threaded-trace" => {
            let trace_path = args
                .first()
                .ok_or("bench-threaded-trace requires a lane trace JSON path")?
                .clone();
            let options = parse_bench_threaded_trace_options(&args[1..])?;
            let trace = read_lane_trace(File::open(trace_path)?)?;
            let report = bench_threaded_trace(&export, &trace, options)?;
            if report.mismatch_count != 0 {
                for mismatch in &report.mismatches {
                    let lane = mismatch
                        .lane
                        .map(|lane| format!(" lane {lane}"))
                        .unwrap_or_default();
                    eprintln!(
                        "step {}{} `{}`: expected {}, actual {}",
                        mismatch.step, lane, mismatch.signal, mismatch.expected, mismatch.actual
                    );
                }
                return Err(format!("{} trace mismatches", report.mismatch_count).into());
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        "plan-backends" => {
            let trace_path = args
                .first()
                .ok_or("plan-backends requires a lane trace JSON path")?
                .clone();
            let options = parse_plan_backends_options(&args[1..])?;
            let trace = read_lane_trace(File::open(trace_path)?)?;
            let report = plan_backends(&export, &trace, options)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        "bench-gpu-trace" => {
            let trace_path = args
                .first()
                .ok_or("bench-gpu-trace requires a lane trace JSON path")?
                .clone();
            let options = parse_bench_gpu_trace_options(&args[1..])?;
            let trace = read_lane_trace(File::open(trace_path)?)?;
            let report = bench_gpu_trace(&export, &trace, options)?;
            if report.mismatch_count != 0 {
                for mismatch in &report.mismatches {
                    let lane = mismatch
                        .lane
                        .map(|lane| format!(" lane {lane}"))
                        .unwrap_or_default();
                    eprintln!(
                        "step {}{} `{}`: expected {}, actual {}",
                        mismatch.step, lane, mismatch.signal, mismatch.expected, mismatch.actual
                    );
                }
                return Err(format!("{} trace mismatches", report.mismatch_count).into());
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        "bench-gpu-options" => {
            let trace_path = args
                .first()
                .ok_or("bench-gpu-options requires a lane trace JSON path")?
                .clone();
            let options = parse_bench_gpu_trace_options(&args[1..])?;
            let trace = read_lane_trace(File::open(trace_path)?)?;
            let report = bench_gpu_options(&export, &trace, options)?;
            if report
                .candidates
                .iter()
                .any(|candidate| candidate.mismatch_count != 0)
            {
                return Err("GPU option sweep trace mismatches".into());
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        "bench-gpu-combined" => {
            let trace_path = args
                .first()
                .ok_or("bench-gpu-combined requires a lane trace JSON path")?
                .clone();
            let options = parse_bench_gpu_trace_options(&args[1..])?;
            let trace = read_lane_trace(File::open(trace_path)?)?;
            let report = bench_gpu_combined(&export, &trace, options)?;
            if report.static_trace.mismatch_count != 0
                || report.measured_trace.mismatch_count != 0
                || report
                    .option_sweep
                    .candidates
                    .iter()
                    .any(|candidate| candidate.mismatch_count != 0)
            {
                return Err("GPU combined trace mismatches".into());
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        "bench-profile-replay" => {
            if args.len() < 3 {
                return Err(
                    "bench-profile-replay requires <trace.json> <lane-trace.json> <runtime_profile.json>"
                        .into(),
                );
            }
            let trace_path = args.remove(0);
            let lane_trace_path = args.remove(0);
            let profile_path = args.remove(0);
            let options = parse_bench_profile_replay_options(&args)?;
            let profile = read_runtime_profile(File::open(profile_path)?)?;
            let trace = read_trace(File::open(trace_path)?)?;
            let lane_trace = read_lane_trace(File::open(lane_trace_path)?)?;
            let report = profile_replay(&export, &trace, &lane_trace, &profile, options)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        other => {
            usage(&mut io::stderr())?;
            return Err(format!("unknown command `{other}`").into());
        }
    }
    Ok(())
}

fn read_runtime_profile(
    reader: impl io::Read,
) -> Result<RuntimeProfile, Box<dyn std::error::Error>> {
    let profile: RuntimeProfile = serde_json::from_reader(reader)?;
    validate_runtime_profile(&profile)?;
    Ok(profile)
}

#[cfg(test)]
fn runtime_profile_workers(
    workers: &[rrtl_pyrtl::RuntimeProfileWorker],
) -> Result<Vec<ThreadedReplayWorkerOptions>, Box<dyn std::error::Error>> {
    workers
        .iter()
        .map(|worker| parse_threaded_worker(&format!("{}:{}", worker.backend, worker.lanes)))
        .collect()
}

fn run_surrogate(mut args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    if args.is_empty() {
        surrogate_usage(&mut io::stderr())?;
        return Err("surrogate requires a subcommand".into());
    }
    let command = args.remove(0);
    match command.as_str() {
        "validate-manifest" => {
            if args.len() != 1 {
                surrogate_usage(&mut io::stderr())?;
                return Err("validate-manifest requires <manifest.json>".into());
            }
            let manifest_path = &args[0];
            let manifest = read_manifest(File::open(manifest_path)?)?;
            let report = validate_manifest_path(&manifest, manifest_path);
            println!("{}", serde_json::to_string_pretty(&report)?);
            if !report.ok {
                return Err("surrogate manifest validation failed".into());
            }
        }
        "export-dataset" => {
            if args.len() < 2 {
                surrogate_usage(&mut io::stderr())?;
                return Err(
                    "export-dataset requires <export.pyrtl.json> <trace.json> [--out dataset.json]"
                        .into(),
                );
            }
            let export_path = args.remove(0);
            let trace_path = args.remove(0);
            let out = parse_optional_out(&args)?;
            let export = read_json_value(&export_path)?;
            let trace = read_json_value(&trace_path)?;
            let dataset = export_dataset(&export, &trace)?;
            let text = serde_json::to_string_pretty(&dataset)? + "\n";
            if let Some(out) = out {
                fs::write(out, text)?;
            } else {
                print!("{text}");
            }
        }
        "validate" => {
            if args.len() < 3 {
                surrogate_usage(&mut io::stderr())?;
                return Err(
                    "validate requires <manifest.json> <export.pyrtl.json> <trace.json>".into(),
                );
            }
            let manifest_path = args.remove(0);
            let export_path = args.remove(0);
            let trace_path = args.remove(0);
            let outputs = parse_summary_outputs(&args)?;
            let manifest = read_manifest(File::open(&manifest_path)?)?;
            let export = read_json_value(&export_path)?;
            let trace = read_json_value(&trace_path)?;
            let report = validate_surrogate(&manifest, &manifest_path, &export, &trace)?;
            let json = serde_json::to_string_pretty(&report)? + "\n";
            if let Some(path) = outputs.summary_json {
                fs::write(path, &json)?;
            } else {
                print!("{json}");
            }
            if let Some(path) = outputs.summary_md {
                fs::write(path, render_validation_markdown(&report))?;
            }
            if !report.ok {
                return Err("surrogate validation failed".into());
            }
        }
        "inspect-events" => {
            if args.is_empty() {
                surrogate_usage(&mut io::stderr())?;
                return Err("inspect-events requires <events.json> [--out report.json]".into());
            }
            let events_path = args.remove(0);
            let out = parse_optional_out(&args)?;
            let events = read_event_corpus(File::open(events_path)?)?;
            let report = inspect_event_corpus(&events);
            let text = serde_json::to_string_pretty(&report)? + "\n";
            if let Some(out) = out {
                fs::write(out, text)?;
            } else {
                print!("{text}");
            }
            if !report.ok {
                return Err("surrogate event inspection failed".into());
            }
        }
        "inspect-instrumentation" => {
            if args.is_empty() {
                surrogate_usage(&mut io::stderr())?;
                return Err(
                    "inspect-instrumentation requires <instrumentation.json> [--config config.json] [--out report.json]"
                        .into(),
                );
            }
            let trace_path = args.remove(0);
            let options = parse_inspect_instrumentation_options(&args)?;
            let trace = read_rrtl_instrumentation_trace(File::open(&trace_path)?)?;
            let config = options
                .config
                .as_ref()
                .map(|path| -> Result<_, Box<dyn std::error::Error>> {
                    Ok(read_event_emitter_config(File::open(path)?)?)
                })
                .transpose()?;
            let report = inspect_rrtl_instrumentation_trace(&trace, config.as_ref());
            let text = serde_json::to_string_pretty(&report)? + "\n";
            if let Some(out) = options.out {
                fs::write(out, text)?;
            } else {
                print!("{text}");
            }
            if !report.ok {
                return Err("RRTL instrumentation inspection failed".into());
            }
        }
        "emit-events" => {
            if args.is_empty() {
                surrogate_usage(&mut io::stderr())?;
                return Err(
                    "emit-events requires <trace.json> --config config.json [--out events.json]"
                        .into(),
                );
            }
            let trace_path = args.remove(0);
            let options = parse_emit_events_options(&args)?;
            let trace = read_json_value(&trace_path)?;
            let config = read_event_emitter_config(File::open(options.config)?)?;
            let corpus = emit_event_corpus(&trace, &config)?;
            let text = serde_json::to_string_pretty(&corpus)? + "\n";
            if let Some(out) = options.out {
                fs::write(out, text)?;
            } else {
                print!("{text}");
            }
        }
        "emit-instrumented-events" => {
            if args.is_empty() {
                surrogate_usage(&mut io::stderr())?;
                return Err(
                    "emit-instrumented-events requires <instrumentation.json> --config config.json [--out events.json]"
                        .into(),
                );
            }
            let trace_path = args.remove(0);
            let options = parse_emit_events_options(&args)?;
            let trace = read_rrtl_instrumentation_trace(File::open(&trace_path)?)?;
            let config = read_event_emitter_config(File::open(options.config)?)?;
            let corpus = emit_instrumented_event_corpus(&trace, &config)?;
            let text = serde_json::to_string_pretty(&corpus)? + "\n";
            if let Some(out) = options.out {
                fs::write(out, text)?;
            } else {
                print!("{text}");
            }
        }
        "validate-events" => {
            if args.len() < 2 {
                surrogate_usage(&mut io::stderr())?;
                return Err(
                    "validate-events requires <manifest.json> <events.json> [--out report.json]"
                        .into(),
                );
            }
            let manifest_path = args.remove(0);
            let events_path = args.remove(0);
            let out = parse_optional_out(&args)?;
            let manifest = read_manifest(File::open(&manifest_path)?)?;
            let events = read_event_corpus(File::open(events_path)?)?;
            let report = validate_event_corpus(&manifest, &manifest_path, &events);
            let text = serde_json::to_string_pretty(&report)? + "\n";
            if let Some(out) = out {
                fs::write(out, text)?;
            } else {
                print!("{text}");
            }
            if !report.ok {
                return Err("surrogate event validation failed".into());
            }
        }
        "shadow-events" => {
            if args.len() < 2 {
                surrogate_usage(&mut io::stderr())?;
                return Err(
                    "shadow-events requires <manifest.json> <events.json> [--out report.json]"
                        .into(),
                );
            }
            let manifest_path = args.remove(0);
            let events_path = args.remove(0);
            let out = parse_optional_out(&args)?;
            let manifest = read_manifest(File::open(&manifest_path)?)?;
            let events = read_event_corpus(File::open(events_path)?)?;
            let report = shadow_event_corpus(&manifest, &manifest_path, &events);
            let text = serde_json::to_string_pretty(&report)? + "\n";
            if let Some(out) = out {
                fs::write(out, text)?;
            } else {
                print!("{text}");
            }
            if !report.ok {
                return Err("surrogate event shadow comparison failed".into());
            }
        }
        "policy-events" => {
            if args.len() < 2 {
                surrogate_usage(&mut io::stderr())?;
                return Err(
                    "policy-events requires <manifest.json> <events.json> [--out report.json]"
                        .into(),
                );
            }
            let manifest_path = args.remove(0);
            let events_path = args.remove(0);
            let out = parse_optional_out(&args)?;
            let manifest = read_manifest(File::open(&manifest_path)?)?;
            let events = read_event_corpus(File::open(events_path)?)?;
            let report = policy_event_corpus(&manifest, &manifest_path, &events);
            let text = serde_json::to_string_pretty(&report)? + "\n";
            if let Some(out) = out {
                fs::write(out, text)?;
            } else {
                print!("{text}");
            }
            if !report.ok {
                return Err("surrogate event policy evaluation failed".into());
            }
        }
        "run-gemm" => {
            if args.len() < 2 {
                surrogate_usage(&mut io::stderr())?;
                return Err(
                    "run-gemm requires <manifest.json> <transaction.json> [--out result.json]"
                        .into(),
                );
            }
            let manifest_path = args.remove(0);
            let transaction_path = args.remove(0);
            let out = parse_optional_out(&args)?;
            let manifest = read_manifest(File::open(&manifest_path)?)?;
            let transaction = read_gemm_transaction(File::open(transaction_path)?)?;
            let result = run_gemm_transaction(&manifest, &manifest_path, &transaction)?;
            let text = serde_json::to_string_pretty(&result)? + "\n";
            if let Some(out) = out {
                fs::write(out, text)?;
            } else {
                print!("{text}");
            }
            if !result.ok {
                return Err("surrogate GEMM validation failed".into());
            }
        }
        "run-gemm-batch" => {
            if args.len() < 2 {
                surrogate_usage(&mut io::stderr())?;
                return Err(
                    "run-gemm-batch requires <manifest.json> <batch.json> [--out result.json]"
                        .into(),
                );
            }
            let manifest_path = args.remove(0);
            let batch_path = args.remove(0);
            let out = parse_optional_out(&args)?;
            let manifest = read_manifest(File::open(&manifest_path)?)?;
            let batch = read_gemm_batch(File::open(batch_path)?)?;
            let report = run_gemm_batch(&manifest, &manifest_path, &batch);
            let text = serde_json::to_string_pretty(&report)? + "\n";
            if let Some(out) = out {
                fs::write(out, text)?;
            } else {
                print!("{text}");
            }
            if !report.ok {
                return Err("surrogate GEMM batch validation failed".into());
            }
        }
        "policy-gemm-batch" => {
            if args.len() < 2 {
                surrogate_usage(&mut io::stderr())?;
                return Err(
                    "policy-gemm-batch requires <manifest.json> <batch.json> [--out report.json]"
                        .into(),
                );
            }
            let manifest_path = args.remove(0);
            let batch_path = args.remove(0);
            let out = parse_optional_out(&args)?;
            let manifest = read_manifest(File::open(&manifest_path)?)?;
            let batch = read_gemm_batch(File::open(batch_path)?)?;
            let report = policy_gemm_batch(&manifest, &manifest_path, &batch);
            let text = serde_json::to_string_pretty(&report)? + "\n";
            if let Some(out) = out {
                fs::write(out, text)?;
            } else {
                print!("{text}");
            }
            if !report.ok {
                return Err("surrogate GEMM policy evaluation failed".into());
            }
        }
        "run-fast-gemm" => {
            if args.len() < 3 {
                surrogate_usage(&mut io::stderr())?;
                return Err(
                    "run-fast-gemm requires <manifest.json> <batch.json> <runtime-plan.json> [--out report.json]"
                        .into(),
                );
            }
            let manifest_path = args.remove(0);
            let batch_path = args.remove(0);
            let plan_path = args.remove(0);
            let options = parse_run_fast_gemm_options(&args)?;
            let manifest = read_manifest(File::open(&manifest_path)?)?;
            let batch = read_gemm_batch(File::open(batch_path)?)?;
            let plan: GemmRuntimePlan = serde_json::from_reader(File::open(plan_path)?)?;
            let report = run_fast_gemm_batch_with_options(
                &manifest,
                &manifest_path,
                &batch,
                &plan,
                options.fast,
            );
            let text = serde_json::to_string_pretty(&report)? + "\n";
            if let Some(out) = options.out {
                fs::write(out, text)?;
            } else {
                print!("{text}");
            }
            if !report.ok {
                return Err("surrogate GEMM FAST run failed".into());
            }
        }
        "run-fast-events" => {
            if args.len() < 3 {
                surrogate_usage(&mut io::stderr())?;
                return Err(
                    "run-fast-events requires <manifest.json> <events.json> <runtime-plan.json> [--out report.json]"
                        .into(),
                );
            }
            let manifest_path = args.remove(0);
            let events_path = args.remove(0);
            let plan_path = args.remove(0);
            let options = parse_run_fast_event_options(&args)?;
            let manifest = read_manifest(File::open(&manifest_path)?)?;
            let events = read_event_corpus(File::open(events_path)?)?;
            let plan: EventRuntimePlan = read_event_runtime_plan(File::open(plan_path)?)?;
            let report = run_fast_event_corpus_with_options(
                &manifest,
                &manifest_path,
                &events,
                &plan,
                options.fast,
            );
            let text = serde_json::to_string_pretty(&report)? + "\n";
            if let Some(out) = options.out {
                fs::write(out, text)?;
            } else {
                print!("{text}");
            }
            if !report.ok {
                return Err("surrogate event FAST run failed".into());
            }
        }
        "plan-model-fast" => {
            let options = parse_plan_model_fast_options(&args)?;
            let mut seen = BTreeSet::new();
            let mut ops = Vec::with_capacity(options.ops.len());
            let golden_paths = parse_model_fast_golden_bindings(&options.golden)?;
            let timings = parse_model_fast_timing_bindings(&options.timing)?;
            for spec in options.ops {
                let op = parse_model_fast_op_spec(&spec)?;
                if !seen.insert(op.op_id.clone()) {
                    return Err(format!("duplicate model FAST op id `{}`", op.op_id).into());
                }
                let op_kind = infer_model_fast_op_kind(File::open(&op.fast_report_path)?)?;
                let golden_path = golden_paths.get(&op.op_id).cloned();
                let timing = timings.get(&op.op_id).copied();
                ops.push(ModelFastOp {
                    op_id: op.op_id.clone(),
                    op_kind,
                    name: op.name,
                    fast_report_path: op.fast_report_path,
                    golden_path,
                    exact_ns: timing.map(|timing| timing.exact_ns),
                    fast_ns: timing.map(|timing| timing.fast_ns),
                    source_hash: None,
                    description: None,
                });
            }
            for op_id in golden_paths.keys() {
                if !seen.contains(op_id) {
                    return Err(format!("model FAST golden references unknown op `{op_id}`").into());
                }
            }
            for op_id in timings.keys() {
                if !seen.contains(op_id) {
                    return Err(format!("model FAST timing references unknown op `{op_id}`").into());
                }
            }
            for path in golden_paths.values() {
                File::open(path)
                    .map_err(|err| format!("failed to open model FAST golden `{path}`: {err}"))?;
            }
            let thresholds = match options.thresholds {
                Some(path) => Some(serde_json::from_reader::<_, ModelFastThresholds>(
                    File::open(path)?,
                )?),
                None => None,
            };
            let plan = ModelFastPlan {
                schema: MODEL_FAST_PLAN_SCHEMA.to_string(),
                ops,
                thresholds,
            };
            let text = serde_json::to_string_pretty(&plan)? + "\n";
            if let Some(out) = options.out {
                fs::write(out, text)?;
            } else {
                print!("{text}");
            }
        }
        "run-model-fast" => {
            if args.is_empty() {
                surrogate_usage(&mut io::stderr())?;
                return Err(
                    "run-model-fast requires <model-fast-plan.json> [--out report.json]".into(),
                );
            }
            let plan_path = args.remove(0);
            let out = parse_optional_out(&args)?;
            let plan = read_model_fast_plan(File::open(&plan_path)?)?;
            let base_dir = Path::new(&plan_path).parent().unwrap_or(Path::new("."));
            let report = run_model_fast_plan(&plan, base_dir);
            let text = serde_json::to_string_pretty(&report)? + "\n";
            if let Some(out) = out {
                fs::write(out, text)?;
            } else {
                print!("{text}");
            }
            if !report.ok {
                return Err("surrogate model FAST run failed".into());
            }
        }
        "plan-runtime-gemm" => {
            if args.is_empty() {
                surrogate_usage(&mut io::stderr())?;
                return Err(
                    "plan-runtime-gemm requires <policy-report.json> (--topology lanes:N|--worker id:lanes...) [--out plan.json]"
                        .into(),
                );
            }
            let report_path = args.remove(0);
            let options = parse_plan_runtime_gemm_options(&args)?;
            let policy = read_gemm_policy_report(File::open(report_path)?)?;
            let plan = match options.topology {
                PlanRuntimeGemmTopology::Lanes(total_lanes) => {
                    plan_runtime_gemm(&policy, total_lanes)
                }
                PlanRuntimeGemmTopology::Workers(workers) => {
                    plan_runtime_gemm_for_workers(&policy, &workers)
                }
            };
            let text = serde_json::to_string_pretty(&plan)? + "\n";
            if let Some(out) = options.out {
                fs::write(out, text)?;
            } else {
                print!("{text}");
            }
            if !plan.ok {
                return Err("surrogate GEMM runtime plan failed".into());
            }
        }
        "plan-runtime-events" => {
            if args.is_empty() {
                surrogate_usage(&mut io::stderr())?;
                return Err(
                    "plan-runtime-events requires <policy-report.json> (--topology lanes:N|--worker id:lanes...) [--out plan.json]"
                        .into(),
                );
            }
            let report_path = args.remove(0);
            let options = parse_plan_runtime_gemm_options(&args)?;
            let policy = read_event_policy_report(File::open(report_path)?)?;
            let plan = match options.topology {
                PlanRuntimeGemmTopology::Lanes(total_lanes) => {
                    plan_runtime_events(&policy, total_lanes)
                }
                PlanRuntimeGemmTopology::Workers(workers) => {
                    plan_runtime_events_for_workers(&policy, &workers)
                }
            };
            let text = serde_json::to_string_pretty(&plan)? + "\n";
            if let Some(out) = options.out {
                fs::write(out, text)?;
            } else {
                print!("{text}");
            }
            if !plan.ok {
                return Err("surrogate event runtime plan failed".into());
            }
        }
        "attach-runtime-gemm" => {
            if args.is_empty() {
                surrogate_usage(&mut io::stderr())?;
                return Err(
                    "attach-runtime-gemm requires <runtime-plan.json> --worker id:lanes... [--out attachment.json]"
                        .into(),
                );
            }
            let plan_path = args.remove(0);
            let options = parse_attach_runtime_gemm_options(&args)?;
            let plan: GemmRuntimePlan = serde_json::from_reader(File::open(plan_path)?)?;
            let topology = runtime_topology_from_worker_specs(&options.workers);
            let attachment = topology.attach_gemm_runtime_plan(&plan)?;
            let text = serde_json::to_string_pretty(&attachment)? + "\n";
            if let Some(out) = options.out {
                fs::write(out, text)?;
            } else {
                print!("{text}");
            }
        }
        "attach-runtime-events" => {
            if args.is_empty() {
                surrogate_usage(&mut io::stderr())?;
                return Err(
                    "attach-runtime-events requires <runtime-plan.json> --worker id:lanes... [--out attachment.json]"
                        .into(),
                );
            }
            let plan_path = args.remove(0);
            let options = parse_attach_runtime_gemm_options(&args)?;
            let plan = read_event_runtime_plan(File::open(plan_path)?)?;
            let topology = runtime_topology_from_worker_specs(&options.workers);
            let attachment = topology.attach_event_runtime_plan(&plan)?;
            let text = serde_json::to_string_pretty(&attachment)? + "\n";
            if let Some(out) = options.out {
                fs::write(out, text)?;
            } else {
                print!("{text}");
            }
        }
        "inspect-runtime-attachment" => {
            if args.is_empty() {
                surrogate_usage(&mut io::stderr())?;
                return Err(
                    "inspect-runtime-attachment requires <attachment.json> [--out report.json]"
                        .into(),
                );
            }
            let attachment_path = args.remove(0);
            let out = parse_optional_out(&args)?;
            let attachment =
                RuntimeSurrogateAttachment::read_json(&mut File::open(attachment_path)?)?;
            let report = RuntimeSurrogateExecutionReport::inspect_attachment(&attachment);
            let text = serde_json::to_string_pretty(&report)? + "\n";
            if let Some(out) = out {
                fs::write(out, text)?;
            } else {
                print!("{text}");
            }
            if !report.ready {
                return Err("surrogate runtime attachment is not ready".into());
            }
        }
        other => {
            surrogate_usage(&mut io::stderr())?;
            return Err(format!("unknown surrogate command `{other}`").into());
        }
    }
    Ok(())
}

struct SummaryOutputs {
    summary_json: Option<String>,
    summary_md: Option<String>,
}

struct EmitEventsOptions {
    config: String,
    out: Option<String>,
}

struct InspectInstrumentationOptions {
    config: Option<String>,
    out: Option<String>,
}

struct PlanRuntimeGemmOptions {
    topology: PlanRuntimeGemmTopology,
    out: Option<String>,
}

enum PlanRuntimeGemmTopology {
    Lanes(usize),
    Workers(Vec<GemmRuntimeWorkerSpec>),
}

struct AttachRuntimeGemmOptions {
    workers: Vec<GemmRuntimeWorkerSpec>,
    out: Option<String>,
}

struct RunFastGemmOptions {
    fast: GemmFastRunOptions,
    out: Option<String>,
}

struct RunFastEventOptions {
    fast: EventFastRunOptions,
    out: Option<String>,
}

struct PlanModelFastOptions {
    ops: Vec<String>,
    golden: Vec<String>,
    timing: Vec<String>,
    thresholds: Option<String>,
    out: Option<String>,
}

#[derive(Clone, Copy)]
struct ParsedModelFastTiming {
    exact_ns: u64,
    fast_ns: u64,
}

struct ParsedModelFastOpSpec {
    op_id: String,
    fast_report_path: String,
    name: String,
}

fn parse_emit_events_options(
    args: &[String],
) -> Result<EmitEventsOptions, Box<dyn std::error::Error>> {
    let mut config = None;
    let mut out = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--config" => {
                index += 1;
                config = Some(args.get(index).ok_or("--config requires a value")?.clone());
            }
            "--out" => {
                index += 1;
                out = Some(args.get(index).ok_or("--out requires a value")?.clone());
            }
            other => return Err(format!("unknown surrogate emit-events argument `{other}`").into()),
        }
        index += 1;
    }
    Ok(EmitEventsOptions {
        config: config.ok_or("emit-events requires --config config.json")?,
        out,
    })
}

fn parse_inspect_instrumentation_options(
    args: &[String],
) -> Result<InspectInstrumentationOptions, Box<dyn std::error::Error>> {
    let mut config = None;
    let mut out = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--config" => {
                index += 1;
                config = Some(args.get(index).ok_or("--config requires a value")?.clone());
            }
            "--out" => {
                index += 1;
                out = Some(args.get(index).ok_or("--out requires a value")?.clone());
            }
            other => {
                return Err(
                    format!("unknown surrogate inspect-instrumentation argument `{other}`").into(),
                )
            }
        }
        index += 1;
    }
    Ok(InspectInstrumentationOptions { config, out })
}

fn parse_plan_runtime_gemm_options(
    args: &[String],
) -> Result<PlanRuntimeGemmOptions, Box<dyn std::error::Error>> {
    let mut total_lanes = None;
    let mut workers = Vec::new();
    let mut out = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--topology" => {
                index += 1;
                total_lanes = Some(parse_runtime_topology_lanes(
                    args.get(index).ok_or("--topology requires lanes:N")?,
                )?);
            }
            "--worker" => {
                index += 1;
                let value = args.get(index).ok_or("--worker requires id:lanes")?;
                let start_lane = workers
                    .last()
                    .map(|worker: &GemmRuntimeWorkerSpec| worker.start_lane + worker.lanes)
                    .unwrap_or(0);
                workers.push(parse_runtime_worker_spec(value, start_lane)?);
            }
            "--out" => {
                index += 1;
                out = Some(args.get(index).ok_or("--out requires a path")?.clone());
            }
            other => {
                return Err(
                    format!("unknown surrogate plan-runtime-gemm argument `{other}`").into(),
                )
            }
        }
        index += 1;
    }
    if total_lanes.is_some() && !workers.is_empty() {
        return Err("plan-runtime-gemm cannot mix --topology and --worker".into());
    }
    let topology = match (total_lanes, workers.is_empty()) {
        (Some(total_lanes), true) => PlanRuntimeGemmTopology::Lanes(total_lanes),
        (None, false) => PlanRuntimeGemmTopology::Workers(workers),
        (None, true) => {
            return Err("plan-runtime-gemm requires --topology lanes:N or --worker id:lanes".into())
        }
        (Some(_), false) => unreachable!(),
    };
    Ok(PlanRuntimeGemmOptions { topology, out })
}

fn parse_attach_runtime_gemm_options(
    args: &[String],
) -> Result<AttachRuntimeGemmOptions, Box<dyn std::error::Error>> {
    let mut workers = Vec::new();
    let mut out = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--worker" => {
                index += 1;
                let value = args.get(index).ok_or("--worker requires id:lanes")?;
                let start_lane = workers
                    .last()
                    .map(|worker: &GemmRuntimeWorkerSpec| worker.start_lane + worker.lanes)
                    .unwrap_or(0);
                workers.push(parse_runtime_worker_spec(value, start_lane)?);
            }
            "--out" => {
                index += 1;
                out = Some(args.get(index).ok_or("--out requires a path")?.clone());
            }
            other => {
                return Err(
                    format!("unknown surrogate attach-runtime-gemm argument `{other}`").into(),
                )
            }
        }
        index += 1;
    }
    if workers.is_empty() {
        return Err("attach-runtime-gemm requires at least one --worker id:lanes".into());
    }
    Ok(AttachRuntimeGemmOptions { workers, out })
}

fn parse_run_fast_gemm_options(
    args: &[String],
) -> Result<RunFastGemmOptions, Box<dyn std::error::Error>> {
    let mut fast = GemmFastRunOptions::default();
    let mut out = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--out" => {
                index += 1;
                out = Some(args.get(index).ok_or("--out requires a value")?.clone());
            }
            "--shadow-sample-stride" => {
                index += 1;
                let stride = args
                    .get(index)
                    .ok_or("--shadow-sample-stride requires a value")?
                    .parse::<usize>()?;
                if stride == 0 {
                    return Err("--shadow-sample-stride must be greater than zero".into());
                }
                fast.shadow_sample_stride = Some(stride);
            }
            "--shadow-sample-offset" => {
                index += 1;
                fast.shadow_sample_offset = args
                    .get(index)
                    .ok_or("--shadow-sample-offset requires a value")?
                    .parse::<usize>()?;
            }
            other => return Err(format!("unknown run-fast-gemm argument `{other}`").into()),
        }
        index += 1;
    }
    Ok(RunFastGemmOptions { fast, out })
}

fn parse_run_fast_event_options(
    args: &[String],
) -> Result<RunFastEventOptions, Box<dyn std::error::Error>> {
    let mut fast = EventFastRunOptions::default();
    let mut out = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--out" => {
                index += 1;
                out = Some(args.get(index).ok_or("--out requires a value")?.clone());
            }
            "--shadow-sample-stride" => {
                index += 1;
                let stride = args
                    .get(index)
                    .ok_or("--shadow-sample-stride requires a value")?
                    .parse::<usize>()?;
                if stride == 0 {
                    return Err("--shadow-sample-stride must be greater than zero".into());
                }
                fast.shadow_sample_stride = Some(stride);
            }
            "--shadow-sample-offset" => {
                index += 1;
                fast.shadow_sample_offset = args
                    .get(index)
                    .ok_or("--shadow-sample-offset requires a value")?
                    .parse::<usize>()?;
            }
            other => return Err(format!("unknown run-fast-events argument `{other}`").into()),
        }
        index += 1;
    }
    Ok(RunFastEventOptions { fast, out })
}

fn parse_plan_model_fast_options(
    args: &[String],
) -> Result<PlanModelFastOptions, Box<dyn std::error::Error>> {
    let mut ops = Vec::new();
    let mut golden = Vec::new();
    let mut timing = Vec::new();
    let mut thresholds = None;
    let mut out = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--op" => {
                index += 1;
                ops.push(args.get(index).ok_or("--op requires a value")?.clone());
            }
            "--golden" => {
                index += 1;
                golden.push(args.get(index).ok_or("--golden requires a value")?.clone());
            }
            "--timing" => {
                index += 1;
                timing.push(args.get(index).ok_or("--timing requires a value")?.clone());
            }
            "--thresholds" => {
                index += 1;
                thresholds = Some(
                    args.get(index)
                        .ok_or("--thresholds requires a value")?
                        .clone(),
                );
            }
            "--out" => {
                index += 1;
                out = Some(args.get(index).ok_or("--out requires a value")?.clone());
            }
            other => {
                return Err(format!("unknown surrogate plan-model-fast argument `{other}`").into())
            }
        }
        index += 1;
    }
    if ops.is_empty() {
        return Err("plan-model-fast requires at least one --op id:path[:name]".into());
    }
    Ok(PlanModelFastOptions {
        ops,
        golden,
        timing,
        thresholds,
        out,
    })
}

fn parse_model_fast_op_spec(
    spec: &str,
) -> Result<ParsedModelFastOpSpec, Box<dyn std::error::Error>> {
    let mut parts = spec.splitn(3, ':');
    let op_id = parts.next().ok_or("--op must use id:path[:name]")?.trim();
    let fast_report_path = parts.next().ok_or("--op must use id:path[:name]")?.trim();
    let name = parts.next().map(str::trim).unwrap_or(op_id);
    if op_id.is_empty() {
        return Err("--op id must not be empty".into());
    }
    if fast_report_path.is_empty() {
        return Err("--op path must not be empty".into());
    }
    if name.is_empty() {
        return Err("--op name must not be empty".into());
    }
    Ok(ParsedModelFastOpSpec {
        op_id: op_id.to_string(),
        fast_report_path: fast_report_path.to_string(),
        name: name.to_string(),
    })
}

fn parse_model_fast_golden_bindings(
    specs: &[String],
) -> Result<BTreeMap<String, String>, Box<dyn std::error::Error>> {
    let mut bindings = BTreeMap::new();
    for spec in specs {
        let (op_id, path) = spec.split_once(':').ok_or("--golden must use op_id:path")?;
        let op_id = op_id.trim();
        let path = path.trim();
        if op_id.is_empty() {
            return Err("--golden op_id must not be empty".into());
        }
        if path.is_empty() {
            return Err("--golden path must not be empty".into());
        }
        if bindings
            .insert(op_id.to_string(), path.to_string())
            .is_some()
        {
            return Err(format!("duplicate model FAST golden binding for op `{op_id}`").into());
        }
    }
    Ok(bindings)
}

fn parse_model_fast_timing_bindings(
    specs: &[String],
) -> Result<BTreeMap<String, ParsedModelFastTiming>, Box<dyn std::error::Error>> {
    let mut bindings = BTreeMap::new();
    for spec in specs {
        let mut parts = spec.splitn(3, ':');
        let op_id = parts
            .next()
            .ok_or("--timing must use op_id:exact_ns:fast_ns")?
            .trim();
        let exact_ns = parts
            .next()
            .ok_or("--timing must use op_id:exact_ns:fast_ns")?
            .trim()
            .parse::<u64>()?;
        let fast_ns = parts
            .next()
            .ok_or("--timing must use op_id:exact_ns:fast_ns")?
            .trim()
            .parse::<u64>()?;
        if op_id.is_empty() {
            return Err("--timing op_id must not be empty".into());
        }
        if exact_ns == 0 {
            return Err("--timing exact_ns must be greater than zero".into());
        }
        if fast_ns == 0 {
            return Err("--timing fast_ns must be greater than zero".into());
        }
        if bindings
            .insert(
                op_id.to_string(),
                ParsedModelFastTiming { exact_ns, fast_ns },
            )
            .is_some()
        {
            return Err(format!("duplicate model FAST timing binding for op `{op_id}`").into());
        }
    }
    Ok(bindings)
}

fn parse_runtime_topology_lanes(value: &str) -> Result<usize, Box<dyn std::error::Error>> {
    let lanes = value
        .strip_prefix("lanes:")
        .ok_or("--topology must use lanes:N")?
        .parse::<usize>()?;
    if lanes == 0 {
        return Err("--topology lanes must be greater than zero".into());
    }
    Ok(lanes)
}

fn parse_runtime_worker_spec(
    value: &str,
    start_lane: usize,
) -> Result<GemmRuntimeWorkerSpec, Box<dyn std::error::Error>> {
    let (worker_id, lanes) = value.split_once(':').ok_or("--worker must use id:lanes")?;
    if worker_id.trim().is_empty() {
        return Err("--worker id must not be empty".into());
    }
    let lanes = lanes.parse::<usize>()?;
    if lanes == 0 {
        return Err("--worker lanes must be greater than zero".into());
    }
    Ok(GemmRuntimeWorkerSpec {
        worker_id: worker_id.to_string(),
        start_lane,
        lanes,
    })
}

fn runtime_topology_from_worker_specs(workers: &[GemmRuntimeWorkerSpec]) -> RuntimeTopology {
    let mut topology = RuntimeTopology::new();
    for worker in workers {
        topology.push(RuntimeWorker::local_cpu(
            worker.worker_id.clone(),
            worker.lanes,
        ));
    }
    topology
}

fn parse_summary_outputs(args: &[String]) -> Result<SummaryOutputs, Box<dyn std::error::Error>> {
    let mut summary_json = None;
    let mut summary_md = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--summary-json" => {
                index += 1;
                summary_json = Some(
                    args.get(index)
                        .ok_or("--summary-json requires a value")?
                        .clone(),
                );
            }
            "--summary-md" => {
                index += 1;
                summary_md = Some(
                    args.get(index)
                        .ok_or("--summary-md requires a value")?
                        .clone(),
                );
            }
            other => return Err(format!("unknown surrogate validate argument `{other}`").into()),
        }
        index += 1;
    }
    Ok(SummaryOutputs {
        summary_json,
        summary_md,
    })
}

fn parse_optional_out(args: &[String]) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let mut out = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--out" => {
                index += 1;
                out = Some(args.get(index).ok_or("--out requires a value")?.clone());
            }
            other => return Err(format!("unknown surrogate argument `{other}`").into()),
        }
        index += 1;
    }
    Ok(out)
}

fn read_json_value(
    path: impl AsRef<Path>,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    Ok(serde_json::from_reader(File::open(path)?)?)
}

fn parse_bench_threaded_trace_options(
    args: &[String],
) -> Result<BenchThreadedTraceOptions, Box<dyn std::error::Error>> {
    let mut repeat = 1usize;
    let mut warmup = 0usize;
    let mut max_workers = std::thread::available_parallelism().map_or(1, usize::from);
    let mut workers = Vec::new();
    let mut autotune = None;
    let mut autotune_prune = true;
    let mut plan_first = false;
    let mut planner_calibration = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--repeat" => {
                index += 1;
                repeat = args
                    .get(index)
                    .ok_or("--repeat requires a value")?
                    .parse::<usize>()?;
                if repeat == 0 {
                    return Err("--repeat must be greater than zero".into());
                }
            }
            "--warmup" => {
                index += 1;
                warmup = args
                    .get(index)
                    .ok_or("--warmup requires a value")?
                    .parse::<usize>()?;
            }
            "--max-workers" => {
                index += 1;
                max_workers = args
                    .get(index)
                    .ok_or("--max-workers requires a value")?
                    .parse::<usize>()?;
                if max_workers == 0 {
                    return Err("--max-workers must be greater than zero".into());
                }
            }
            "--worker" => {
                index += 1;
                let value = args.get(index).ok_or("--worker requires a value")?;
                workers.push(parse_threaded_worker(value)?);
            }
            "--autotune" => {
                autotune = Some(true);
            }
            "--no-autotune" => {
                autotune = Some(false);
            }
            "--no-autotune-prune" => {
                autotune_prune = false;
            }
            "--plan-first" => {
                plan_first = true;
            }
            "--planner-calibration" => {
                index += 1;
                planner_calibration = Some(read_planner_calibration(
                    args.get(index)
                        .ok_or("--planner-calibration requires a value")?,
                )?);
            }
            other => return Err(format!("unknown bench-threaded-trace argument `{other}`").into()),
        }
        index += 1;
    }
    let autotune = autotune.unwrap_or(workers.is_empty() && !plan_first);
    if !autotune && workers.is_empty() && !plan_first {
        return Err("--no-autotune requires at least one --worker".into());
    }
    Ok(BenchThreadedTraceOptions {
        repeat,
        warmup,
        max_workers,
        workers,
        autotune,
        autotune_prune,
        plan_first,
        planner_calibration,
    })
}

fn parse_bench_gpu_trace_options(
    args: &[String],
) -> Result<BenchGpuTraceOptions, Box<dyn std::error::Error>> {
    let mut options = BenchGpuTraceOptions::default();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--repeat" => {
                index += 1;
                options.repeat = args
                    .get(index)
                    .ok_or("--repeat requires a value")?
                    .parse::<usize>()?;
                if options.repeat == 0 {
                    return Err("--repeat must be greater than zero".into());
                }
            }
            "--warmup" => {
                index += 1;
                options.warmup = args
                    .get(index)
                    .ok_or("--warmup requires a value")?
                    .parse::<usize>()?;
            }
            "--workgroup-size" => {
                index += 1;
                options.workgroup_size = args
                    .get(index)
                    .ok_or("--workgroup-size requires a value")?
                    .parse::<u32>()?;
                if options.workgroup_size == 0 {
                    return Err("--workgroup-size must be greater than zero".into());
                }
            }
            "--memory-layout" => {
                index += 1;
                options.memory_layout = parse_gpu_memory_layout(
                    args.get(index).ok_or("--memory-layout requires a value")?,
                )?;
            }
            "--max-mismatches" => {
                index += 1;
                options.max_mismatches = args
                    .get(index)
                    .ok_or("--max-mismatches requires a value")?
                    .parse::<usize>()?;
            }
            "--host-loop" => {
                options.fused = false;
            }
            "--reuse-temporaries" => {
                options.reuse_temporaries = true;
            }
            "--plan-first" => {
                options.plan_first = true;
            }
            "--planner-calibration" => {
                index += 1;
                options.planner_calibration = Some(read_planner_calibration(
                    args.get(index)
                        .ok_or("--planner-calibration requires a value")?,
                )?);
            }
            other => return Err(format!("unknown bench-gpu-trace argument `{other}`").into()),
        }
        index += 1;
    }
    Ok(options)
}

fn parse_plan_backends_options(
    args: &[String],
) -> Result<PlanBackendsOptions, Box<dyn std::error::Error>> {
    let mut options = PlanBackendsOptions::default();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--max-workers" => {
                index += 1;
                options.max_workers = args
                    .get(index)
                    .ok_or("--max-workers requires a value")?
                    .parse::<usize>()?;
                if options.max_workers == 0 {
                    return Err("--max-workers must be greater than zero".into());
                }
            }
            "--autotune-prune" => {
                options.autotune_prune = true;
            }
            "--no-autotune-prune" => {
                options.autotune_prune = false;
            }
            "--planner-calibration" => {
                index += 1;
                options.planner_calibration = Some(read_planner_calibration(
                    args.get(index)
                        .ok_or("--planner-calibration requires a value")?,
                )?);
            }
            other => return Err(format!("unknown plan-backends argument `{other}`").into()),
        }
        index += 1;
    }
    Ok(options)
}

fn read_planner_calibration(
    path: impl AsRef<Path>,
) -> Result<PlannerCalibration, Box<dyn std::error::Error>> {
    let calibration: PlannerCalibration = serde_json::from_reader(File::open(path)?)?;
    if calibration.schema != "rrtl-pyrtl-planner-calibration-v1" {
        return Err(format!(
            "unsupported planner calibration schema `{}`",
            calibration.schema
        )
        .into());
    }
    Ok(calibration)
}

fn parse_gpu_memory_layout(value: &str) -> Result<GpuMemoryLayout, Box<dyn std::error::Error>> {
    match value {
        "lane-major" | "lane_major" => Ok(GpuMemoryLayout::LaneMajor),
        "word-major" | "word_major" => Ok(GpuMemoryLayout::WordMajor),
        other => Err(format!(
            "unknown GPU memory layout `{other}`; expected lane-major or word-major"
        )
        .into()),
    }
}

fn parse_threaded_worker(
    value: &str,
) -> Result<ThreadedReplayWorkerOptions, Box<dyn std::error::Error>> {
    let (backend, lanes) = value
        .split_once(':')
        .ok_or("--worker must use backend:lanes")?;
    let backend = match backend {
        "scalar" => SimBackendKind::Scalar,
        "packed" | "packed-cpu" => SimBackendKind::PackedCpu,
        "simd" | "simd-cpu" => SimBackendKind::SimdCpu,
        "jit" | "jit-cpu" => SimBackendKind::JitCpu,
        other => return Err(format!("unknown worker backend `{other}`").into()),
    };
    let lanes = lanes.parse::<usize>()?;
    if lanes == 0 {
        return Err("--worker lanes must be greater than zero".into());
    }
    Ok(ThreadedReplayWorkerOptions { backend, lanes })
}

fn parse_bench_trace_options(
    args: &[String],
) -> Result<BenchTraceOptions, Box<dyn std::error::Error>> {
    let mut repeat = 1usize;
    let mut warmup = 0usize;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--repeat" => {
                index += 1;
                repeat = args
                    .get(index)
                    .ok_or("--repeat requires a value")?
                    .parse::<usize>()?;
                if repeat == 0 {
                    return Err("--repeat must be greater than zero".into());
                }
            }
            "--warmup" => {
                index += 1;
                warmup = args
                    .get(index)
                    .ok_or("--warmup requires a value")?
                    .parse::<usize>()?;
            }
            other => return Err(format!("unknown bench-trace argument `{other}`").into()),
        }
        index += 1;
    }
    Ok(BenchTraceOptions { repeat, warmup })
}

fn parse_bench_packed_trace_options(
    args: &[String],
) -> Result<BenchPackedTraceOptions, Box<dyn std::error::Error>> {
    let mut repeat = 1usize;
    let mut warmup = 0usize;
    let mut lanes = 1usize;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--repeat" => {
                index += 1;
                repeat = args
                    .get(index)
                    .ok_or("--repeat requires a value")?
                    .parse::<usize>()?;
                if repeat == 0 {
                    return Err("--repeat must be greater than zero".into());
                }
            }
            "--warmup" => {
                index += 1;
                warmup = args
                    .get(index)
                    .ok_or("--warmup requires a value")?
                    .parse::<usize>()?;
            }
            "--lanes" => {
                index += 1;
                lanes = args
                    .get(index)
                    .ok_or("--lanes requires a value")?
                    .parse::<usize>()?;
                if lanes == 0 {
                    return Err("--lanes must be greater than zero".into());
                }
            }
            other => return Err(format!("unknown bench-packed-trace argument `{other}`").into()),
        }
        index += 1;
    }
    Ok(BenchPackedTraceOptions {
        repeat,
        warmup,
        lanes,
    })
}

fn parse_bench_single_trace_options(
    args: &[String],
) -> Result<BenchSingleTraceOptions, Box<dyn std::error::Error>> {
    let mut repeat = 1usize;
    let mut warmup = 0usize;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--repeat" => {
                index += 1;
                repeat = args
                    .get(index)
                    .ok_or("--repeat requires a value")?
                    .parse::<usize>()?;
                if repeat == 0 {
                    return Err("--repeat must be greater than zero".into());
                }
            }
            "--warmup" => {
                index += 1;
                warmup = args
                    .get(index)
                    .ok_or("--warmup requires a value")?
                    .parse::<usize>()?;
            }
            other => return Err(format!("unknown bench-single-trace argument `{other}`").into()),
        }
        index += 1;
    }
    Ok(BenchSingleTraceOptions { repeat, warmup })
}

fn parse_bench_backends_trace_options(
    args: &[String],
) -> Result<BenchBackendsTraceOptions, Box<dyn std::error::Error>> {
    let mut repeat = 1usize;
    let mut warmup = 0usize;
    let mut lanes = 1usize;
    let mut backends = vec![
        PyrtlBenchBackendKind::Scalar,
        PyrtlBenchBackendKind::PackedCpu,
        PyrtlBenchBackendKind::SimdCpu,
    ];
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--repeat" => {
                index += 1;
                repeat = args
                    .get(index)
                    .ok_or("--repeat requires a value")?
                    .parse::<usize>()?;
                if repeat == 0 {
                    return Err("--repeat must be greater than zero".into());
                }
            }
            "--warmup" => {
                index += 1;
                warmup = args
                    .get(index)
                    .ok_or("--warmup requires a value")?
                    .parse::<usize>()?;
            }
            "--lanes" => {
                index += 1;
                lanes = args
                    .get(index)
                    .ok_or("--lanes requires a value")?
                    .parse::<usize>()?;
                if lanes == 0 {
                    return Err("--lanes must be greater than zero".into());
                }
            }
            "--backend" | "--backends" => {
                index += 1;
                let value = args.get(index).ok_or("--backend requires a value")?;
                backends = parse_backend_list(value)?;
            }
            other => return Err(format!("unknown bench-backends argument `{other}`").into()),
        }
        index += 1;
    }
    Ok(BenchBackendsTraceOptions {
        repeat,
        warmup,
        lanes,
        backends,
    })
}

fn parse_bench_profile_replay_options(
    args: &[String],
) -> Result<ProfileReplayOptions, Box<dyn std::error::Error>> {
    let mut options = ProfileReplayOptions::default();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--repeat" => {
                index += 1;
                options.repeat = args
                    .get(index)
                    .ok_or("--repeat requires a value")?
                    .parse::<usize>()?;
                if options.repeat == 0 {
                    return Err("--repeat must be greater than zero".into());
                }
            }
            "--warmup" => {
                index += 1;
                options.warmup = args
                    .get(index)
                    .ok_or("--warmup requires a value")?
                    .parse::<usize>()?;
            }
            "--lanes" => {
                index += 1;
                options.lanes = args
                    .get(index)
                    .ok_or("--lanes requires a value")?
                    .parse::<usize>()?;
                if options.lanes == 0 {
                    return Err("--lanes must be greater than zero".into());
                }
            }
            other => return Err(format!("unknown bench-profile-replay argument `{other}`").into()),
        }
        index += 1;
    }
    Ok(options)
}

fn parse_backend_list(
    value: &str,
) -> Result<Vec<PyrtlBenchBackendKind>, Box<dyn std::error::Error>> {
    let mut backends = Vec::new();
    for raw in value.split(',') {
        let backend = match raw.trim() {
            "scalar" => PyrtlBenchBackendKind::Scalar,
            "packed" | "packed-cpu" => PyrtlBenchBackendKind::PackedCpu,
            "simd" | "simd-cpu" => PyrtlBenchBackendKind::SimdCpu,
            "jit" | "jit-cpu" => PyrtlBenchBackendKind::JitCpu,
            other => return Err(format!("unknown backend `{other}`").into()),
        };
        if !backends.contains(&backend) {
            backends.push(backend);
        }
    }
    if backends.is_empty() {
        return Err("--backend must name at least one backend".into());
    }
    Ok(backends)
}

fn usage(out: &mut impl Write) -> io::Result<()> {
    writeln!(
        out,
        "usage: pyrtl2rrtl <check|sv|json> <export.pyrtl.json>\n       pyrtl2rrtl compare <export.pyrtl.json> <trace.json>\n       pyrtl2rrtl bench-trace <export.pyrtl.json> <trace.json> [--repeat N] [--warmup N]\n       pyrtl2rrtl bench-packed-trace <export.pyrtl.json> <trace.json> [--repeat N] [--warmup N] [--lanes N]\n       pyrtl2rrtl bench-single-trace <export.pyrtl.json> <trace.json> [--repeat N] [--warmup N]\n       pyrtl2rrtl bench-backends <export.pyrtl.json> <trace.json> [--backend scalar,packed-cpu,simd-cpu,jit-cpu] [--repeat N] [--warmup N] [--lanes N]\n       pyrtl2rrtl bench-threaded-trace <export.pyrtl.json> <lane-trace.json> [--repeat N] [--warmup N] [--max-workers N] [--plan-first] [--planner-calibration calibration.json] [--autotune|--no-autotune] [--no-autotune-prune] [--worker scalar:N|packed-cpu:N|simd-cpu:N|jit-cpu:N]\n       pyrtl2rrtl plan-backends <export.pyrtl.json> <lane-trace.json> [--max-workers N] [--no-autotune-prune] [--planner-calibration calibration.json]\n       pyrtl2rrtl bench-gpu-trace <export.pyrtl.json> <lane-trace.json> [--repeat N] [--warmup N] [--workgroup-size N] [--memory-layout lane-major|word-major] [--max-mismatches N] [--host-loop] [--reuse-temporaries] [--plan-first] [--planner-calibration calibration.json]\n       pyrtl2rrtl bench-gpu-options <export.pyrtl.json> <lane-trace.json> [--repeat N] [--warmup N] [--max-mismatches N] [--host-loop] [--planner-calibration calibration.json]\n       pyrtl2rrtl bench-gpu-combined <export.pyrtl.json> <lane-trace.json> [--repeat N] [--warmup N] [--max-mismatches N] [--host-loop] [--planner-calibration calibration.json]\n       pyrtl2rrtl bench-profile-replay <export.pyrtl.json> <trace.json> <lane-trace.json> <runtime_profile.json> [--repeat N] [--warmup N] [--lanes N]\n       pyrtl2rrtl surrogate <validate-manifest|export-dataset|validate|inspect-instrumentation|emit-events|emit-instrumented-events|inspect-events|validate-events|shadow-events|policy-events|run-fast-events|plan-model-fast|run-model-fast|run-gemm|run-gemm-batch|policy-gemm-batch|run-fast-gemm|plan-runtime-gemm|plan-runtime-events|attach-runtime-gemm|attach-runtime-events|inspect-runtime-attachment> ..."
    )
}

fn surrogate_usage(out: &mut impl Write) -> io::Result<()> {
    writeln!(
        out,
        "usage: pyrtl2rrtl surrogate validate-manifest <manifest.json>\n       pyrtl2rrtl surrogate export-dataset <export.pyrtl.json> <trace.json> [--out dataset.json]\n       pyrtl2rrtl surrogate validate <manifest.json> <export.pyrtl.json> <trace.json> [--summary-json out.json] [--summary-md out.md]\n       pyrtl2rrtl surrogate inspect-instrumentation <instrumentation.json> [--config config.json] [--out report.json]\n       pyrtl2rrtl surrogate emit-events <trace.json> --config config.json [--out events.json]\n       pyrtl2rrtl surrogate emit-instrumented-events <instrumentation.json> --config config.json [--out events.json]\n       pyrtl2rrtl surrogate inspect-events <events.json> [--out report.json]\n       pyrtl2rrtl surrogate validate-events <manifest.json> <events.json> [--out report.json]\n       pyrtl2rrtl surrogate shadow-events <manifest.json> <events.json> [--out report.json]\n       pyrtl2rrtl surrogate policy-events <manifest.json> <events.json> [--out report.json]\n       pyrtl2rrtl surrogate run-fast-events <manifest.json> <events.json> <runtime-plan.json> [--shadow-sample-stride N] [--shadow-sample-offset N] [--out report.json]\n       pyrtl2rrtl surrogate plan-model-fast --op id:path[:name]... [--golden op_id:path]... [--timing op_id:exact_ns:fast_ns]... [--thresholds thresholds.json] [--out plan.json]\n       pyrtl2rrtl surrogate run-model-fast <model-fast-plan.json> [--out report.json]\n       pyrtl2rrtl surrogate run-gemm <manifest.json> <transaction.json> [--out result.json]\n       pyrtl2rrtl surrogate run-gemm-batch <manifest.json> <batch.json> [--out result.json]\n       pyrtl2rrtl surrogate policy-gemm-batch <manifest.json> <batch.json> [--out report.json]\n       pyrtl2rrtl surrogate run-fast-gemm <manifest.json> <batch.json> <runtime-plan.json> [--shadow-sample-stride N] [--shadow-sample-offset N] [--out report.json]\n       pyrtl2rrtl surrogate plan-runtime-gemm <policy-report.json> (--topology lanes:N|--worker id:lanes...) [--out plan.json]\n       pyrtl2rrtl surrogate plan-runtime-events <policy-report.json> (--topology lanes:N|--worker id:lanes...) [--out plan.json]\n       pyrtl2rrtl surrogate attach-runtime-gemm <runtime-plan.json> --worker id:lanes... [--out attachment.json]\n       pyrtl2rrtl surrogate attach-runtime-events <runtime-plan.json> --worker id:lanes... [--out attachment.json]\n       pyrtl2rrtl surrogate inspect-runtime-attachment <attachment.json> [--out report.json]"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rrtl_pyrtl::RuntimeProfileWorker;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn parses_bench_profile_replay_options() {
        let options = parse_bench_profile_replay_options(&args(&[
            "--repeat", "3", "--warmup", "1", "--lanes", "4",
        ]))
        .unwrap();

        assert_eq!(options.repeat, 3);
        assert_eq!(options.warmup, 1);
        assert_eq!(options.lanes, 4);
        assert!(parse_bench_profile_replay_options(&args(&["--lanes", "0"])).is_err());
    }

    #[test]
    fn runtime_profile_validation_rejects_bad_schema_and_missing_selection() {
        let bad_schema: RuntimeProfile = serde_json::from_str(
            r#"{"schema":"bad","recommended_runtime_backend":"rrtl_backend:simd-cpu","selected_backend":{}}"#,
        )
        .unwrap();
        assert!(validate_runtime_profile(&bad_schema)
            .unwrap_err()
            .to_string()
            .contains("unsupported runtime profile schema"));

        let missing_selection: RuntimeProfile = serde_json::from_str(
            r#"{"schema":"rrtl-pyrtl-runtime-profile-v1","recommended_runtime_backend":"rrtl_backend:simd-cpu"}"#,
        )
        .unwrap();
        assert!(validate_runtime_profile(&missing_selection)
            .unwrap_err()
            .to_string()
            .contains("missing selected backend details"));
    }

    #[test]
    fn runtime_profile_workers_parse_backend_layout() {
        let workers = runtime_profile_workers(&[
            RuntimeProfileWorker {
                backend: "scalar".to_string(),
                lanes: 1,
            },
            RuntimeProfileWorker {
                backend: "simd-cpu".to_string(),
                lanes: 3,
            },
        ])
        .unwrap();

        assert_eq!(workers.len(), 2);
        assert_eq!(workers[0].backend, SimBackendKind::Scalar);
        assert_eq!(workers[1].backend, SimBackendKind::SimdCpu);
        assert_eq!(workers[1].lanes, 3);
    }

    #[test]
    fn parses_plan_runtime_gemm_worker_topology() {
        let options = parse_plan_runtime_gemm_options(&args(&[
            "--worker",
            "cpu-a:2",
            "--worker",
            "cpu-b:3",
            "--out",
            "plan.json",
        ]))
        .unwrap();

        let PlanRuntimeGemmTopology::Workers(workers) = options.topology else {
            panic!("expected worker topology");
        };
        assert_eq!(workers.len(), 2);
        assert_eq!(workers[0].worker_id, "cpu-a");
        assert_eq!(workers[0].start_lane, 0);
        assert_eq!(workers[0].lanes, 2);
        assert_eq!(workers[1].worker_id, "cpu-b");
        assert_eq!(workers[1].start_lane, 2);
        assert_eq!(workers[1].lanes, 3);
        assert_eq!(options.out.as_deref(), Some("plan.json"));
    }

    #[test]
    fn rejects_mixed_plan_runtime_gemm_topologies() {
        let result = parse_plan_runtime_gemm_options(&args(&[
            "--topology",
            "lanes:2",
            "--worker",
            "cpu-a:2",
        ]));

        let Err(err) = result else {
            panic!("expected mixed topology rejection");
        };
        assert!(err.to_string().contains("cannot mix"));
    }

    #[test]
    fn parses_attach_runtime_gemm_workers() {
        let options = parse_attach_runtime_gemm_options(&args(&[
            "--worker", "cpu-a:2", "--worker", "cpu-b:1",
        ]))
        .unwrap();

        assert_eq!(options.workers.len(), 2);
        assert_eq!(options.workers[1].worker_id, "cpu-b");
        assert_eq!(options.workers[1].start_lane, 2);
        assert_eq!(options.workers[1].lanes, 1);
    }

    #[test]
    fn parses_run_fast_gemm_options() {
        let options = parse_run_fast_gemm_options(&args(&[
            "--shadow-sample-stride",
            "3",
            "--shadow-sample-offset",
            "1",
            "--out",
            "fast.json",
        ]))
        .unwrap();

        assert_eq!(options.fast.shadow_sample_stride, Some(3));
        assert_eq!(options.fast.shadow_sample_offset, 1);
        assert_eq!(options.out.as_deref(), Some("fast.json"));
    }

    #[test]
    fn rejects_zero_run_fast_gemm_shadow_stride() {
        let result = parse_run_fast_gemm_options(&args(&["--shadow-sample-stride", "0"]));

        let Err(err) = result else {
            panic!("expected zero shadow stride rejection");
        };
        assert!(err.to_string().contains("greater than zero"));
    }

    #[test]
    fn parses_run_fast_event_options() {
        let options = parse_run_fast_event_options(&args(&[
            "--shadow-sample-stride",
            "4",
            "--shadow-sample-offset",
            "2",
            "--out",
            "events-fast.json",
        ]))
        .unwrap();

        assert_eq!(options.fast.shadow_sample_stride, Some(4));
        assert_eq!(options.fast.shadow_sample_offset, 2);
        assert_eq!(options.out.as_deref(), Some("events-fast.json"));
    }

    #[test]
    fn rejects_zero_run_fast_event_shadow_stride() {
        let result = parse_run_fast_event_options(&args(&["--shadow-sample-stride", "0"]));

        let Err(err) = result else {
            panic!("expected zero shadow stride rejection");
        };
        assert!(err.to_string().contains("greater than zero"));
    }

    #[test]
    fn parses_plan_model_fast_options_and_op_specs() {
        let options = parse_plan_model_fast_options(&args(&[
            "--op",
            "gemm0:gemm.json:GEMM tile",
            "--golden",
            "gemm0:gemm_golden.json",
            "--timing",
            "gemm0:1000:250",
            "--thresholds",
            "thresholds.json",
            "--out",
            "plan.json",
        ]))
        .unwrap();

        assert_eq!(options.ops, vec!["gemm0:gemm.json:GEMM tile"]);
        assert_eq!(options.golden, vec!["gemm0:gemm_golden.json"]);
        assert_eq!(options.timing, vec!["gemm0:1000:250"]);
        assert_eq!(options.thresholds.as_deref(), Some("thresholds.json"));
        assert_eq!(options.out.as_deref(), Some("plan.json"));

        let op = parse_model_fast_op_spec("event0:event.json").unwrap();
        assert_eq!(op.op_id, "event0");
        assert_eq!(op.fast_report_path, "event.json");
        assert_eq!(op.name, "event0");
        assert!(parse_model_fast_op_spec("missing-path").is_err());
        assert!(parse_model_fast_op_spec(":path").is_err());
        assert!(parse_model_fast_op_spec("id:").is_err());
        assert!(parse_model_fast_op_spec("id:path:").is_err());
        assert!(parse_model_fast_golden_bindings(&args(&["op0:golden.json"])).is_ok());
        assert!(parse_model_fast_golden_bindings(&args(&["op0"])).is_err());
        assert!(parse_model_fast_golden_bindings(&args(&[":golden.json"])).is_err());
        assert!(parse_model_fast_golden_bindings(&args(&["op0:"])).is_err());
        assert!(parse_model_fast_golden_bindings(&args(&["op0:a.json", "op0:b.json"])).is_err());
        assert!(parse_model_fast_timing_bindings(&args(&["op0:1000:250"])).is_ok());
        assert!(parse_model_fast_timing_bindings(&args(&["op0"])).is_err());
        assert!(parse_model_fast_timing_bindings(&args(&[":1000:250"])).is_err());
        assert!(parse_model_fast_timing_bindings(&args(&["op0:0:250"])).is_err());
        assert!(parse_model_fast_timing_bindings(&args(&["op0:1000:0"])).is_err());
        assert!(parse_model_fast_timing_bindings(&args(&["op0:1000:250", "op0:900:300"])).is_err());
    }

    #[test]
    fn plan_model_fast_generates_ordered_mixed_plan() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-pyrtl-test-plan-model-fast-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let gemm = dir.join("gemm_fast.json");
        let event = dir.join("event_fast.json");
        let event_golden = dir.join("event_golden.json");
        let thresholds = dir.join("thresholds.json");
        let out = dir.join("plan.json");
        fs::write(
            &gemm,
            serde_json::json!({ "schema": rrtl_surrogate::GEMM_FAST_RUN_SCHEMA }).to_string(),
        )
        .unwrap();
        fs::write(
            &event,
            serde_json::json!({ "schema": rrtl_surrogate::EVENT_FAST_RUN_SCHEMA }).to_string(),
        )
        .unwrap();
        fs::write(
            &event_golden,
            serde_json::json!({ "schema": "golden" }).to_string(),
        )
        .unwrap();
        fs::write(
            &thresholds,
            serde_json::json!({
                "min_op_coverage": 1.0,
                "min_item_coverage": 0.5,
                "max_fallback_ratio": 0.25,
                "min_shadow_sample_ratio": 0.1
            })
            .to_string(),
        )
        .unwrap();

        run_surrogate(args(&[
            "plan-model-fast",
            "--op",
            &format!("event0:{}:Cache miss", event.display()),
            "--op",
            &format!("gemm0:{}:GEMM tile", gemm.display()),
            "--golden",
            &format!("event0:{}", event_golden.display()),
            "--timing",
            "event0:1000:250",
            "--thresholds",
            thresholds.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ]))
        .unwrap();

        let plan: rrtl_surrogate::ModelFastPlan =
            serde_json::from_reader(File::open(&out).unwrap()).unwrap();
        assert_eq!(plan.schema, rrtl_surrogate::MODEL_FAST_PLAN_SCHEMA);
        assert_eq!(plan.ops.len(), 2);
        assert_eq!(plan.ops[0].op_id, "event0");
        assert_eq!(plan.ops[0].op_kind, "event");
        assert_eq!(plan.ops[0].name, "Cache miss");
        assert_eq!(
            plan.ops[0].golden_path.as_deref(),
            Some(event_golden.to_str().unwrap())
        );
        assert_eq!(plan.ops[0].exact_ns, Some(1000));
        assert_eq!(plan.ops[0].fast_ns, Some(250));
        assert_eq!(plan.ops[1].op_id, "gemm0");
        assert_eq!(plan.ops[1].op_kind, "gemm");
        assert_eq!(plan.ops[1].name, "GEMM tile");
        assert_eq!(plan.ops[1].golden_path, None);
        assert_eq!(plan.ops[1].exact_ns, None);
        assert_eq!(plan.ops[1].fast_ns, None);
        let thresholds = plan.thresholds.unwrap();
        assert_eq!(thresholds.min_op_coverage, Some(1.0));
        assert_eq!(thresholds.min_item_coverage, Some(0.5));
        assert_eq!(thresholds.max_fallback_ratio, Some(0.25));
        assert_eq!(thresholds.min_shadow_sample_ratio, Some(0.1));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn plan_model_fast_rejects_duplicate_ids_and_unknown_schemas() {
        let dir = std::env::temp_dir().join(format!(
            "rrtl-pyrtl-test-plan-model-fast-errors-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let gemm = dir.join("gemm_fast.json");
        let bad = dir.join("bad_fast.json");
        let golden = dir.join("golden.json");
        fs::write(
            &gemm,
            serde_json::json!({ "schema": rrtl_surrogate::GEMM_FAST_RUN_SCHEMA }).to_string(),
        )
        .unwrap();
        fs::write(&bad, serde_json::json!({ "schema": "bad" }).to_string()).unwrap();
        fs::write(
            &golden,
            serde_json::json!({ "schema": "golden" }).to_string(),
        )
        .unwrap();

        let duplicate = run_surrogate(args(&[
            "plan-model-fast",
            "--op",
            &format!("op0:{}", gemm.display()),
            "--op",
            &format!("op0:{}", gemm.display()),
        ]));
        assert!(duplicate
            .unwrap_err()
            .to_string()
            .contains("duplicate model FAST op id"));

        let unsupported = run_surrogate(args(&[
            "plan-model-fast",
            "--op",
            &format!("bad0:{}", bad.display()),
        ]));
        assert!(unsupported
            .unwrap_err()
            .to_string()
            .contains("unsupported model FAST op report schema `bad`"));

        let duplicate_golden = run_surrogate(args(&[
            "plan-model-fast",
            "--op",
            &format!("op0:{}", gemm.display()),
            "--golden",
            &format!("op0:{}", golden.display()),
            "--golden",
            &format!("op0:{}", golden.display()),
        ]));
        assert!(duplicate_golden
            .unwrap_err()
            .to_string()
            .contains("duplicate model FAST golden binding for op `op0`"));

        let unknown_golden = run_surrogate(args(&[
            "plan-model-fast",
            "--op",
            &format!("op0:{}", gemm.display()),
            "--golden",
            &format!("missing:{}", golden.display()),
        ]));
        assert!(unknown_golden
            .unwrap_err()
            .to_string()
            .contains("model FAST golden references unknown op `missing`"));

        let malformed_golden = run_surrogate(args(&[
            "plan-model-fast",
            "--op",
            &format!("op0:{}", gemm.display()),
            "--golden",
            "op0",
        ]));
        assert!(malformed_golden
            .unwrap_err()
            .to_string()
            .contains("--golden must use op_id:path"));

        let unreadable_golden = run_surrogate(args(&[
            "plan-model-fast",
            "--op",
            &format!("op0:{}", gemm.display()),
            "--golden",
            &format!("op0:{}", dir.join("missing_golden.json").display()),
        ]));
        assert!(unreadable_golden
            .unwrap_err()
            .to_string()
            .contains("failed to open model FAST golden"));

        let duplicate_timing = run_surrogate(args(&[
            "plan-model-fast",
            "--op",
            &format!("op0:{}", gemm.display()),
            "--timing",
            "op0:1000:250",
            "--timing",
            "op0:900:300",
        ]));
        assert!(duplicate_timing
            .unwrap_err()
            .to_string()
            .contains("duplicate model FAST timing binding for op `op0`"));

        let unknown_timing = run_surrogate(args(&[
            "plan-model-fast",
            "--op",
            &format!("op0:{}", gemm.display()),
            "--timing",
            "missing:1000:250",
        ]));
        assert!(unknown_timing
            .unwrap_err()
            .to_string()
            .contains("model FAST timing references unknown op `missing`"));

        let malformed_timing = run_surrogate(args(&[
            "plan-model-fast",
            "--op",
            &format!("op0:{}", gemm.display()),
            "--timing",
            "op0:1000",
        ]));
        assert!(malformed_timing
            .unwrap_err()
            .to_string()
            .contains("--timing must use op_id:exact_ns:fast_ns"));

        let zero_timing = run_surrogate(args(&[
            "plan-model-fast",
            "--op",
            &format!("op0:{}", gemm.display()),
            "--timing",
            "op0:1000:0",
        ]));
        assert!(zero_timing
            .unwrap_err()
            .to_string()
            .contains("--timing fast_ns must be greater than zero"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn shared_runtime_worker_parser_supports_event_commands() {
        let options = parse_plan_runtime_gemm_options(&args(&[
            "--worker",
            "event-a:1",
            "--worker",
            "event-b:2",
        ]))
        .unwrap();

        let PlanRuntimeGemmTopology::Workers(workers) = options.topology else {
            panic!("expected worker topology");
        };
        assert_eq!(workers[0].worker_id, "event-a");
        assert_eq!(workers[1].worker_id, "event-b");
        assert_eq!(workers[1].start_lane, 1);
        assert_eq!(workers[1].lanes, 2);
    }

    #[test]
    fn surrogate_usage_lists_run_fast_gemm() {
        let mut bytes = Vec::new();

        surrogate_usage(&mut bytes).unwrap();

        let text = String::from_utf8(bytes).unwrap();
        assert!(text.contains("run-fast-gemm <manifest.json> <batch.json> <runtime-plan.json>"));
        assert!(text.contains("run-fast-events <manifest.json> <events.json> <runtime-plan.json>"));
        assert!(text.contains("plan-model-fast --op id:path[:name]"));
        assert!(text.contains("[--golden op_id:path]"));
        assert!(text.contains("[--timing op_id:exact_ns:fast_ns]"));
        assert!(text.contains("run-model-fast <model-fast-plan.json>"));
        assert!(text.contains("--shadow-sample-stride N"));
    }
}
