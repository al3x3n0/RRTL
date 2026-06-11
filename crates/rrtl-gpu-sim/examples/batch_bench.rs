use std::env;
use std::fs::File;
use std::io::{self, Write};
use std::time::{Duration, Instant};

use rrtl_core::{compile, concat, lit_u, mux, uint, Design, Signal, Simulator};
use rrtl_gpu_sim::{
    gpu_shader_stats, GpuAutotuneRecommendation, GpuBatchOptions, GpuBatchSimulator,
    GpuMemoryLayout, WORKGROUP_SIZE,
};
use rrtl_sim_ir::{lower_to_packed_program, PackedScheduleOptions, PackedSimulator};
use serde::{Deserialize, Serialize};

const LANE_COUNTS: [usize; 4] = [64, 256, 1024, 4096];
const STEP_COUNTS: [usize; 4] = [1, 16, 128, 512];
const SCHEDULE_CAPS: [Option<usize>; 4] = [Some(4), Some(8), Some(16), None];
const WORKGROUP_COUNTS: [u32; 1] = [WORKGROUP_SIZE];
const MEMORY_LAYOUTS: [GpuMemoryLayout; 2] =
    [GpuMemoryLayout::LaneMajor, GpuMemoryLayout::WordMajor];
const CASE_NAMES: [&str; 7] = [
    "counter",
    "wide_datapath",
    "register_file",
    "fifo_like",
    "mixed_memory_datapath",
    "multi_read_memory",
    "deep_mixed_pipeline",
];

#[derive(Clone, Debug, PartialEq)]
struct BenchConfig {
    format: OutputFormat,
    output: Option<String>,
    compare: Option<String>,
    repeat: usize,
    timing_threshold_pct: f64,
    strict: bool,
    autotune: bool,
    autotune_metric: AutotuneMetric,
    recommend_config: bool,
    cases: Vec<String>,
    lanes: Vec<usize>,
    steps: Vec<usize>,
    caps: Vec<Option<usize>>,
    mem_read_caps: Vec<Option<usize>>,
    liveness_priority: bool,
    reuse_temporaries: bool,
    workgroups: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputFormat {
    Csv,
    Human,
    Json,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AutotuneMetric {
    Packed,
    GpuTick,
    GpuTickMany,
}

struct BenchCase {
    name: &'static str,
    module: &'static str,
    design: Design,
    inputs: Vec<BenchInput>,
    output: Signal,
}

struct BenchInput {
    signal: Signal,
    seed: u128,
    stride: u128,
}

#[derive(Clone, Debug)]
struct TimedChecksum {
    duration: Duration,
    checksum: u128,
}

#[derive(Clone, Copy)]
enum GpuMode {
    TickLoop,
    TickMany,
}

#[derive(Clone, Debug)]
struct GpuTiming {
    construct: Duration,
    run: Duration,
    total: Duration,
    checksum: u128,
}

#[derive(Clone, Debug, Default)]
struct TimingSamples {
    cpu: Vec<TimedChecksum>,
    packed: Vec<TimedChecksum>,
    gpu_tick: Vec<Option<GpuTiming>>,
    gpu_tick_many: Vec<Option<GpuTiming>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct BenchRow {
    case: String,
    lanes: usize,
    steps: usize,
    schedule_cap: Option<usize>,
    memory_read_cap: Option<usize>,
    liveness_priority: bool,
    reuse_temporaries: bool,
    memory_layout: String,
    gpu_mode: String,
    workgroup_size: u32,
    wgsl_bytes: usize,
    optimized_temp_slots: usize,
    optimized_value_vars: usize,
    packets_total: usize,
    packets_async_reset_comb: usize,
    packets_comb: usize,
    packets_tick_next: usize,
    packets_tick_commit: usize,
    max_packet_width: usize,
    max_live_values: usize,
    avg_live_values_x100: usize,
    unoptimized_packets_total: usize,
    unoptimized_max_packet_width: usize,
    unoptimized_max_live_values: usize,
    unoptimized_avg_live_values_x100: usize,
    optimized_vs_unoptimized_packets_delta: i128,
    optimized_vs_unoptimized_packets_delta_pct_x100: i128,
    packet_utilization_x100: usize,
    max_packet_memory_reads: usize,
    memory_reads: usize,
    memory_writes: usize,
    total_memory_words_per_lane: usize,
    repeat: usize,
    cpu_ns: u128,
    cpu_ns_min: u128,
    cpu_ns_max: u128,
    packed_ns: u128,
    packed_ns_min: u128,
    packed_ns_max: u128,
    gpu_tick_construct_ns: Option<u128>,
    gpu_tick_construct_ns_min: Option<u128>,
    gpu_tick_construct_ns_max: Option<u128>,
    gpu_tick_ns: Option<u128>,
    gpu_tick_ns_min: Option<u128>,
    gpu_tick_ns_max: Option<u128>,
    gpu_tick_total_ns: Option<u128>,
    gpu_tick_total_ns_min: Option<u128>,
    gpu_tick_total_ns_max: Option<u128>,
    gpu_tick_many_construct_ns: Option<u128>,
    gpu_tick_many_construct_ns_min: Option<u128>,
    gpu_tick_many_construct_ns_max: Option<u128>,
    gpu_tick_many_ns: Option<u128>,
    gpu_tick_many_ns_min: Option<u128>,
    gpu_tick_many_ns_max: Option<u128>,
    gpu_tick_many_total_ns: Option<u128>,
    gpu_tick_many_total_ns_min: Option<u128>,
    gpu_tick_many_total_ns_max: Option<u128>,
    cpu_checksum: String,
    packed_checksum: String,
    gpu_tick_checksum: Option<String>,
    gpu_tick_many_checksum: Option<String>,
    gpu_available: bool,
    packed_vs_cpu: Option<f64>,
    gpu_tick_many_vs_packed: Option<f64>,
    autotune_rank: Option<usize>,
    autotune_metric: String,
    autotune_metric_ns: Option<u128>,
    autotune_best: bool,
}

fn main() {
    let config = BenchConfig::parse(env::args().skip(1));
    let rows = run_benchmarks(&config);
    let mut out: Box<dyn Write> = match &config.output {
        Some(path) => Box::new(io::BufWriter::new(File::create(path).unwrap())),
        None => Box::new(io::BufWriter::new(io::stdout().lock())),
    };

    if let Some(path) = &config.compare {
        let baseline = read_json_rows(path);
        let issues = print_compare_report(&mut out, &baseline, &rows, config.timing_threshold_pct);
        let exit_code = compare_exit_code(config.strict, issues);
        if exit_code != 0 {
            std::process::exit(exit_code);
        }
    } else if config.recommend_config {
        print_recommended_configs(&mut out, &rows);
    } else {
        print_rows(&mut out, config.format, &rows);
    }
}

fn all_cases() -> Vec<BenchCase> {
    vec![
        counter_case(),
        wide_datapath_case(),
        register_file_case(),
        fifo_like_case(),
        mixed_memory_datapath_case(),
        multi_read_memory_case(),
        deep_mixed_pipeline_case(),
    ]
}

impl BenchConfig {
    fn parse(args: impl IntoIterator<Item = String>) -> Self {
        let mut quick = false;
        let mut format = OutputFormat::Csv;
        let mut output = None;
        let mut compare = None;
        let mut repeat = 1usize;
        let mut timing_threshold_pct = 10.0f64;
        let mut strict = false;
        let mut autotune = false;
        let mut autotune_metric = AutotuneMetric::GpuTickMany;
        let mut recommend_config = false;
        let mut cases = Vec::new();
        let mut lanes = None;
        let mut steps = None;
        let mut caps = None;
        let mut mem_read_caps = None;
        let mut liveness_priority = false;
        let mut reuse_temporaries = false;
        let mut workgroups = None;
        let mut args = args.into_iter();

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--quick" => quick = true,
                "--human" => format = OutputFormat::Human,
                "--case" => cases.push(next_arg(&mut args, "--case")),
                "--format" => format = parse_output_format(&next_arg(&mut args, "--format")),
                "--output" => output = Some(next_arg(&mut args, "--output")),
                "--compare" => compare = Some(next_arg(&mut args, "--compare")),
                "--repeat" => {
                    repeat = parse_positive_usize(&next_arg(&mut args, "--repeat"), "--repeat")
                }
                "--timing-threshold" => {
                    timing_threshold_pct = parse_nonnegative_f64(
                        &next_arg(&mut args, "--timing-threshold"),
                        "--timing-threshold",
                    )
                }
                "--strict" => strict = true,
                "--autotune" => autotune = true,
                "--autotune-metric" => {
                    autotune_metric =
                        parse_autotune_metric(&next_arg(&mut args, "--autotune-metric"))
                }
                "--recommend-config" => recommend_config = true,
                "--lanes" => {
                    lanes = Some(parse_usize_list(&next_arg(&mut args, "--lanes"), "--lanes"))
                }
                "--steps" => {
                    steps = Some(parse_usize_list(&next_arg(&mut args, "--steps"), "--steps"))
                }
                "--caps" => caps = Some(parse_cap_list(&next_arg(&mut args, "--caps"))),
                "--mem-read-caps" => {
                    mem_read_caps = Some(parse_memory_read_cap_list(&next_arg(
                        &mut args,
                        "--mem-read-caps",
                    )))
                }
                "--liveness-priority" => liveness_priority = true,
                "--reuse-temporaries" => reuse_temporaries = true,
                "--workgroups" => {
                    workgroups = Some(parse_u32_list(
                        &next_arg(&mut args, "--workgroups"),
                        "--workgroups",
                    ))
                }
                "--help" | "-h" => {
                    print_help_and_exit();
                }
                other => panic!("unknown argument `{other}`; use --help for usage"),
            }
        }

        for case in &cases {
            if !CASE_NAMES.contains(&case.as_str()) {
                panic!(
                    "unknown benchmark case `{case}`; expected one of {}",
                    CASE_NAMES.join(",")
                );
            }
        }
        if recommend_config && !autotune {
            panic!("--recommend-config requires --autotune");
        }

        Self {
            format,
            output,
            compare,
            repeat,
            timing_threshold_pct,
            strict,
            autotune,
            autotune_metric,
            recommend_config,
            cases,
            lanes: lanes.unwrap_or_else(|| {
                if quick {
                    vec![64]
                } else {
                    LANE_COUNTS.to_vec()
                }
            }),
            steps: steps.unwrap_or_else(|| {
                if quick {
                    vec![1, 16]
                } else {
                    STEP_COUNTS.to_vec()
                }
            }),
            caps: caps.unwrap_or_else(|| {
                if quick {
                    vec![Some(16)]
                } else {
                    SCHEDULE_CAPS.to_vec()
                }
            }),
            mem_read_caps: mem_read_caps.unwrap_or_else(|| vec![None]),
            liveness_priority,
            reuse_temporaries,
            workgroups: workgroups.unwrap_or_else(|| WORKGROUP_COUNTS.to_vec()),
        }
    }

    fn should_run_case(&self, name: &str) -> bool {
        self.cases.is_empty() || self.cases.iter().any(|case| case == name)
    }
}

fn parse_output_format(value: &str) -> OutputFormat {
    match value {
        "csv" => OutputFormat::Csv,
        "human" => OutputFormat::Human,
        "json" => OutputFormat::Json,
        other => panic!("unknown --format `{other}`; expected csv, human, or json"),
    }
}

fn parse_autotune_metric(value: &str) -> AutotuneMetric {
    match value {
        "packed" => AutotuneMetric::Packed,
        "gpu_tick" => AutotuneMetric::GpuTick,
        "gpu_tick_many" => AutotuneMetric::GpuTickMany,
        other => panic!(
            "unknown --autotune-metric `{other}`; expected packed, gpu_tick, or gpu_tick_many"
        ),
    }
}

fn format_autotune_metric(metric: AutotuneMetric) -> &'static str {
    match metric {
        AutotuneMetric::Packed => "packed",
        AutotuneMetric::GpuTick => "gpu_tick",
        AutotuneMetric::GpuTickMany => "gpu_tick_many",
    }
}

fn parse_positive_usize(value: &str, flag: &str) -> usize {
    let parsed = value
        .parse::<usize>()
        .unwrap_or_else(|_| panic!("{flag} contains invalid integer `{value}`"));
    if parsed == 0 {
        panic!("{flag} must be greater than zero");
    }
    parsed
}

fn parse_nonnegative_f64(value: &str, flag: &str) -> f64 {
    let parsed = value
        .parse::<f64>()
        .unwrap_or_else(|_| panic!("{flag} contains invalid number `{value}`"));
    if !parsed.is_finite() || parsed < 0.0 {
        panic!("{flag} must be a non-negative finite number");
    }
    parsed
}

fn next_arg(args: &mut impl Iterator<Item = String>, flag: &str) -> String {
    args.next()
        .unwrap_or_else(|| panic!("{flag} requires a value"))
}

fn parse_usize_list(value: &str, flag: &str) -> Vec<usize> {
    let values = value
        .split(',')
        .map(|part| {
            let part = part.trim();
            part.parse::<usize>()
                .unwrap_or_else(|_| panic!("{flag} contains invalid integer `{part}`"))
        })
        .collect::<Vec<_>>();
    if values.is_empty() {
        panic!("{flag} requires at least one value");
    }
    values
}

fn parse_u32_list(value: &str, flag: &str) -> Vec<u32> {
    let values = value
        .split(',')
        .map(|part| {
            let part = part.trim();
            let parsed = part
                .parse::<u32>()
                .unwrap_or_else(|_| panic!("{flag} contains invalid integer `{part}`"));
            if parsed == 0 {
                panic!("{flag} values must be greater than zero");
            }
            parsed
        })
        .collect::<Vec<_>>();
    if values.is_empty() {
        panic!("{flag} requires at least one value");
    }
    values
}

fn parse_cap_list(value: &str) -> Vec<Option<usize>> {
    let values = value
        .split(',')
        .map(|part| {
            let part = part.trim();
            if part == "none" {
                None
            } else {
                Some(
                    part.parse::<usize>()
                        .unwrap_or_else(|_| panic!("--caps contains invalid cap `{part}`")),
                )
            }
        })
        .collect::<Vec<_>>();
    if values.is_empty() {
        panic!("--caps requires at least one value");
    }
    values
}

fn parse_memory_read_cap_list(value: &str) -> Vec<Option<usize>> {
    let values = value
        .split(',')
        .map(|part| {
            let part = part.trim();
            if part == "none" {
                None
            } else {
                let parsed = part
                    .parse::<usize>()
                    .unwrap_or_else(|_| panic!("--mem-read-caps contains invalid cap `{part}`"));
                if parsed == 0 {
                    panic!("--mem-read-caps values must be greater than zero");
                }
                Some(parsed)
            }
        })
        .collect::<Vec<_>>();
    if values.is_empty() {
        panic!("--mem-read-caps requires at least one value");
    }
    values
}

fn print_help_and_exit() -> ! {
    println!(
        "usage: cargo run -p rrtl-gpu-sim --example batch_bench --release -- [--quick] [--human] [--format csv|human|json] [--output PATH] [--compare BASELINE_JSON] [--repeat N] [--timing-threshold PCT] [--strict] [--case NAME] [--lanes CSV] [--steps CSV] [--caps CSV] [--mem-read-caps CSV] [--liveness-priority] [--reuse-temporaries] [--autotune] [--autotune-metric packed|gpu_tick|gpu_tick_many] [--recommend-config] [--workgroups CSV]\n\ncases: {}",
        CASE_NAMES.join(",")
    );
    std::process::exit(0);
}

fn run_benchmarks(config: &BenchConfig) -> Vec<BenchRow> {
    let mut rows = Vec::new();
    for case in all_cases() {
        if config.should_run_case(case.name) {
            run_case(&case, config, &mut rows);
        }
    }
    if config.autotune {
        rank_autotune_rows(&mut rows, config.autotune_metric);
    }
    rows
}

fn run_case(case: &BenchCase, config: &BenchConfig, rows: &mut Vec<BenchRow>) {
    let compiled = compile(&case.design).unwrap();
    let program = lower_to_packed_program(&compiled, case.module).unwrap();

    for lanes in config.lanes.iter().copied() {
        let inputs = materialize_inputs(case, lanes);
        for steps in config.steps.iter().copied() {
            for schedule_cap in config.caps.iter().copied() {
                for mem_read_cap in config.mem_read_caps.iter().copied() {
                    for liveness_priority in bool_sweep(config.autotune, config.liveness_priority)
                        .iter()
                        .copied()
                    {
                        for reuse_temporaries in
                            bool_sweep(config.autotune, config.reuse_temporaries)
                                .iter()
                                .copied()
                        {
                            for workgroup_size in config.workgroups.iter().copied() {
                                for memory_layout in memory_layouts(program.memories.is_empty()) {
                                    let options = GpuBatchOptions {
                                        schedule: PackedScheduleOptions {
                                            max_packet_width: schedule_cap,
                                            max_memory_reads_per_packet: mem_read_cap,
                                            liveness_priority,
                                        },
                                        memory_layout: *memory_layout,
                                        workgroup_size,
                                        reuse_temporaries,
                                    };
                                    let stats = gpu_shader_stats(&program, options).unwrap();
                                    let mut samples = TimingSamples::default();
                                    for _ in 0..config.repeat {
                                        let cpu = time_cpu(case, &inputs, lanes, steps);
                                        let packed = time_packed(
                                            case,
                                            &inputs,
                                            program.clone(),
                                            lanes,
                                            steps,
                                        );
                                        assert_eq!(packed.checksum, cpu.checksum);
                                        let gpu_tick = time_gpu(
                                            case,
                                            &inputs,
                                            lanes,
                                            steps,
                                            options,
                                            GpuMode::TickLoop,
                                        );
                                        let gpu_tick_many = time_gpu(
                                            case,
                                            &inputs,
                                            lanes,
                                            steps,
                                            options,
                                            GpuMode::TickMany,
                                        );

                                        if let Some(gpu) = &gpu_tick {
                                            assert_eq!(gpu.checksum, cpu.checksum);
                                        }
                                        if let Some(gpu) = &gpu_tick_many {
                                            assert_eq!(gpu.checksum, cpu.checksum);
                                        }

                                        samples.cpu.push(cpu);
                                        samples.packed.push(packed);
                                        samples.gpu_tick.push(gpu_tick);
                                        samples.gpu_tick_many.push(gpu_tick_many);
                                    }

                                    rows.push(BenchRow::new(
                                        case,
                                        lanes,
                                        steps,
                                        schedule_cap,
                                        &stats,
                                        config.repeat,
                                        &samples,
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn bool_sweep(autotune: bool, configured: bool) -> &'static [bool] {
    if autotune {
        &[false, true]
    } else if configured {
        &[true]
    } else {
        &[false]
    }
}

fn memory_layouts(no_memories: bool) -> &'static [GpuMemoryLayout] {
    if no_memories {
        &MEMORY_LAYOUTS[..1]
    } else {
        &MEMORY_LAYOUTS
    }
}

fn counter_case() -> BenchCase {
    let mut design = Design::new();
    let (en, count);
    {
        let mut m = design.module("CounterBench");
        let clk = m.input("clk", uint(1));
        en = m.input("en", uint(1));
        count = m.reg("count", uint(32));
        m.clock(count, clk);
        m.next(count, mux(en, count + lit_u(1, 32), count));
    }
    BenchCase {
        name: "counter",
        module: "CounterBench",
        design,
        inputs: vec![BenchInput {
            signal: en,
            seed: 1,
            stride: 0,
        }],
        output: count,
    }
}

fn wide_datapath_case() -> BenchCase {
    let mut design = Design::new();
    let (i0, i1, i2, i3, i4, i5, i6, i7, y);
    {
        let mut m = design.module("WideDatapathBench");
        i0 = m.input("i0", uint(32));
        i1 = m.input("i1", uint(32));
        i2 = m.input("i2", uint(32));
        i3 = m.input("i3", uint(32));
        i4 = m.input("i4", uint(32));
        i5 = m.input("i5", uint(32));
        i6 = m.input("i6", uint(32));
        i7 = m.input("i7", uint(32));
        y = m.output("y", uint(32));
        let a = (i0 + i1) ^ (i2 + i3);
        let b = (i4 + i5) ^ (i6 + i7);
        let c = concat([i0.value().slice(0, 8), i7.value().slice(8, 24)]);
        m.assign(y, mux(i0.value().eq_expr(i1), a.clone() + b, c + a));
    }
    BenchCase {
        name: "wide_datapath",
        module: "WideDatapathBench",
        design,
        inputs: vec![
            BenchInput {
                signal: i0,
                seed: 1,
                stride: 3,
            },
            BenchInput {
                signal: i1,
                seed: 2,
                stride: 5,
            },
            BenchInput {
                signal: i2,
                seed: 3,
                stride: 7,
            },
            BenchInput {
                signal: i3,
                seed: 4,
                stride: 11,
            },
            BenchInput {
                signal: i4,
                seed: 5,
                stride: 13,
            },
            BenchInput {
                signal: i5,
                seed: 6,
                stride: 17,
            },
            BenchInput {
                signal: i6,
                seed: 7,
                stride: 19,
            },
            BenchInput {
                signal: i7,
                seed: 8,
                stride: 23,
            },
        ],
        output: y,
    }
}

fn register_file_case() -> BenchCase {
    let mut design = Design::new();
    let (we, addr, data, read);
    {
        let mut m = design.module("RegisterFileBench");
        let clk = m.input("clk", uint(1));
        we = m.input("we", uint(1));
        addr = m.input("addr", uint(4));
        data = m.input("data", uint(32));
        let mem = m.mem("regs", 4, uint(32), 16);
        read = m.output("read", uint(32));
        let read_expr = m.mem_read(mem, addr);
        m.assign(read, read_expr);
        m.mem_write(mem, clk, we, addr, data);
    }
    BenchCase {
        name: "register_file",
        module: "RegisterFileBench",
        design,
        inputs: vec![
            BenchInput {
                signal: we,
                seed: 1,
                stride: 0,
            },
            BenchInput {
                signal: addr,
                seed: 0,
                stride: 1,
            },
            BenchInput {
                signal: data,
                seed: 0x100,
                stride: 0x11,
            },
        ],
        output: read,
    }
}

fn fifo_like_case() -> BenchCase {
    let mut design = Design::new();
    let (push, data, read);
    {
        let mut m = design.module("FifoLikeBench");
        let clk = m.input("clk", uint(1));
        push = m.input("push", uint(1));
        data = m.input("data", uint(32));
        let wr_ptr = m.reg("wr_ptr", uint(4));
        let rd_ptr = m.reg("rd_ptr", uint(4));
        let mem = m.mem("fifo_mem", 4, uint(32), 16);
        read = m.output("read", uint(32));
        m.clock(wr_ptr, clk);
        m.clock(rd_ptr, clk);
        let read_expr = m.mem_read(mem, rd_ptr);
        m.assign(read, read_expr);
        m.next(wr_ptr, mux(push, wr_ptr + lit_u(1, 4), wr_ptr));
        m.next(rd_ptr, rd_ptr + lit_u(1, 4));
        m.mem_write(mem, clk, push, wr_ptr, data);
    }
    BenchCase {
        name: "fifo_like",
        module: "FifoLikeBench",
        design,
        inputs: vec![
            BenchInput {
                signal: push,
                seed: 1,
                stride: 0,
            },
            BenchInput {
                signal: data,
                seed: 0x200,
                stride: 0x21,
            },
        ],
        output: read,
    }
}

fn mixed_memory_datapath_case() -> BenchCase {
    let mut design = Design::new();
    let (we, addr, a, b, y);
    {
        let mut m = design.module("MixedMemoryDatapathBench");
        let clk = m.input("clk", uint(1));
        we = m.input("we", uint(1));
        addr = m.input("addr", uint(3));
        a = m.input("a", uint(32));
        b = m.input("b", uint(32));
        let mem = m.mem("scratch", 3, uint(32), 8);
        y = m.output("y", uint(32));
        let read_expr = m.mem_read(mem, addr);
        let mixed =
            ((a + b) ^ concat([a.value().slice(0, 16), b.value().slice(16, 16)])) + read_expr;
        m.assign(y, mixed.clone());
        m.mem_write(mem, clk, we, addr, mixed);
    }
    BenchCase {
        name: "mixed_memory_datapath",
        module: "MixedMemoryDatapathBench",
        design,
        inputs: vec![
            BenchInput {
                signal: we,
                seed: 1,
                stride: 0,
            },
            BenchInput {
                signal: addr,
                seed: 0,
                stride: 1,
            },
            BenchInput {
                signal: a,
                seed: 0x300,
                stride: 0x31,
            },
            BenchInput {
                signal: b,
                seed: 0x500,
                stride: 0x41,
            },
        ],
        output: y,
    }
}

fn multi_read_memory_case() -> BenchCase {
    let mut design = Design::new();
    let (we, addr0, addr1, addr2, waddr, data, salt, y);
    {
        let mut m = design.module("MultiReadMemoryBench");
        let clk = m.input("clk", uint(1));
        we = m.input("we", uint(1));
        addr0 = m.input("addr0", uint(4));
        addr1 = m.input("addr1", uint(4));
        addr2 = m.input("addr2", uint(4));
        waddr = m.input("waddr", uint(4));
        data = m.input("data", uint(32));
        salt = m.input("salt", uint(32));
        let mem = m.mem("bank", 4, uint(32), 16);
        y = m.output("y", uint(32));
        let r0 = m.mem_read(mem, addr0);
        let r1 = m.mem_read(mem, addr1);
        let r2 = m.mem_read(mem, addr2);
        let folded = ((r0.clone() + r1.clone()) ^ r2.clone())
            + (data ^ concat([salt.value().slice(0, 12), r1.clone().slice(12, 20)]));
        let mixed = mux(we, folded.clone(), folded ^ r0);
        m.assign(y, mixed.clone());
        m.mem_write(mem, clk, we, waddr, mixed);
    }
    BenchCase {
        name: "multi_read_memory",
        module: "MultiReadMemoryBench",
        design,
        inputs: vec![
            BenchInput {
                signal: we,
                seed: 1,
                stride: 0,
            },
            BenchInput {
                signal: addr0,
                seed: 0,
                stride: 1,
            },
            BenchInput {
                signal: addr1,
                seed: 3,
                stride: 5,
            },
            BenchInput {
                signal: addr2,
                seed: 7,
                stride: 9,
            },
            BenchInput {
                signal: waddr,
                seed: 11,
                stride: 13,
            },
            BenchInput {
                signal: data,
                seed: 0x700,
                stride: 0x51,
            },
            BenchInput {
                signal: salt,
                seed: 0x900,
                stride: 0x61,
            },
        ],
        output: y,
    }
}

fn deep_mixed_pipeline_case() -> BenchCase {
    let mut design = Design::new();
    let (en, sel, a, b, c, d, y);
    {
        let mut m = design.module("DeepMixedPipelineBench");
        let clk = m.input("clk", uint(1));
        en = m.input("en", uint(1));
        sel = m.input("sel", uint(1));
        a = m.input("a", uint(32));
        b = m.input("b", uint(32));
        c = m.input("c", uint(32));
        d = m.input("d", uint(32));
        let r0 = m.reg("r0", uint(32));
        let r1 = m.reg("r1", uint(32));
        let r2 = m.reg("r2", uint(32));
        let r3 = m.reg("r3", uint(32));
        let r4 = m.reg("r4", uint(32));
        let r5 = m.reg("r5", uint(32));
        y = m.output("y", uint(32));
        for reg in [r0, r1, r2, r3, r4, r5] {
            m.clock(reg, clk);
        }
        let mix0 = (a + r0) ^ (b + r1);
        let mix1 = concat([mix0.clone().slice(0, 16), (c ^ r2).slice(16, 16)]);
        let mix2 = mux(
            sel,
            mix1.clone() + d,
            (mix1.clone() ^ r3) + lit_u(0x9e37, 32),
        );
        let mix3 =
            (mix2.clone() ^ r4) + concat([r5.value().slice(8, 16), mix2.clone().slice(0, 16)]);
        let feedback = mux(en, mix3.clone(), r5);
        m.next(r0, mux(en, a + lit_u(1, 32), r0));
        m.next(r1, mux(en, mix0, r1));
        m.next(r2, mux(en, mix1, r2));
        m.next(r3, mux(en, mix2, r3));
        m.next(r4, mux(en, mix3.clone(), r4));
        m.next(r5, feedback.clone());
        m.assign(y, feedback ^ r2);
    }
    BenchCase {
        name: "deep_mixed_pipeline",
        module: "DeepMixedPipelineBench",
        design,
        inputs: vec![
            BenchInput {
                signal: en,
                seed: 1,
                stride: 0,
            },
            BenchInput {
                signal: sel,
                seed: 0,
                stride: 1,
            },
            BenchInput {
                signal: a,
                seed: 0xb00,
                stride: 0x71,
            },
            BenchInput {
                signal: b,
                seed: 0xd00,
                stride: 0x83,
            },
            BenchInput {
                signal: c,
                seed: 0x1100,
                stride: 0x95,
            },
            BenchInput {
                signal: d,
                seed: 0x1300,
                stride: 0xa7,
            },
        ],
        output: y,
    }
}

fn materialize_inputs(case: &BenchCase, lanes: usize) -> Vec<(Signal, Vec<u128>)> {
    case.inputs
        .iter()
        .map(|input| (input.signal, lane_values(lanes, input.seed, input.stride)))
        .collect()
}

fn lane_values(lanes: usize, seed: u128, stride: u128) -> Vec<u128> {
    (0..lanes)
        .map(|lane| seed.wrapping_add(stride.wrapping_mul(lane as u128)) & 0xffff_ffff)
        .collect()
}

fn time_cpu(
    case: &BenchCase,
    inputs: &[(Signal, Vec<u128>)],
    lanes: usize,
    steps: usize,
) -> TimedChecksum {
    let start = Instant::now();
    let mut sims = (0..lanes)
        .map(|lane| {
            let mut sim = Simulator::new(&case.design, case.module).unwrap();
            for (signal, values) in inputs {
                sim.set(*signal, values[lane]);
            }
            sim
        })
        .collect::<Vec<_>>();
    for _ in 0..steps {
        for sim in &mut sims {
            sim.tick();
        }
    }
    let checksum = sims
        .iter()
        .fold(0u128, |acc, sim| acc ^ sim.get(case.output));
    TimedChecksum {
        duration: start.elapsed(),
        checksum,
    }
}

fn time_packed(
    case: &BenchCase,
    inputs: &[(Signal, Vec<u128>)],
    program: rrtl_sim_ir::PackedProgram,
    lanes: usize,
    steps: usize,
) -> TimedChecksum {
    let start = Instant::now();
    let mut sim = PackedSimulator::new(program, lanes).unwrap();
    for (signal, values) in inputs {
        sim.set_signal(*signal, values).unwrap();
    }
    for _ in 0..steps {
        sim.tick();
    }
    let checksum = sim
        .get_signal(case.output)
        .unwrap()
        .into_iter()
        .fold(0u128, |acc, value| acc ^ value);
    TimedChecksum {
        duration: start.elapsed(),
        checksum,
    }
}

fn time_gpu(
    case: &BenchCase,
    inputs: &[(Signal, Vec<u128>)],
    lanes: usize,
    steps: usize,
    options: GpuBatchOptions,
    mode: GpuMode,
) -> Option<GpuTiming> {
    let total_start = Instant::now();
    let construct_start = Instant::now();
    let mut sim =
        GpuBatchSimulator::new_with_options(&case.design, case.module, lanes, options).ok()?;
    let construct = construct_start.elapsed();

    for (signal, values) in inputs {
        let values = values.iter().map(|value| *value as u32).collect::<Vec<_>>();
        sim.set_input(*signal, &values).unwrap();
    }

    let run_start = Instant::now();
    match mode {
        GpuMode::TickLoop => {
            for _ in 0..steps {
                sim.tick().unwrap();
            }
        }
        GpuMode::TickMany => sim.tick_many(steps).unwrap(),
    }
    let run = run_start.elapsed();

    let checksum = sim
        .get_signal(case.output)
        .unwrap()
        .into_iter()
        .fold(0u128, |acc, value| acc ^ value as u128);
    Some(GpuTiming {
        construct,
        run,
        total: total_start.elapsed(),
        checksum,
    })
}

fn print_csv_header(out: &mut impl Write) {
    writeln!(
        out,
        "case,lanes,steps,schedule_cap,memory_read_cap,liveness_priority,reuse_temporaries,memory_layout,gpu_mode,workgroup_size,wgsl_bytes,optimized_temp_slots,optimized_value_vars,packets_total,packets_async_reset_comb,packets_comb,packets_tick_next,packets_tick_commit,max_packet_width,max_live_values,avg_live_values_x100,unoptimized_packets_total,unoptimized_max_packet_width,unoptimized_max_live_values,unoptimized_avg_live_values_x100,optimized_vs_unoptimized_packets_delta,optimized_vs_unoptimized_packets_delta_pct_x100,packet_utilization_x100,max_packet_memory_reads,memory_reads,memory_writes,total_memory_words_per_lane,repeat,cpu_ns,cpu_ns_min,cpu_ns_max,packed_ns,packed_ns_min,packed_ns_max,gpu_tick_construct_ns,gpu_tick_construct_ns_min,gpu_tick_construct_ns_max,gpu_tick_ns,gpu_tick_ns_min,gpu_tick_ns_max,gpu_tick_total_ns,gpu_tick_total_ns_min,gpu_tick_total_ns_max,gpu_tick_many_construct_ns,gpu_tick_many_construct_ns_min,gpu_tick_many_construct_ns_max,gpu_tick_many_ns,gpu_tick_many_ns_min,gpu_tick_many_ns_max,gpu_tick_many_total_ns,gpu_tick_many_total_ns_min,gpu_tick_many_total_ns_max,cpu_checksum,packed_checksum,gpu_tick_checksum,gpu_tick_many_checksum,gpu_available,autotune_rank,autotune_metric,autotune_metric_ns,autotune_best"
    )
    .unwrap();
    out.flush().unwrap();
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TimingStats {
    median: u128,
    min: u128,
    max: u128,
}

impl BenchRow {
    fn new(
        case: &BenchCase,
        lanes: usize,
        steps: usize,
        schedule_cap: Option<usize>,
        stats: &rrtl_gpu_sim::GpuShaderStats,
        repeat: usize,
        samples: &TimingSamples,
    ) -> Self {
        let cpu_timing = timing_stats(
            samples
                .cpu
                .iter()
                .map(|sample| duration_ns(sample.duration)),
        );
        let packed_timing = timing_stats(
            samples
                .packed
                .iter()
                .map(|sample| duration_ns(sample.duration)),
        );
        let gpu_tick_construct = optional_timing_stats(
            samples
                .gpu_tick
                .iter()
                .filter_map(|sample| sample.as_ref().map(|gpu| duration_ns(gpu.construct))),
        );
        let gpu_tick_run = optional_timing_stats(
            samples
                .gpu_tick
                .iter()
                .filter_map(|sample| sample.as_ref().map(|gpu| duration_ns(gpu.run))),
        );
        let gpu_tick_total = optional_timing_stats(
            samples
                .gpu_tick
                .iter()
                .filter_map(|sample| sample.as_ref().map(|gpu| duration_ns(gpu.total))),
        );
        let gpu_tick_many_construct = optional_timing_stats(
            samples
                .gpu_tick_many
                .iter()
                .filter_map(|sample| sample.as_ref().map(|gpu| duration_ns(gpu.construct))),
        );
        let gpu_tick_many_run = optional_timing_stats(
            samples
                .gpu_tick_many
                .iter()
                .filter_map(|sample| sample.as_ref().map(|gpu| duration_ns(gpu.run))),
        );
        let gpu_tick_many_total = optional_timing_stats(
            samples
                .gpu_tick_many
                .iter()
                .filter_map(|sample| sample.as_ref().map(|gpu| duration_ns(gpu.total))),
        );
        Self {
            case: case.name.to_string(),
            lanes,
            steps,
            schedule_cap,
            memory_read_cap: stats.schedule.max_memory_reads_per_packet,
            liveness_priority: stats.schedule.liveness_priority,
            reuse_temporaries: stats.reuse_temporaries,
            memory_layout: format_memory_layout(stats.memory_layout).to_string(),
            gpu_mode: "shader_loop_tick_many".to_string(),
            workgroup_size: stats.workgroup_size,
            wgsl_bytes: stats.wgsl_bytes,
            optimized_temp_slots: stats.optimized_temp_slots,
            optimized_value_vars: stats.optimized_value_vars,
            packets_total: stats.optimized_packets.total,
            packets_async_reset_comb: stats.optimized_packets.async_reset_comb,
            packets_comb: stats.optimized_packets.comb,
            packets_tick_next: stats.optimized_packets.tick_next,
            packets_tick_commit: stats.optimized_packets.tick_commit,
            max_packet_width: stats.optimized.max_packet_width,
            max_live_values: stats.optimized.max_live_values,
            avg_live_values_x100: stats.optimized.avg_live_values_x100,
            unoptimized_packets_total: stats.unoptimized_packets.total,
            unoptimized_max_packet_width: stats.unoptimized.max_packet_width,
            unoptimized_max_live_values: stats.unoptimized.max_live_values,
            unoptimized_avg_live_values_x100: stats.unoptimized.avg_live_values_x100,
            optimized_vs_unoptimized_packets_delta: packet_delta(
                stats.optimized_packets.total,
                stats.unoptimized_packets.total,
            ),
            optimized_vs_unoptimized_packets_delta_pct_x100: packet_delta_pct_x100(
                stats.optimized_packets.total,
                stats.unoptimized_packets.total,
            ),
            packet_utilization_x100: packet_utilization_x100(
                stats.optimized.instr_count,
                stats.optimized_packets.total,
                stats.optimized.max_packet_width,
            ),
            max_packet_memory_reads: stats.optimized.max_packet_memory_reads,
            memory_reads: stats.optimized_memory.total_reads,
            memory_writes: stats.optimized_memory.total_writes,
            total_memory_words_per_lane: stats.total_memory_words_per_lane,
            repeat,
            cpu_ns: cpu_timing.median,
            cpu_ns_min: cpu_timing.min,
            cpu_ns_max: cpu_timing.max,
            packed_ns: packed_timing.median,
            packed_ns_min: packed_timing.min,
            packed_ns_max: packed_timing.max,
            gpu_tick_construct_ns: gpu_tick_construct.map(|stats| stats.median),
            gpu_tick_construct_ns_min: gpu_tick_construct.map(|stats| stats.min),
            gpu_tick_construct_ns_max: gpu_tick_construct.map(|stats| stats.max),
            gpu_tick_ns: gpu_tick_run.map(|stats| stats.median),
            gpu_tick_ns_min: gpu_tick_run.map(|stats| stats.min),
            gpu_tick_ns_max: gpu_tick_run.map(|stats| stats.max),
            gpu_tick_total_ns: gpu_tick_total.map(|stats| stats.median),
            gpu_tick_total_ns_min: gpu_tick_total.map(|stats| stats.min),
            gpu_tick_total_ns_max: gpu_tick_total.map(|stats| stats.max),
            gpu_tick_many_construct_ns: gpu_tick_many_construct.map(|stats| stats.median),
            gpu_tick_many_construct_ns_min: gpu_tick_many_construct.map(|stats| stats.min),
            gpu_tick_many_construct_ns_max: gpu_tick_many_construct.map(|stats| stats.max),
            gpu_tick_many_ns: gpu_tick_many_run.map(|stats| stats.median),
            gpu_tick_many_ns_min: gpu_tick_many_run.map(|stats| stats.min),
            gpu_tick_many_ns_max: gpu_tick_many_run.map(|stats| stats.max),
            gpu_tick_many_total_ns: gpu_tick_many_total.map(|stats| stats.median),
            gpu_tick_many_total_ns_min: gpu_tick_many_total.map(|stats| stats.min),
            gpu_tick_many_total_ns_max: gpu_tick_many_total.map(|stats| stats.max),
            cpu_checksum: checksum(samples.cpu[0].checksum),
            packed_checksum: checksum(samples.packed[0].checksum),
            gpu_tick_checksum: samples
                .gpu_tick
                .iter()
                .find_map(|sample| sample.as_ref().map(|gpu| checksum(gpu.checksum))),
            gpu_tick_many_checksum: samples
                .gpu_tick_many
                .iter()
                .find_map(|sample| sample.as_ref().map(|gpu| checksum(gpu.checksum))),
            gpu_available: samples
                .gpu_tick
                .iter()
                .chain(samples.gpu_tick_many.iter())
                .any(Option::is_some),
            packed_vs_cpu: ratio(packed_timing.median, cpu_timing.median),
            gpu_tick_many_vs_packed: gpu_tick_many_run
                .and_then(|gpu_stats| ratio(gpu_stats.median, packed_timing.median)),
            autotune_rank: None,
            autotune_metric: "none".to_string(),
            autotune_metric_ns: None,
            autotune_best: false,
        }
    }

    fn key(&self) -> String {
        format!(
            "{} lanes={} steps={} cap={} mem_read_cap={} live_prio={} reuse_temps={} layout={} wg={} autotune_metric={}",
            self.case,
            self.lanes,
            self.steps,
            format_schedule_cap(self.schedule_cap),
            format_schedule_cap(self.memory_read_cap),
            self.liveness_priority,
            self.reuse_temporaries,
            self.memory_layout,
            self.workgroup_size,
            self.autotune_metric
        )
    }
}

fn rank_autotune_rows(rows: &mut [BenchRow], metric: AutotuneMetric) {
    let metric_name = format_autotune_metric(metric).to_string();
    for row in rows.iter_mut() {
        row.autotune_rank = None;
        row.autotune_metric = metric_name.clone();
        row.autotune_metric_ns = Some(autotune_metric_ns(row, metric));
        row.autotune_best = false;
    }

    let mut groups = std::collections::BTreeMap::<(String, usize, usize), Vec<usize>>::new();
    for (index, row) in rows.iter().enumerate() {
        groups
            .entry((row.case.clone(), row.lanes, row.steps))
            .or_default()
            .push(index);
    }

    for indices in groups.values_mut() {
        indices.sort_by(|lhs, rhs| {
            rows[*lhs]
                .autotune_metric_ns
                .cmp(&rows[*rhs].autotune_metric_ns)
                .then_with(|| rows[*lhs].key().cmp(&rows[*rhs].key()))
        });
        for (rank, index) in indices.iter().copied().enumerate() {
            rows[index].autotune_rank = Some(rank + 1);
            rows[index].autotune_best = rank == 0;
        }
    }
}

fn autotune_metric_ns(row: &BenchRow, metric: AutotuneMetric) -> u128 {
    match metric {
        AutotuneMetric::Packed => row.packed_ns,
        AutotuneMetric::GpuTick => row.gpu_tick_ns.unwrap_or(row.packed_ns),
        AutotuneMetric::GpuTickMany => row.gpu_tick_many_ns.unwrap_or(row.packed_ns),
    }
}

fn print_rows(out: &mut impl Write, format: OutputFormat, rows: &[BenchRow]) {
    match format {
        OutputFormat::Csv => {
            print_csv_header(out);
            for row in rows {
                print_csv_row(out, row);
            }
        }
        OutputFormat::Human => {
            for row in human_ordered_rows(rows) {
                print_human_row(out, row);
            }
        }
        OutputFormat::Json => {
            serde_json::to_writer_pretty(&mut *out, rows).unwrap();
            writeln!(out).unwrap();
        }
    }
    out.flush().unwrap();
}

fn print_recommended_configs(out: &mut impl Write, rows: &[BenchRow]) {
    let recommendations = recommended_configs(rows);
    serde_json::to_writer_pretty(&mut *out, &recommendations).unwrap();
    writeln!(out).unwrap();
    out.flush().unwrap();
}

fn recommended_configs(rows: &[BenchRow]) -> Vec<GpuAutotuneRecommendation> {
    let mut configs = rows
        .iter()
        .filter(|row| row.autotune_best)
        .map(row_to_gpu_autotune_recommendation)
        .collect::<Vec<_>>();
    configs.sort_by(|lhs, rhs| {
        lhs.case
            .cmp(&rhs.case)
            .then_with(|| lhs.lanes.cmp(&rhs.lanes))
            .then_with(|| lhs.steps.cmp(&rhs.steps))
    });
    configs
}

fn row_to_gpu_autotune_recommendation(row: &BenchRow) -> GpuAutotuneRecommendation {
    GpuAutotuneRecommendation {
        case: row.case.clone(),
        lanes: row.lanes,
        steps: row.steps,
        schedule_cap: row.schedule_cap,
        memory_read_cap: row.memory_read_cap,
        liveness_priority: row.liveness_priority,
        reuse_temporaries: row.reuse_temporaries,
        memory_layout: row.memory_layout.clone(),
        workgroup_size: row.workgroup_size,
        autotune_metric: row.autotune_metric.clone(),
        autotune_metric_ns: row.autotune_metric_ns,
        packed_ns: row.packed_ns,
        gpu_tick_ns: row.gpu_tick_ns,
        gpu_tick_many_ns: row.gpu_tick_many_ns,
    }
}

fn human_ordered_rows(rows: &[BenchRow]) -> Vec<&BenchRow> {
    let mut ordered = rows.iter().collect::<Vec<_>>();
    if rows.iter().any(|row| row.autotune_rank.is_some()) {
        ordered.sort_by(|lhs, rhs| {
            lhs.autotune_rank
                .unwrap_or(usize::MAX)
                .cmp(&rhs.autotune_rank.unwrap_or(usize::MAX))
                .then_with(|| lhs.case.cmp(&rhs.case))
                .then_with(|| lhs.lanes.cmp(&rhs.lanes))
                .then_with(|| lhs.steps.cmp(&rhs.steps))
                .then_with(|| lhs.key().cmp(&rhs.key()))
        });
    }
    ordered
}

fn print_csv_row(out: &mut impl Write, row: &BenchRow) {
    let fields = vec![
        row.case.clone(),
        row.lanes.to_string(),
        row.steps.to_string(),
        format_schedule_cap(row.schedule_cap),
        format_schedule_cap(row.memory_read_cap),
        row.liveness_priority.to_string(),
        row.reuse_temporaries.to_string(),
        row.memory_layout.clone(),
        row.gpu_mode.clone(),
        row.workgroup_size.to_string(),
        row.wgsl_bytes.to_string(),
        row.optimized_temp_slots.to_string(),
        row.optimized_value_vars.to_string(),
        row.packets_total.to_string(),
        row.packets_async_reset_comb.to_string(),
        row.packets_comb.to_string(),
        row.packets_tick_next.to_string(),
        row.packets_tick_commit.to_string(),
        row.max_packet_width.to_string(),
        row.max_live_values.to_string(),
        row.avg_live_values_x100.to_string(),
        row.unoptimized_packets_total.to_string(),
        row.unoptimized_max_packet_width.to_string(),
        row.unoptimized_max_live_values.to_string(),
        row.unoptimized_avg_live_values_x100.to_string(),
        row.optimized_vs_unoptimized_packets_delta.to_string(),
        row.optimized_vs_unoptimized_packets_delta_pct_x100
            .to_string(),
        row.packet_utilization_x100.to_string(),
        row.max_packet_memory_reads.to_string(),
        row.memory_reads.to_string(),
        row.memory_writes.to_string(),
        row.total_memory_words_per_lane.to_string(),
        row.repeat.to_string(),
        row.cpu_ns.to_string(),
        row.cpu_ns_min.to_string(),
        row.cpu_ns_max.to_string(),
        row.packed_ns.to_string(),
        row.packed_ns_min.to_string(),
        row.packed_ns_max.to_string(),
        maybe_ns(row.gpu_tick_construct_ns),
        maybe_ns(row.gpu_tick_construct_ns_min),
        maybe_ns(row.gpu_tick_construct_ns_max),
        maybe_ns(row.gpu_tick_ns),
        maybe_ns(row.gpu_tick_ns_min),
        maybe_ns(row.gpu_tick_ns_max),
        maybe_ns(row.gpu_tick_total_ns),
        maybe_ns(row.gpu_tick_total_ns_min),
        maybe_ns(row.gpu_tick_total_ns_max),
        maybe_ns(row.gpu_tick_many_construct_ns),
        maybe_ns(row.gpu_tick_many_construct_ns_min),
        maybe_ns(row.gpu_tick_many_construct_ns_max),
        maybe_ns(row.gpu_tick_many_ns),
        maybe_ns(row.gpu_tick_many_ns_min),
        maybe_ns(row.gpu_tick_many_ns_max),
        maybe_ns(row.gpu_tick_many_total_ns),
        maybe_ns(row.gpu_tick_many_total_ns_min),
        maybe_ns(row.gpu_tick_many_total_ns_max),
        row.cpu_checksum.clone(),
        row.packed_checksum.clone(),
        row.gpu_tick_checksum.clone().unwrap_or_default(),
        row.gpu_tick_many_checksum.clone().unwrap_or_default(),
        row.gpu_available.to_string(),
        row.autotune_rank
            .map(|rank| rank.to_string())
            .unwrap_or_default(),
        row.autotune_metric.clone(),
        maybe_ns(row.autotune_metric_ns),
        row.autotune_best.to_string(),
    ];
    writeln!(out, "{}", fields.join(",")).unwrap();
}

fn print_human_row(out: &mut impl Write, row: &BenchRow) {
    writeln!(
        out,
        "case={} lanes={} steps={} repeat={} cap={} mem_read_cap={} liveness_priority={} reuse_temporaries={} memory_layout={} autotune_rank={} autotune_best={} autotune_metric={} autotune_metric_ns={} packets={} streams=async:{} comb:{} next:{} commit:{} wgsl_bytes={} temp_slots={} value_vars={} max_width={} max_live={} avg_live_x100={} unopt_packets={} unopt_max_width={} unopt_max_live={} unopt_avg_live_x100={} packet_delta={} packet_delta_pct_x100={} packet_util_x100={} max_mem_reads={} mem_reads={} mem_writes={} mem_words_per_lane={}",
        row.case,
        row.lanes,
        row.steps,
        row.repeat,
        format_schedule_cap(row.schedule_cap),
        format_schedule_cap(row.memory_read_cap),
        row.liveness_priority,
        row.reuse_temporaries,
        row.memory_layout,
        row.autotune_rank
            .map(|rank| rank.to_string())
            .unwrap_or_else(|| "none".to_string()),
        row.autotune_best,
        row.autotune_metric,
        maybe_ns(row.autotune_metric_ns),
        row.packets_total,
        row.packets_async_reset_comb,
        row.packets_comb,
        row.packets_tick_next,
        row.packets_tick_commit,
        row.wgsl_bytes,
        row.optimized_temp_slots,
        row.optimized_value_vars,
        row.max_packet_width,
        row.max_live_values,
        row.avg_live_values_x100,
        row.unoptimized_packets_total,
        row.unoptimized_max_packet_width,
        row.unoptimized_max_live_values,
        row.unoptimized_avg_live_values_x100,
        row.optimized_vs_unoptimized_packets_delta,
        row.optimized_vs_unoptimized_packets_delta_pct_x100,
        row.packet_utilization_x100,
        row.max_packet_memory_reads,
        row.memory_reads,
        row.memory_writes,
        row.total_memory_words_per_lane
    )
    .unwrap();
    writeln!(
        out,
        "  cpu={} min:{} max:{} packed={} min:{} max:{} checksums cpu={} packed={}",
        row.cpu_ns,
        row.cpu_ns_min,
        row.cpu_ns_max,
        row.packed_ns,
        row.packed_ns_min,
        row.packed_ns_max,
        row.cpu_checksum,
        row.packed_checksum
    )
    .unwrap();
    writeln!(
        out,
        "  gpu_tick={} gpu_tick_many={}",
        format_gpu_timing(
            row.gpu_tick_construct_ns,
            row.gpu_tick_ns,
            row.gpu_tick_total_ns,
            row.gpu_tick_checksum.as_deref()
        ),
        format_gpu_timing(
            row.gpu_tick_many_construct_ns,
            row.gpu_tick_many_ns,
            row.gpu_tick_many_total_ns,
            row.gpu_tick_many_checksum.as_deref()
        )
    )
    .unwrap();
}

fn read_json_rows(path: &str) -> Vec<BenchRow> {
    let file = File::open(path).unwrap();
    serde_json::from_reader(file).unwrap()
}

fn compare_exit_code(strict: bool, issues: usize) -> i32 {
    if strict && issues > 0 {
        1
    } else {
        0
    }
}

fn print_compare_report(
    out: &mut impl Write,
    baseline: &[BenchRow],
    current: &[BenchRow],
    timing_threshold_pct: f64,
) -> usize {
    let baseline_by_key = baseline
        .iter()
        .map(|row| (row.key(), row))
        .collect::<std::collections::HashMap<_, _>>();
    let mut issue_count = 0usize;

    writeln!(
        out,
        "compare baseline_rows={} current_rows={}",
        baseline.len(),
        current.len()
    )
    .unwrap();
    for row in current {
        let key = row.key();
        let Some(base) = baseline_by_key.get(&key) else {
            issue_count += 1;
            writeln!(out, "new row: {key}").unwrap();
            continue;
        };
        if base.cpu_checksum != row.cpu_checksum
            || base.packed_checksum != row.packed_checksum
            || base.gpu_tick_checksum != row.gpu_tick_checksum
            || base.gpu_tick_many_checksum != row.gpu_tick_many_checksum
        {
            issue_count += 1;
            writeln!(out, "checksum mismatch: {key}").unwrap();
        }
        if row.packets_total > base.packets_total {
            issue_count += 1;
            writeln!(
                out,
                "packet regression: {key} {} -> {}",
                base.packets_total, row.packets_total
            )
            .unwrap();
        }
        if row.wgsl_bytes > base.wgsl_bytes {
            issue_count += 1;
            writeln!(
                out,
                "wgsl regression: {key} {} -> {}",
                base.wgsl_bytes, row.wgsl_bytes
            )
            .unwrap();
        }
        report_timing_regression(
            out,
            "cpu_ns",
            &key,
            base.cpu_ns,
            row.cpu_ns,
            timing_threshold_pct,
            &mut issue_count,
        );
        report_timing_regression(
            out,
            "packed_ns",
            &key,
            base.packed_ns,
            row.packed_ns,
            timing_threshold_pct,
            &mut issue_count,
        );
        if let (Some(base_ns), Some(current_ns)) = (base.gpu_tick_many_ns, row.gpu_tick_many_ns) {
            report_timing_regression(
                out,
                "gpu_tick_many_ns",
                &key,
                base_ns,
                current_ns,
                timing_threshold_pct,
                &mut issue_count,
            );
        }
    }
    writeln!(
        out,
        "compare issues={issue_count} timing_threshold_pct={timing_threshold_pct}"
    )
    .unwrap();
    out.flush().unwrap();
    issue_count
}

fn report_timing_regression(
    out: &mut impl Write,
    name: &str,
    key: &str,
    baseline: u128,
    current: u128,
    threshold_pct: f64,
    issue_count: &mut usize,
) {
    if timing_regressed(baseline, current, threshold_pct) {
        *issue_count += 1;
        writeln!(
            out,
            "timing regression: {key} {name} {baseline} -> {current} threshold_pct={threshold_pct}"
        )
        .unwrap();
    }
}

fn timing_regressed(baseline: u128, current: u128, threshold_pct: f64) -> bool {
    if baseline == 0 {
        return current > 0;
    }
    let threshold = baseline as f64 * (1.0 + threshold_pct / 100.0);
    current as f64 > threshold
}

fn format_gpu_timing(
    construct: Option<u128>,
    run: Option<u128>,
    total: Option<u128>,
    checksum: Option<&str>,
) -> String {
    match (construct, run, total, checksum) {
        (Some(construct), Some(run), Some(total), Some(checksum)) => format!(
            "construct:{} run:{} total:{} checksum:{}",
            construct, run, total, checksum
        ),
        _ => "unavailable".to_string(),
    }
}

fn format_schedule_cap(cap: Option<usize>) -> String {
    cap.map(|cap| cap.to_string())
        .unwrap_or_else(|| "none".to_string())
}

fn format_memory_layout(layout: GpuMemoryLayout) -> &'static str {
    match layout {
        GpuMemoryLayout::LaneMajor => "lane_major",
        GpuMemoryLayout::WordMajor => "word_major",
    }
}

fn duration_ns(duration: Duration) -> u128 {
    duration.as_nanos()
}

fn optional_timing_stats(values: impl IntoIterator<Item = u128>) -> Option<TimingStats> {
    let values = values.into_iter().collect::<Vec<_>>();
    (!values.is_empty()).then(|| timing_stats(values))
}

fn timing_stats(values: impl IntoIterator<Item = u128>) -> TimingStats {
    let mut values = values.into_iter().collect::<Vec<_>>();
    assert!(
        !values.is_empty(),
        "timing stats require at least one value"
    );
    values.sort_unstable();
    let min = values[0];
    let max = values[values.len() - 1];
    let median = if values.len() % 2 == 1 {
        values[values.len() / 2]
    } else {
        let lhs = values[values.len() / 2 - 1];
        let rhs = values[values.len() / 2];
        lhs / 2 + rhs / 2 + (lhs % 2 + rhs % 2) / 2
    };
    TimingStats { median, min, max }
}

fn ratio(numerator: u128, denominator: u128) -> Option<f64> {
    (denominator != 0).then_some(numerator as f64 / denominator as f64)
}

fn packet_delta(optimized_packets: usize, unoptimized_packets: usize) -> i128 {
    optimized_packets as i128 - unoptimized_packets as i128
}

fn packet_delta_pct_x100(optimized_packets: usize, unoptimized_packets: usize) -> i128 {
    if unoptimized_packets == 0 {
        0
    } else {
        packet_delta(optimized_packets, unoptimized_packets) * 10_000 / unoptimized_packets as i128
    }
}

fn packet_utilization_x100(
    optimized_instr_count: usize,
    optimized_packet_count: usize,
    max_packet_width: usize,
) -> usize {
    let capacity = optimized_packet_count.saturating_mul(max_packet_width);
    if capacity == 0 {
        0
    } else {
        optimized_instr_count * 100 / capacity
    }
}

fn maybe_ns(value: Option<u128>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn checksum(value: u128) -> String {
    format!("0x{value:x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    fn sample_row() -> BenchRow {
        BenchRow {
            case: "counter".to_string(),
            lanes: 64,
            steps: 16,
            schedule_cap: Some(16),
            memory_read_cap: Some(1),
            liveness_priority: true,
            reuse_temporaries: true,
            memory_layout: "lane_major".to_string(),
            gpu_mode: "shader_loop_tick_many".to_string(),
            workgroup_size: 128,
            wgsl_bytes: 100,
            optimized_temp_slots: 2,
            optimized_value_vars: 4,
            packets_total: 3,
            packets_async_reset_comb: 0,
            packets_comb: 0,
            packets_tick_next: 3,
            packets_tick_commit: 0,
            max_packet_width: 3,
            max_live_values: 3,
            avg_live_values_x100: 150,
            unoptimized_packets_total: 2,
            unoptimized_max_packet_width: 4,
            unoptimized_max_live_values: 5,
            unoptimized_avg_live_values_x100: 200,
            optimized_vs_unoptimized_packets_delta: 1,
            optimized_vs_unoptimized_packets_delta_pct_x100: 5000,
            packet_utilization_x100: 66,
            max_packet_memory_reads: 1,
            memory_reads: 0,
            memory_writes: 0,
            total_memory_words_per_lane: 0,
            repeat: 1,
            cpu_ns: 10,
            cpu_ns_min: 10,
            cpu_ns_max: 10,
            packed_ns: 5,
            packed_ns_min: 5,
            packed_ns_max: 5,
            gpu_tick_construct_ns: None,
            gpu_tick_construct_ns_min: None,
            gpu_tick_construct_ns_max: None,
            gpu_tick_ns: None,
            gpu_tick_ns_min: None,
            gpu_tick_ns_max: None,
            gpu_tick_total_ns: None,
            gpu_tick_total_ns_min: None,
            gpu_tick_total_ns_max: None,
            gpu_tick_many_construct_ns: None,
            gpu_tick_many_construct_ns_min: None,
            gpu_tick_many_construct_ns_max: None,
            gpu_tick_many_ns: None,
            gpu_tick_many_ns_min: None,
            gpu_tick_many_ns_max: None,
            gpu_tick_many_total_ns: None,
            gpu_tick_many_total_ns_min: None,
            gpu_tick_many_total_ns_max: None,
            cpu_checksum: "0x0".to_string(),
            packed_checksum: "0x0".to_string(),
            gpu_tick_checksum: None,
            gpu_tick_many_checksum: None,
            gpu_available: false,
            packed_vs_cpu: Some(0.5),
            gpu_tick_many_vs_packed: None,
            autotune_rank: None,
            autotune_metric: "none".to_string(),
            autotune_metric_ns: None,
            autotune_best: false,
        }
    }

    fn quick_case_rows(case: &str) -> Vec<BenchRow> {
        let config = BenchConfig::parse(args(&["--quick", "--case", case, "--format", "json"]));
        let rows = run_benchmarks(&config);
        assert!(!rows.is_empty());
        rows
    }

    #[test]
    fn parses_quick_defaults() {
        let config = BenchConfig::parse(args(&["--quick"]));
        assert_eq!(config.format, OutputFormat::Csv);
        assert_eq!(config.lanes, vec![64]);
        assert_eq!(config.steps, vec![1, 16]);
        assert_eq!(config.caps, vec![Some(16)]);
        assert_eq!(config.mem_read_caps, vec![None]);
        assert!(!config.liveness_priority);
        assert!(!config.reuse_temporaries);
        assert_eq!(config.workgroups, vec![WORKGROUP_SIZE]);
        assert_eq!(config.repeat, 1);
        assert_eq!(config.timing_threshold_pct, 10.0);
        assert!(!config.strict);
        assert!(!config.autotune);
        assert_eq!(config.autotune_metric, AutotuneMetric::GpuTickMany);
        assert!(!config.recommend_config);
        assert!(config.cases.is_empty());
    }

    #[test]
    fn explicit_overrides_replace_quick_dimensions() {
        let config = BenchConfig::parse(args(&[
            "--quick", "--lanes", "64,1024", "--steps", "128", "--caps", "4,none",
        ]));
        assert_eq!(config.lanes, vec![64, 1024]);
        assert_eq!(config.steps, vec![128]);
        assert_eq!(config.caps, vec![Some(4), None]);
    }

    #[test]
    fn parses_repeated_cases_and_human_mode() {
        let config = BenchConfig::parse(args(&[
            "--human",
            "--case",
            "counter",
            "--case",
            "multi_read_memory",
            "--case",
            "deep_mixed_pipeline",
        ]));
        assert_eq!(config.format, OutputFormat::Human);
        assert!(config.should_run_case("counter"));
        assert!(config.should_run_case("multi_read_memory"));
        assert!(config.should_run_case("deep_mixed_pipeline"));
        assert!(!config.should_run_case("fifo_like"));
    }

    #[test]
    #[should_panic(expected = "unknown benchmark case `missing_case`")]
    fn rejects_unknown_benchmark_case() {
        BenchConfig::parse(args(&["--case", "missing_case"]));
    }

    #[test]
    fn parses_json_output_and_compare_options() {
        let config = BenchConfig::parse(args(&[
            "--format",
            "json",
            "--output",
            "bench.json",
            "--compare",
            "baseline.json",
            "--repeat",
            "3",
            "--timing-threshold",
            "7.5",
            "--workgroups",
            "32,64,128",
            "--mem-read-caps",
            "1,2,none",
            "--liveness-priority",
            "--reuse-temporaries",
            "--autotune",
            "--autotune-metric",
            "gpu_tick",
            "--recommend-config",
            "--strict",
        ]));
        assert_eq!(config.format, OutputFormat::Json);
        assert_eq!(config.output.as_deref(), Some("bench.json"));
        assert_eq!(config.compare.as_deref(), Some("baseline.json"));
        assert_eq!(config.repeat, 3);
        assert_eq!(config.timing_threshold_pct, 7.5);
        assert_eq!(config.workgroups, vec![32, 64, 128]);
        assert_eq!(config.mem_read_caps, vec![Some(1), Some(2), None]);
        assert!(config.liveness_priority);
        assert!(config.reuse_temporaries);
        assert!(config.autotune);
        assert_eq!(config.autotune_metric, AutotuneMetric::GpuTick);
        assert!(config.recommend_config);
        assert!(config.strict);
    }

    #[test]
    #[should_panic(expected = "--recommend-config requires --autotune")]
    fn recommendation_export_requires_autotune() {
        BenchConfig::parse(args(&["--recommend-config"]));
    }

    #[test]
    fn human_flag_is_format_alias() {
        let config = BenchConfig::parse(args(&["--human"]));
        assert_eq!(config.format, OutputFormat::Human);
    }

    #[test]
    fn serializes_benchmark_rows_to_json() {
        let row = sample_row();
        let json = serde_json::to_string(&vec![row.clone()]).unwrap();
        let rows = serde_json::from_str::<Vec<BenchRow>>(&json).unwrap();
        assert_eq!(rows, vec![row]);
    }

    #[test]
    fn packet_delta_reports_signed_packet_change() {
        assert_eq!(packet_delta(7, 5), 2);
        assert_eq!(packet_delta(3, 5), -2);
    }

    #[test]
    fn packet_delta_pct_reports_x100_percentage() {
        assert_eq!(packet_delta_pct_x100(12, 10), 2000);
        assert_eq!(packet_delta_pct_x100(8, 10), -2000);
        assert_eq!(packet_delta_pct_x100(8, 0), 0);
    }

    #[test]
    fn packet_utilization_reports_percent_x100_capacity_use() {
        assert_eq!(packet_utilization_x100(12, 3, 4), 100);
        assert_eq!(packet_utilization_x100(9, 3, 4), 75);
        assert_eq!(packet_utilization_x100(9, 0, 4), 0);
        assert_eq!(packet_utilization_x100(9, 3, 0), 0);
    }

    #[test]
    fn quick_multi_read_memory_case_produces_valid_stats() {
        let rows = quick_case_rows("multi_read_memory");

        assert!(rows.iter().all(|row| row.case == "multi_read_memory"));
        assert!(rows
            .iter()
            .all(|row| row.cpu_checksum == row.packed_checksum));
        assert!(rows.iter().all(|row| row.wgsl_bytes > 0));
        assert!(rows.iter().any(|row| row.memory_reads >= 3));
        assert!(rows.iter().any(|row| row.max_packet_memory_reads >= 2));
        assert!(rows.iter().all(|row| row.total_memory_words_per_lane > 0));
    }

    #[test]
    fn quick_deep_mixed_pipeline_case_produces_valid_stats() {
        let rows = quick_case_rows("deep_mixed_pipeline");

        assert!(rows.iter().all(|row| row.case == "deep_mixed_pipeline"));
        assert!(rows
            .iter()
            .all(|row| row.cpu_checksum == row.packed_checksum));
        assert!(rows.iter().all(|row| row.wgsl_bytes > 0));
        assert!(rows.iter().all(|row| row.packets_tick_next > 0));
        assert!(rows.iter().any(|row| row.max_live_values > 0));
        assert!(rows.iter().any(|row| row.packet_utilization_x100 > 0));
        assert!(rows.iter().all(|row| row.unoptimized_packets_total > 0));
        assert!(rows.iter().all(|row| row.total_memory_words_per_lane == 0));
    }

    #[test]
    fn autotune_ranking_prefers_lower_metric() {
        let mut fast = sample_row();
        fast.gpu_tick_many_ns = Some(20);
        let mut slow = sample_row();
        slow.schedule_cap = Some(8);
        slow.gpu_tick_many_ns = Some(30);
        let mut rows = vec![slow, fast];

        rank_autotune_rows(&mut rows, AutotuneMetric::GpuTickMany);

        assert_eq!(rows[1].autotune_rank, Some(1));
        assert!(rows[1].autotune_best);
        assert_eq!(rows[1].autotune_metric, "gpu_tick_many");
        assert_eq!(rows[1].autotune_metric_ns, Some(20));
        assert_eq!(rows[0].autotune_rank, Some(2));
        assert!(!rows[0].autotune_best);
    }

    #[test]
    fn autotune_ranking_falls_back_to_packed_when_gpu_metric_missing() {
        let mut fast = sample_row();
        fast.gpu_tick_many_ns = None;
        fast.packed_ns = 11;
        let mut slow = sample_row();
        slow.schedule_cap = Some(8);
        slow.gpu_tick_many_ns = None;
        slow.packed_ns = 12;
        let mut rows = vec![slow, fast];

        rank_autotune_rows(&mut rows, AutotuneMetric::GpuTickMany);

        assert_eq!(rows[1].autotune_rank, Some(1));
        assert_eq!(rows[1].autotune_metric_ns, Some(11));
        assert!(rows[1].autotune_best);
    }

    #[test]
    fn autotune_ranking_groups_by_case_lanes_and_steps() {
        let mut counter = sample_row();
        counter.gpu_tick_many_ns = Some(50);
        let mut wide = sample_row();
        wide.case = "wide_datapath".to_string();
        wide.gpu_tick_many_ns = Some(10);
        let mut other_lanes = sample_row();
        other_lanes.lanes = 128;
        other_lanes.gpu_tick_many_ns = Some(5);
        let mut rows = vec![counter, wide, other_lanes];

        rank_autotune_rows(&mut rows, AutotuneMetric::GpuTickMany);

        assert_eq!(rows[0].autotune_rank, Some(1));
        assert_eq!(rows[1].autotune_rank, Some(1));
        assert_eq!(rows[2].autotune_rank, Some(1));
        assert!(rows.iter().all(|row| row.autotune_best));
    }

    #[test]
    fn recommendations_include_only_autotune_best_rows() {
        let mut best = sample_row();
        best.gpu_tick_many_ns = Some(20);
        let mut slow = sample_row();
        slow.schedule_cap = Some(8);
        slow.gpu_tick_many_ns = Some(30);
        let mut rows = vec![slow, best];
        rank_autotune_rows(&mut rows, AutotuneMetric::GpuTickMany);

        let recommendations = recommended_configs(&rows);

        assert_eq!(recommendations.len(), 1);
        assert_eq!(recommendations[0].schedule_cap, Some(16));
        assert_eq!(recommendations[0].autotune_metric_ns, Some(20));
    }

    #[test]
    fn recommendations_project_selected_knobs_and_timings() {
        let mut row = sample_row();
        row.schedule_cap = None;
        row.memory_read_cap = Some(2);
        row.liveness_priority = false;
        row.reuse_temporaries = true;
        row.memory_layout = "word_major".to_string();
        row.workgroup_size = 256;
        row.packed_ns = 40;
        row.gpu_tick_ns = Some(25);
        row.gpu_tick_many_ns = Some(18);
        let mut rows = vec![row];
        rank_autotune_rows(&mut rows, AutotuneMetric::GpuTickMany);

        let recommendations = recommended_configs(&rows);

        assert_eq!(
            recommendations,
            vec![GpuAutotuneRecommendation {
                case: "counter".to_string(),
                lanes: 64,
                steps: 16,
                schedule_cap: None,
                memory_read_cap: Some(2),
                liveness_priority: false,
                reuse_temporaries: true,
                memory_layout: "word_major".to_string(),
                workgroup_size: 256,
                autotune_metric: "gpu_tick_many".to_string(),
                autotune_metric_ns: Some(18),
                packed_ns: 40,
                gpu_tick_ns: Some(25),
                gpu_tick_many_ns: Some(18),
            }]
        );
    }

    #[test]
    fn recommendations_are_ordered_by_case_lanes_and_steps() {
        let mut wide = sample_row();
        wide.case = "wide_datapath".to_string();
        wide.lanes = 64;
        wide.steps = 1;
        wide.gpu_tick_many_ns = Some(1);
        let mut counter_large = sample_row();
        counter_large.case = "counter".to_string();
        counter_large.lanes = 256;
        counter_large.steps = 1;
        counter_large.gpu_tick_many_ns = Some(1);
        let mut counter_small_steps = sample_row();
        counter_small_steps.case = "counter".to_string();
        counter_small_steps.lanes = 64;
        counter_small_steps.steps = 16;
        counter_small_steps.gpu_tick_many_ns = Some(1);
        let mut counter_first = sample_row();
        counter_first.case = "counter".to_string();
        counter_first.lanes = 64;
        counter_first.steps = 1;
        counter_first.gpu_tick_many_ns = Some(1);
        let mut rows = vec![wide, counter_large, counter_small_steps, counter_first];
        rank_autotune_rows(&mut rows, AutotuneMetric::GpuTickMany);

        let keys = recommended_configs(&rows)
            .into_iter()
            .map(|row| (row.case, row.lanes, row.steps))
            .collect::<Vec<_>>();

        assert_eq!(
            keys,
            vec![
                ("counter".to_string(), 64, 1),
                ("counter".to_string(), 64, 16),
                ("counter".to_string(), 256, 1),
                ("wide_datapath".to_string(), 64, 1),
            ]
        );
    }

    #[test]
    fn recommendation_export_writes_json_array() {
        let mut row = sample_row();
        row.gpu_tick_many_ns = Some(33);
        let mut rows = vec![row];
        rank_autotune_rows(&mut rows, AutotuneMetric::GpuTickMany);
        let mut out = Vec::new();

        print_recommended_configs(&mut out, &rows);

        let recommendations =
            serde_json::from_slice::<Vec<GpuAutotuneRecommendation>>(&out).unwrap();
        assert_eq!(recommendations.len(), 1);
        assert_eq!(recommendations[0].autotune_metric, "gpu_tick_many");
        assert_eq!(recommendations[0].gpu_tick_many_ns, Some(33));
    }

    #[test]
    fn compare_reports_no_change_baseline() {
        let row = sample_row();
        let mut out = Vec::new();
        let issues = print_compare_report(&mut out, &[row.clone()], &[row], 10.0);
        let report = String::from_utf8(out).unwrap();
        assert_eq!(issues, 0);
        assert!(report.contains("compare issues=0"));
    }

    #[test]
    fn compare_reports_packet_and_wgsl_regressions() {
        let baseline = sample_row();
        let mut current = baseline.clone();
        current.packets_total += 1;
        current.wgsl_bytes += 1;

        let mut out = Vec::new();
        let issues = print_compare_report(&mut out, &[baseline], &[current], 10.0);
        let report = String::from_utf8(out).unwrap();
        assert_eq!(issues, 2);
        assert!(report.contains("packet regression"));
        assert!(report.contains("wgsl regression"));
    }

    #[test]
    fn compare_ignores_missing_gpu_timing() {
        let baseline = sample_row();
        let current = baseline.clone();

        let mut out = Vec::new();
        print_compare_report(&mut out, &[baseline], &[current], 10.0);
        let report = String::from_utf8(out).unwrap();
        assert!(!report.contains("gpu_tick_many_ns"));
    }

    #[test]
    fn timing_stats_reports_median_min_and_max() {
        assert_eq!(
            timing_stats([30, 10, 20]),
            TimingStats {
                median: 20,
                min: 10,
                max: 30
            }
        );
        assert_eq!(
            timing_stats([40, 10, 30, 20]),
            TimingStats {
                median: 25,
                min: 10,
                max: 40
            }
        );
    }

    #[test]
    fn compare_ignores_timing_noise_below_threshold() {
        let baseline = sample_row();
        let mut current = baseline.clone();
        current.cpu_ns = 11;

        let mut out = Vec::new();
        let issues = print_compare_report(&mut out, &[baseline], &[current], 10.0);
        let report = String::from_utf8(out).unwrap();
        assert_eq!(issues, 0);
        assert!(!report.contains("timing regression"));
    }

    #[test]
    fn compare_reports_timing_regression_above_threshold() {
        let baseline = sample_row();
        let mut current = baseline.clone();
        current.cpu_ns = 12;

        let mut out = Vec::new();
        let issues = print_compare_report(&mut out, &[baseline], &[current], 10.0);
        let report = String::from_utf8(out).unwrap();
        assert_eq!(issues, 1);
        assert!(report.contains("timing regression"));
    }

    #[test]
    fn strict_compare_exits_nonzero_on_issues() {
        assert_eq!(compare_exit_code(false, 1), 0);
        assert_eq!(compare_exit_code(true, 0), 0);
        assert_eq!(compare_exit_code(true, 1), 1);
    }
}
