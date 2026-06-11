use std::{path::PathBuf, time::Duration};

use rrtl_core::{lit_u, mux, uint, Design, Diagnostic, ErrorReport, Signal};
use rrtl_runtime::{
    DistributedRuntime, DistributedRuntimeOptions, RuntimeCheckpointCadence,
    RuntimeCheckpointEvent, RuntimeCheckpointReason, RuntimeTopology, RuntimeWorker,
    TcpRuntimeSupervisor, TcpRuntimeSupervisorConfig, TcpRuntimeSupervisorTelemetry,
    TcpRuntimeWorkerProcess, TcpRuntimeWorkerProcessConfig, TcpRuntimeWorkerProcessSet,
    RUNTIME_TELEMETRY_FORMAT_VERSION,
};

fn worker_executable() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_rrtl-runtime-worker"))
}

fn counter_design() -> (Design, Signal, Signal) {
    let mut design = Design::new();
    let en;
    let out;
    {
        let mut m = design.module("Counter");
        let clk = m.input("clk", uint(1));
        let rst = m.input("rst", uint(1));
        en = m.input("en", uint(1));
        out = m.output("out", uint(8));
        let count = m.reg("count", uint(8));

        m.clock(count, clk);
        m.reset(count, rst, 0);
        m.next(count, mux(en, count + lit_u(1, 8), count));
        m.assign(out, count);
    }
    (design, en, out)
}

fn two_cpu_topology() -> RuntimeTopology {
    let mut topology = RuntimeTopology::new();
    topology.push(RuntimeWorker::local_cpu("cpu-a", 2));
    topology.push(RuntimeWorker::local_cpu("cpu-b", 3));
    topology
}

#[test]
fn worker_process_set_runs_tcp_runtime_and_exits_cleanly() {
    let (design, en, out) = counter_design();
    let topology = two_cpu_topology();
    let mut config = TcpRuntimeWorkerProcessConfig::new(worker_executable());
    config.max_connections = Some(1);
    config.read_timeout = Some(Duration::from_secs(5));
    config.write_timeout = Some(Duration::from_secs(5));

    let mut processes = TcpRuntimeWorkerProcessSet::spawn(&topology, &config).unwrap();
    assert_eq!(processes.endpoints().len(), 2);
    let health = processes.health().unwrap();
    assert_eq!(health.len(), 2);
    for worker in &health {
        assert!(worker.running);
        assert!(worker.exit.is_none());
        assert!(processes.endpoints().contains_key(&worker.worker_id));
    }

    {
        let mut runtime = DistributedRuntime::new_tcp_workers(
            &design,
            "Counter",
            topology,
            DistributedRuntimeOptions::default(),
            processes.endpoints().clone(),
        )
        .unwrap();
        runtime.set_input(en, &[1, 0, 1, 0, 1]).unwrap();
        runtime.tick().unwrap();
        assert_eq!(runtime.get_signal(out).unwrap(), vec![1, 0, 1, 0, 1]);
    }

    let exits = processes.wait_all().unwrap();
    assert_eq!(exits.len(), 2);
    for exit in exits {
        assert!(exit.success, "{} did not exit cleanly", exit.worker_id);
    }
    let health = processes.health().unwrap();
    for worker in health {
        assert!(!worker.running);
        assert!(worker.exit.unwrap().success);
    }
}

#[test]
fn worker_process_set_restarts_all_workers() {
    let topology = two_cpu_topology();
    let mut config = TcpRuntimeWorkerProcessConfig::new(worker_executable());
    config.read_timeout = Some(Duration::from_secs(5));
    config.write_timeout = Some(Duration::from_secs(5));

    let mut processes = TcpRuntimeWorkerProcessSet::spawn(&topology, &config).unwrap();
    let exits = processes.restart_all(&topology, &config).unwrap();

    assert_eq!(exits.len(), 2);
    assert_eq!(processes.endpoints().len(), 2);
    for worker in processes.health().unwrap() {
        assert!(worker.running);
        assert!(worker.exit.is_none());
    }

    processes.kill_all().unwrap();
    assert_eq!(processes.wait_all().unwrap().len(), 2);
}

#[test]
fn supervisor_runs_checkpointed_tcp_runtime() {
    let (design, en, out) = counter_design();
    let topology = two_cpu_topology();
    let mut process_config = TcpRuntimeWorkerProcessConfig::new(worker_executable());
    process_config.read_timeout = Some(Duration::from_secs(5));
    process_config.write_timeout = Some(Duration::from_secs(5));
    let config = TcpRuntimeSupervisorConfig::new(process_config);
    let mut supervisor = TcpRuntimeSupervisor::spawn(&design, "Counter", topology, config).unwrap();

    assert_eq!(supervisor.module_name(), "Counter");
    assert_eq!(supervisor.endpoints().len(), 2);
    let health = supervisor.health().unwrap();
    assert_eq!(health.processes.len(), 2);
    assert!(health.runtime_error.is_none());
    assert_eq!(health.runtime.unwrap().shards.len(), 2);

    supervisor
        .runtime_mut()
        .set_input(en, &[1, 1, 1, 1, 1])
        .unwrap();
    let mut events = Vec::new();
    let report = supervisor
        .tick_many_with_checkpoints(
            3,
            RuntimeCheckpointCadence {
                every_steps: 2,
                include_initial: true,
                include_final: true,
            },
            |event, checkpoint| {
                events.push(event);
                assert_eq!(checkpoint.tcp_endpoint_map().unwrap().len(), 2);
                Ok(())
            },
        )
        .unwrap();

    assert_eq!(report.completed_steps, 3);
    assert_eq!(report.checkpoints_emitted, 3);
    assert_eq!(
        events,
        vec![
            RuntimeCheckpointEvent {
                completed_steps: 0,
                reason: RuntimeCheckpointReason::Initial,
            },
            RuntimeCheckpointEvent {
                completed_steps: 2,
                reason: RuntimeCheckpointReason::Cadence,
            },
            RuntimeCheckpointEvent {
                completed_steps: 3,
                reason: RuntimeCheckpointReason::Final,
            },
        ]
    );
    assert!(supervisor.latest_checkpoint().is_some());
    assert_eq!(
        supervisor.runtime_mut().get_signal(out).unwrap(),
        vec![3, 3, 3, 3, 3]
    );

    assert_eq!(supervisor.shutdown().unwrap().len(), 2);
}

#[test]
fn supervisor_telemetry_serializes_running_workers_without_checkpoint() {
    let (design, _, _) = counter_design();
    let topology = two_cpu_topology();
    let config =
        TcpRuntimeSupervisorConfig::new(TcpRuntimeWorkerProcessConfig::new(worker_executable()));
    let mut supervisor =
        TcpRuntimeSupervisor::spawn(&design, "Counter", topology.clone(), config).unwrap();

    let telemetry = supervisor.telemetry().unwrap();
    assert_eq!(telemetry.format_version, RUNTIME_TELEMETRY_FORMAT_VERSION);
    assert_eq!(telemetry.module_name, "Counter");
    assert_eq!(telemetry.topology, topology);
    assert_eq!(telemetry.endpoints.len(), 2);
    assert_eq!(telemetry.processes.len(), 2);
    assert!(telemetry.processes.iter().all(|process| process.running));
    assert!(telemetry.runtime_health.is_some());
    assert!(telemetry.runtime_health_error.is_none());
    assert!(telemetry.latest_checkpoint.is_none());
    assert!(telemetry.last_recovery.is_none());

    let mut bytes = Vec::new();
    telemetry.write_json(&mut bytes).unwrap();
    let decoded = TcpRuntimeSupervisorTelemetry::read_json(&mut bytes.as_slice()).unwrap();
    assert_eq!(decoded, telemetry);

    assert_eq!(supervisor.shutdown().unwrap().len(), 2);
}

#[test]
fn supervisor_telemetry_includes_full_checkpoint_and_event() {
    let (design, en, _) = counter_design();
    let topology = two_cpu_topology();
    let config =
        TcpRuntimeSupervisorConfig::new(TcpRuntimeWorkerProcessConfig::new(worker_executable()));
    let mut supervisor = TcpRuntimeSupervisor::spawn(&design, "Counter", topology, config).unwrap();
    supervisor
        .runtime_mut()
        .set_input(en, &[1, 1, 1, 1, 1])
        .unwrap();
    supervisor
        .tick_many_with_checkpoints(2, RuntimeCheckpointCadence::every_steps(2), |_, _| Ok(()))
        .unwrap();

    let telemetry = supervisor.telemetry().unwrap();
    let latest = telemetry.latest_checkpoint.unwrap();
    assert_eq!(
        latest.event,
        Some(RuntimeCheckpointEvent {
            completed_steps: 2,
            reason: RuntimeCheckpointReason::Cadence,
        })
    );
    assert_eq!(latest.checkpoint.module_name, "Counter");
    assert_eq!(latest.checkpoint.tcp_endpoints.len(), 2);
    assert_eq!(latest.checkpoint.snapshot.total_lanes, 5);
    assert!(!latest.checkpoint.snapshot.shards[0].values.is_empty());

    assert_eq!(supervisor.shutdown().unwrap().len(), 2);
}

#[test]
fn supervisor_telemetry_reports_worker_failure_and_runtime_error() {
    let (design, _, _) = counter_design();
    let topology = two_cpu_topology();
    let config =
        TcpRuntimeSupervisorConfig::new(TcpRuntimeWorkerProcessConfig::new(worker_executable()));
    let mut supervisor = TcpRuntimeSupervisor::spawn(&design, "Counter", topology, config).unwrap();

    supervisor.processes_mut().processes_mut()[0]
        .kill()
        .unwrap();
    supervisor.processes_mut().processes_mut()[0]
        .wait()
        .unwrap();

    let telemetry = supervisor.telemetry().unwrap();
    assert!(telemetry.processes.iter().any(|process| !process.running));
    assert!(telemetry.runtime_health.is_none());
    assert!(telemetry.runtime_health_error.is_some());

    assert_eq!(supervisor.shutdown().unwrap().len(), 2);
}

#[test]
fn supervisor_recovers_from_latest_checkpoint_after_worker_failure() {
    let (design, en, out) = counter_design();
    let topology = two_cpu_topology();
    let mut process_config = TcpRuntimeWorkerProcessConfig::new(worker_executable());
    process_config.read_timeout = Some(Duration::from_secs(5));
    process_config.write_timeout = Some(Duration::from_secs(5));
    let config = TcpRuntimeSupervisorConfig::new(process_config);
    let mut supervisor =
        TcpRuntimeSupervisor::spawn(&design, "Counter", topology.clone(), config).unwrap();

    supervisor
        .runtime_mut()
        .set_input(en, &[1, 1, 1, 1, 1])
        .unwrap();
    supervisor
        .tick_many_with_checkpoints(2, RuntimeCheckpointCadence::every_steps(2), |_, _| Ok(()))
        .unwrap();
    let checkpoint_before_recovery = supervisor.latest_checkpoint().unwrap().clone();

    supervisor.processes_mut().processes_mut()[0]
        .kill()
        .unwrap();
    supervisor.processes_mut().processes_mut()[0]
        .wait()
        .unwrap();
    let health = supervisor.health().unwrap();
    assert!(health.processes.iter().any(|worker| !worker.running));

    let recovery = supervisor.recover_from_latest_checkpoint().unwrap();
    assert_eq!(recovery.restarted_workers.len(), 2);
    assert_eq!(
        recovery.runtime_recovery.recovered_workers,
        vec!["cpu-a".to_string(), "cpu-b".to_string()]
    );
    for worker in supervisor.health().unwrap().processes {
        assert!(worker.running);
        assert!(worker.exit.is_none());
    }
    assert_eq!(
        supervisor.runtime_mut().get_signal(out).unwrap(),
        vec![2, 2, 2, 2, 2]
    );

    supervisor
        .runtime_mut()
        .set_input(en, &[1, 1, 1, 1, 1])
        .unwrap();
    supervisor.runtime_mut().tick().unwrap();
    let mut direct = DistributedRuntime::new(&design, "Counter", topology).unwrap();
    direct
        .restore_checkpoint(&checkpoint_before_recovery)
        .unwrap();
    direct.set_input(en, &[1, 1, 1, 1, 1]).unwrap();
    direct.tick().unwrap();
    assert_eq!(
        supervisor.runtime_mut().get_signal(out).unwrap(),
        direct.get_signal(out).unwrap()
    );

    assert_eq!(supervisor.shutdown().unwrap().len(), 2);
}

#[test]
fn supervisor_telemetry_includes_last_recovery() {
    let (design, en, _) = counter_design();
    let topology = two_cpu_topology();
    let config =
        TcpRuntimeSupervisorConfig::new(TcpRuntimeWorkerProcessConfig::new(worker_executable()));
    let mut supervisor = TcpRuntimeSupervisor::spawn(&design, "Counter", topology, config).unwrap();
    supervisor
        .runtime_mut()
        .set_input(en, &[1, 1, 1, 1, 1])
        .unwrap();
    supervisor
        .tick_many_with_checkpoints(1, RuntimeCheckpointCadence::every_steps(1), |_, _| Ok(()))
        .unwrap();
    supervisor.processes_mut().processes_mut()[0]
        .kill()
        .unwrap();
    supervisor.processes_mut().processes_mut()[0]
        .wait()
        .unwrap();
    let recovery = supervisor.recover_from_latest_checkpoint().unwrap();

    let telemetry = supervisor.telemetry().unwrap();
    assert_eq!(telemetry.last_recovery, Some(recovery));
    assert!(telemetry.latest_checkpoint.is_some());
    assert!(telemetry.processes.iter().all(|process| process.running));

    assert_eq!(supervisor.shutdown().unwrap().len(), 2);
}

#[test]
fn supervisor_telemetry_rejects_version_mismatch() {
    let (design, _, _) = counter_design();
    let topology = two_cpu_topology();
    let config =
        TcpRuntimeSupervisorConfig::new(TcpRuntimeWorkerProcessConfig::new(worker_executable()));
    let mut supervisor = TcpRuntimeSupervisor::spawn(&design, "Counter", topology, config).unwrap();

    let mut telemetry = supervisor.telemetry().unwrap();
    telemetry.format_version = RUNTIME_TELEMETRY_FORMAT_VERSION + 1;
    let mut bytes = Vec::new();
    serde_json::to_writer(&mut bytes, &telemetry).unwrap();
    let err = TcpRuntimeSupervisorTelemetry::read_json(&mut bytes.as_slice()).unwrap_err();
    assert_eq!(err.diagnostics[0].code, "E_RUNTIME_TELEMETRY_VERSION");

    assert_eq!(supervisor.shutdown().unwrap().len(), 2);
}

#[test]
fn supervisor_recovery_requires_checkpoint() {
    let (design, _, _) = counter_design();
    let topology = two_cpu_topology();
    let config =
        TcpRuntimeSupervisorConfig::new(TcpRuntimeWorkerProcessConfig::new(worker_executable()));
    let mut supervisor = TcpRuntimeSupervisor::spawn(&design, "Counter", topology, config).unwrap();

    let err = supervisor.recover_from_latest_checkpoint().unwrap_err();
    assert_eq!(err.diagnostics[0].code, "E_RUNTIME_SUPERVISOR_CHECKPOINT");

    assert_eq!(supervisor.shutdown().unwrap().len(), 2);
}

#[test]
fn supervisor_checkpoint_callback_errors_are_propagated_after_storing_checkpoint() {
    let (design, en, _) = counter_design();
    let topology = two_cpu_topology();
    let config =
        TcpRuntimeSupervisorConfig::new(TcpRuntimeWorkerProcessConfig::new(worker_executable()));
    let mut supervisor = TcpRuntimeSupervisor::spawn(&design, "Counter", topology, config).unwrap();
    supervisor
        .runtime_mut()
        .set_input(en, &[1, 1, 1, 1, 1])
        .unwrap();

    let err = supervisor
        .tick_many_with_checkpoints(1, RuntimeCheckpointCadence::every_steps(1), |_, _| {
            Err(ErrorReport::new(vec![Diagnostic::new(
                "E_TEST_SUPERVISOR_SINK",
                "supervisor checkpoint sink failed",
            )]))
        })
        .unwrap_err();

    assert_eq!(err.diagnostics[0].code, "E_TEST_SUPERVISOR_SINK");
    assert!(supervisor.latest_checkpoint().is_some());

    assert_eq!(supervisor.shutdown().unwrap().len(), 2);
}

#[test]
fn worker_process_spawn_rejects_missing_executable() {
    let config = TcpRuntimeWorkerProcessConfig::new("/definitely/missing/rrtl-runtime-worker");
    let err = match TcpRuntimeWorkerProcess::spawn("cpu-a", &config) {
        Ok(_) => panic!("expected missing executable to fail"),
        Err(err) => err,
    };
    assert_eq!(err.diagnostics[0].code, "E_RUNTIME_WORKER_PROCESS_IO");
}

#[cfg(unix)]
#[test]
fn worker_process_spawn_rejects_invalid_startup_output() {
    let config = TcpRuntimeWorkerProcessConfig::new("/bin/echo");
    let err = match TcpRuntimeWorkerProcess::spawn("cpu-a", &config) {
        Ok(_) => panic!("expected invalid startup output to fail"),
        Err(err) => err,
    };
    assert_eq!(err.diagnostics[0].code, "E_RUNTIME_WORKER_PROCESS_STARTUP");
}
