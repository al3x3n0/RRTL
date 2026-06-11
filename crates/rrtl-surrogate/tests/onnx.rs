#![cfg(feature = "onnx-ort")]

use std::fs;
use std::path::{Path, PathBuf};

use rrtl_surrogate::{
    canonical_json_hash, policy_event_corpus, read_manifest, run_gemm_transaction, sha256_hex,
    shadow_event_corpus, validate_manifest, ArtifactFormat, ArtifactSpec, DomainSpec,
    FallbackPolicy, GemmTransaction, InstrumentationEvent, InstrumentationEventCorpus, LabelSpec,
    ModelFamily, PolicyMode, PolicySpec, SourceSpec, SurrogateClass, SurrogateManifest, TaskSpec,
    ValidationSpec, EVENT_CORPUS_SCHEMA, EVENT_SCHEMA, MANIFEST_SCHEMA,
};

fn gemm_fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tiny_gemm.onnx")
}

fn event_fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tiny_event_predictor.onnx")
}

fn manifest_for_fixture() -> Option<SurrogateManifest> {
    let fixture = gemm_fixture_path();
    let bytes = fs::read(&fixture).ok()?;
    Some(SurrogateManifest {
        schema: MANIFEST_SCHEMA.to_string(),
        surrogate_id: "tiny_gemm_onnx".to_string(),
        surrogate_class: SurrogateClass::TransactionKernel,
        model_family: ModelFamily::GnnTransformer,
        task: None,
        source: SourceSpec {
            top_name: "Top".to_string(),
            export_schema: "rrtl-pyrtl-block-v1".to_string(),
            source_hash: canonical_json_hash(&serde_json::json!({"fixture": "tiny_gemm"})).unwrap(),
        },
        artifact: ArtifactSpec {
            format: ArtifactFormat::Onnx,
            path: fixture.to_string_lossy().to_string(),
            sha256: sha256_hex(&bytes),
            input_tensors: vec![
                "gemm_descriptor".to_string(),
                "a_tensor".to_string(),
                "w_tensor".to_string(),
            ],
            output_tensors: vec!["c_tensor".to_string(), "telemetry".to_string()],
            opset: Some(17),
        },
        domain: DomainSpec {
            rows: 2,
            cols: 2,
            k_min: 2,
            k_max: 2,
            data_width: 8,
            acc_width: 32,
        },
        validation: ValidationSpec {
            max_abs_error: 0,
            max_mean_abs_error: 0.0,
            max_latency_error_cycles: 0,
        },
        policy: PolicySpec {
            mode: PolicyMode::ApproximateWithTolerance,
            fallback: FallbackPolicy::FailClosed,
            provenance_tag: "approximate".to_string(),
        },
    })
}

fn transaction(expected_c: Option<Vec<Vec<i128>>>) -> GemmTransaction {
    GemmTransaction {
        schema: "rrtl-surrogate-gemm-transaction-v1".to_string(),
        lane: None,
        rows: 2,
        cols: 2,
        k: 2,
        a: vec![vec![1, 2], vec![3, 4]],
        w: vec![vec![5, 6], vec![7, 8]],
        expected_c,
        expected_latency_cycles: Some(5),
    }
}

fn event_manifest_for_fixture() -> Option<SurrogateManifest> {
    let fixture = event_fixture_path();
    let bytes = fs::read(&fixture).ok()?;
    Some(SurrogateManifest {
        schema: MANIFEST_SCHEMA.to_string(),
        surrogate_id: "tiny_event_onnx".to_string(),
        surrogate_class: SurrogateClass::EventPredictor,
        model_family: ModelFamily::GnnTransformer,
        task: Some(TaskSpec {
            prediction_target: "cache_miss".to_string(),
            input_window_cycles: 2,
            horizon_cycles: 1,
            signal_features: vec!["load".to_string()],
            program_features: vec!["pc".to_string()],
            label: Some(LabelSpec {
                name: "cache_miss".to_string(),
                kind: "binary".to_string(),
                positive_value: 1,
            }),
        }),
        source: SourceSpec {
            top_name: "InstrumentedCache".to_string(),
            export_schema: EVENT_CORPUS_SCHEMA.to_string(),
            source_hash: "event-source".to_string(),
        },
        artifact: ArtifactSpec {
            format: ArtifactFormat::Onnx,
            path: fixture.to_string_lossy().to_string(),
            sha256: sha256_hex(&bytes),
            input_tensors: vec!["signal_window".to_string(), "program_context".to_string()],
            output_tensors: vec![
                "event_probability".to_string(),
                "predicted_event".to_string(),
            ],
            opset: Some(17),
        },
        domain: DomainSpec {
            rows: 1,
            cols: 1,
            k_min: 1,
            k_max: 2,
            data_width: 64,
            acc_width: 64,
        },
        validation: ValidationSpec {
            max_abs_error: 0,
            max_mean_abs_error: 0.0,
            max_latency_error_cycles: 0,
        },
        policy: PolicySpec {
            mode: PolicyMode::TelemetryOnly,
            fallback: FallbackPolicy::FailClosed,
            provenance_tag: "instrumentation_prediction".to_string(),
        },
    })
}

fn event_manifest_for_gemm_fixture() -> Option<SurrogateManifest> {
    let fixture = gemm_fixture_path();
    let bytes = fs::read(&fixture).ok()?;
    let mut manifest = event_manifest_for_fixture()?;
    manifest.artifact.path = fixture.to_string_lossy().to_string();
    manifest.artifact.sha256 = sha256_hex(&bytes);
    Some(manifest)
}

fn event_corpus() -> InstrumentationEventCorpus {
    InstrumentationEventCorpus {
        schema: EVENT_CORPUS_SCHEMA.to_string(),
        source_hash: "event-source".to_string(),
        top_name: "InstrumentedCache".to_string(),
        events: vec![InstrumentationEvent {
            schema: EVENT_SCHEMA.to_string(),
            sample_id: 0,
            lane: Some(0),
            target: "cache_miss".to_string(),
            window_cycles: 2,
            horizon_cycles: 1,
            program: [("pc".to_string(), 4096)].into_iter().collect(),
            signals: vec![
                [("load".to_string(), 1)].into_iter().collect(),
                [("load".to_string(), 1)].into_iter().collect(),
            ],
            label: [("cache_miss".to_string(), 1)].into_iter().collect(),
        }],
    }
}

#[test]
fn onnx_gemm_fixture_runs_and_validates() {
    let Some(manifest) = manifest_for_fixture() else {
        eprintln!("skipping ONNX fixture test: tiny_gemm.onnx is missing");
        return;
    };
    let result = run_gemm_transaction(
        &manifest,
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("manifest.json"),
        &transaction(Some(vec![vec![19, 22], vec![43, 50]])),
    )
    .unwrap();

    assert!(result.ok, "{result:?}");
    assert_eq!(result.c, vec![vec![19, 22], vec![43, 50]]);
    assert_eq!(result.telemetry.latency_cycles, 5);
    assert_eq!(result.telemetry.active_cycles, 2);
    assert!((result.telemetry.utilization - 0.4).abs() < 1e-6);
    assert_eq!(result.provenance.artifact_format, ArtifactFormat::Onnx);
}

#[test]
fn onnx_gemm_reports_tolerance_failure() {
    let Some(manifest) = manifest_for_fixture() else {
        eprintln!("skipping ONNX fixture test: tiny_gemm.onnx is missing");
        return;
    };
    let result = run_gemm_transaction(
        &manifest,
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("manifest.json"),
        &transaction(Some(vec![vec![19, 99], vec![43, 50]])),
    )
    .unwrap();

    assert!(!result.ok);
    let metrics = result.metrics.unwrap();
    assert_eq!(metrics.max_abs_error, 77);
    let divergence = metrics.first_divergence.unwrap();
    assert_eq!(divergence.row, 0);
    assert_eq!(divergence.col, 1);
    assert_eq!(divergence.expected, 99);
    assert_eq!(divergence.actual, 22);
}

#[test]
fn onnx_manifest_requires_standard_tensor_names() {
    let Some(mut manifest) = manifest_for_fixture() else {
        eprintln!("skipping ONNX fixture test: tiny_gemm.onnx is missing");
        return;
    };
    manifest.artifact.input_tensors = vec!["a_tensor".to_string(), "w_tensor".to_string()];
    let report = validate_manifest(&manifest, Path::new("."));
    assert!(!report.ok);
    assert!(report
        .errors
        .iter()
        .any(|err| err.contains("gemm_descriptor")));
}

#[test]
fn onnx_session_reports_manifest_tensor_mismatch() {
    let Some(mut manifest) = manifest_for_fixture() else {
        eprintln!("skipping ONNX fixture test: tiny_gemm.onnx is missing");
        return;
    };
    manifest
        .artifact
        .output_tensors
        .push("missing_output".to_string());
    let err = run_gemm_transaction(
        &manifest,
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("manifest.json"),
        &transaction(None),
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("missing_output"), "{err}");
}

#[test]
fn onnx_event_predictor_fixture_shadows_successfully() {
    let Some(manifest) = event_manifest_for_fixture() else {
        eprintln!("skipping ONNX fixture test: tiny_event_predictor.onnx is missing");
        return;
    };
    let report = shadow_event_corpus(
        &manifest,
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("manifest.json"),
        &event_corpus(),
    );

    assert!(report.ok, "{report:?}");
    assert!(report.manifest.ok, "{report:?}");
    assert_eq!(report.results.len(), 1);
    assert_eq!(report.results[0].expected, 1);
    assert_eq!(report.results[0].predicted, 1);
    assert_eq!(report.metrics.accuracy, 1.0);
    assert_eq!(report.provenance.model_family, ModelFamily::GnnTransformer);
    assert_eq!(report.provenance.artifact_format, ArtifactFormat::Onnx);
}

#[test]
fn onnx_event_policy_fixture_uses_surrogate() {
    let Some(mut manifest) = event_manifest_for_fixture() else {
        eprintln!("skipping ONNX fixture test: tiny_event_predictor.onnx is missing");
        return;
    };
    manifest.policy.mode = PolicyMode::ApproximateWithTolerance;
    let report = policy_event_corpus(
        &manifest,
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("manifest.json"),
        &event_corpus(),
    );

    assert!(report.ok, "{report:?}");
    assert_eq!(report.count, 1);
    assert_eq!(report.used_surrogate, 1);
    assert_eq!(report.results[0].predicted, 1);
    assert_eq!(
        report.results[0].decision,
        rrtl_surrogate::EventPolicyDecision::SurrogateUsed
    );
    assert_eq!(
        report.results[0].provenance.model_family,
        ModelFamily::GnnTransformer
    );
    assert_eq!(
        report.results[0].provenance.artifact_format,
        ArtifactFormat::Onnx
    );
}

#[test]
fn onnx_event_predictor_reports_session_io_mismatch() {
    let Some(manifest) = event_manifest_for_gemm_fixture() else {
        eprintln!("skipping ONNX fixture test: tiny_gemm.onnx is missing");
        return;
    };
    let report = shadow_event_corpus(
        &manifest,
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("manifest.json"),
        &event_corpus(),
    );

    assert!(!report.ok);
    assert!(
        report
            .errors
            .iter()
            .any(|err| err.contains("signal_window")),
        "{report:?}"
    );
}

#[test]
fn onnx_manifest_round_trips_json() {
    let Some(manifest) = manifest_for_fixture() else {
        eprintln!("skipping ONNX fixture test: tiny_gemm.onnx is missing");
        return;
    };
    let text = serde_json::to_string(&manifest).unwrap();
    let loaded = read_manifest(text.as_bytes()).unwrap();
    assert_eq!(loaded.artifact.format, ArtifactFormat::Onnx);
}
