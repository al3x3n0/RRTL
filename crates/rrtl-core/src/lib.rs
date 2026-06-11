use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

pub use rrtl_ir::{
    as_sint, as_uint, bundle_type, concat, field, iface_input, iface_output, interface_type, lit,
    lit_s, lit_u, mem_read, mux, nested, sext, sint, trunc, uint, zext, Assertion, Assignment,
    BitType, BundleField, BundleSignal, BundleSignalField, BundleType, CoverPoint, Diagnostic,
    ErrorReport, Expr, InitialMemoryValue, InitialRegisterValue, Instance, InterfaceDirection,
    InterfacePort, InterfacePortType, InterfaceSignal, InterfaceSignalPort,
    InterfaceSignalPortSignals, InterfaceType, MemoryWrite, Module, ModuleId, Reset, ResetKind,
    ResetPolarity, Signal, SignalId, SignalInfo, SignalKind, Signedness, StateSignal, StateType,
    StateVariant, Width,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PortDirection {
    Input,
    Output,
    Inout,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledDesign {
    modules: Vec<CompiledModule>,
}

impl CompiledDesign {
    pub fn modules(&self) -> &[CompiledModule] {
        &self.modules
    }

    pub fn find_module(&self, name: &str) -> Option<&CompiledModule> {
        self.modules.iter().find(|module| module.name == name)
    }

    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledModule {
    pub id: ModuleId,
    pub name: String,
    pub is_external: bool,
    pub signals: Vec<SignalInfo>,
    pub assignments: Vec<CompiledAssignment>,
    pub registers: Vec<CompiledRegister>,
    pub memory_writes: Vec<CompiledMemoryWrite>,
    pub initial_register_values: Vec<InitialRegisterValue>,
    pub initial_memory_values: Vec<InitialMemoryValue>,
    pub assertions: Vec<CompiledAssertion>,
    pub cover_points: Vec<CompiledCoverPoint>,
    pub instances: Vec<CompiledInstance>,
    pub state_types: Vec<StateType>,
    pub state_signals: Vec<StateSignal>,
    pub bundle_types: Vec<BundleType>,
    pub bundle_signals: Vec<BundleSignal>,
    pub interface_types: Vec<InterfaceType>,
    pub interface_signals: Vec<InterfaceSignal>,
}

impl CompiledModule {
    pub fn signal(&self, signal: Signal) -> Option<&SignalInfo> {
        if signal.module != self.id {
            return None;
        }
        self.signals.get(signal.id.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledExpr {
    pub expr: Expr,
    pub width: Width,
    pub ty: BitType,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledAssignment {
    pub dst: Signal,
    pub expr: CompiledExpr,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledRegister {
    pub signal: Signal,
    pub clock: Signal,
    pub reset: Option<Reset>,
    pub next: CompiledExpr,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledMemoryWrite {
    pub mem: Signal,
    pub clock: Signal,
    pub enable: CompiledExpr,
    pub addr: CompiledExpr,
    pub data: CompiledExpr,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledAssertion {
    pub name: String,
    pub clock: Option<Signal>,
    pub enable: Option<CompiledExpr>,
    pub condition: CompiledExpr,
    pub message: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledCoverPoint {
    pub name: String,
    pub clock: Option<Signal>,
    pub enable: Option<CompiledExpr>,
    pub condition: CompiledExpr,
    pub message: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledInstance {
    pub name: String,
    pub module: String,
    pub connections: Vec<CompiledConnection>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledConnection {
    pub port: String,
    pub signal: Signal,
    pub direction: PortDirection,
    pub width: Width,
    pub ty: BitType,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoverHit {
    pub name: String,
    pub hits: u64,
}

#[derive(Clone, Debug, Default)]
pub struct Design {
    ir: rrtl_ir::Design,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateReg {
    pub signal: Signal,
    pub state_type: StateType,
}

impl StateReg {
    pub fn signal(&self) -> Signal {
        self.signal
    }

    pub fn value(&self) -> Expr {
        self.signal.value()
    }

    pub fn lit(&self, variant: &str) -> Option<Expr> {
        self.state_type.lit(variant)
    }

    pub fn is(&self, variant: &str) -> Expr {
        self.value().eq_expr(
            self.lit(variant)
                .unwrap_or_else(|| lit_u(0, self.state_type.ty.width)),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BundleRef {
    pub name: String,
    pub bundle_type: BundleType,
    fields: Vec<BundleRefField>,
}

pub type BundleValue = BundleRef;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BundleRefField {
    pub path: Vec<String>,
    pub signal: Signal,
    pub ty: BitType,
}

impl BundleRef {
    pub fn field(&self, name: &str) -> Option<Signal> {
        self.path([name])
    }

    pub fn path<I, S>(&self, path: I) -> Option<Signal>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let path = path
            .into_iter()
            .map(|part| part.as_ref().to_string())
            .collect::<Vec<_>>();
        self.fields
            .iter()
            .find(|field| field.path == path)
            .map(|field| field.signal)
    }

    pub fn fields(&self) -> &[BundleRefField] {
        &self.fields
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InterfaceRef {
    pub name: String,
    pub interface_type: InterfaceType,
    ports: Vec<PortRef>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortRef {
    pub name: String,
    pub direction: InterfaceDirection,
    pub ty: InterfacePortType,
    signal: Option<Signal>,
    bundle: Option<BundleRef>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadyValidRole {
    Source,
    Sink,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadyValidRef {
    interface: InterfaceRef,
    role: ReadyValidRole,
    payload: InterfacePortType,
}

impl InterfaceRef {
    pub fn port(&self, name: &str) -> Option<PortRef> {
        self.ports.iter().find(|port| port.name == name).cloned()
    }

    pub fn signal(&self, port_name: &str) -> Option<Signal> {
        self.port(port_name).and_then(|port| port.signal())
    }

    pub fn bundle(&self, port_name: &str) -> Option<&BundleRef> {
        self.ports
            .iter()
            .find(|port| port.name == port_name)
            .and_then(|port| port.bundle())
    }

    pub fn field(&self, port_name: &str, field_name: &str) -> Option<Signal> {
        self.bundle(port_name)
            .and_then(|bundle| bundle.field(field_name))
    }

    pub fn path<I, S>(&self, port_name: &str, path: I) -> Option<Signal>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.bundle(port_name).and_then(|bundle| bundle.path(path))
    }

    pub fn ports(&self) -> &[PortRef] {
        &self.ports
    }
}

impl PortRef {
    pub fn signal(&self) -> Option<Signal> {
        self.signal
    }

    pub fn bundle(&self) -> Option<&BundleRef> {
        self.bundle.as_ref()
    }

    pub fn field(&self, name: &str) -> Option<Signal> {
        self.bundle().and_then(|bundle| bundle.field(name))
    }

    pub fn path<I, S>(&self, path: I) -> Option<Signal>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.bundle().and_then(|bundle| bundle.path(path))
    }
}

impl ReadyValidRef {
    pub fn interface(&self) -> &InterfaceRef {
        &self.interface
    }

    pub fn into_interface(self) -> InterfaceRef {
        self.interface
    }

    pub fn role(&self) -> ReadyValidRole {
        self.role
    }

    pub fn payload(&self) -> &InterfacePortType {
        &self.payload
    }

    pub fn valid(&self) -> Signal {
        self.interface
            .port("valid")
            .and_then(|port| port.signal())
            .expect("ready/valid channel has scalar valid")
    }

    pub fn ready(&self) -> Signal {
        self.interface
            .port("ready")
            .and_then(|port| port.signal())
            .expect("ready/valid channel has scalar ready")
    }

    pub fn bits_signal(&self) -> Option<Signal> {
        self.interface
            .ports()
            .iter()
            .find(|port| port.name == "bits")
            .and_then(|port| port.signal())
    }

    pub fn bits_bundle(&self) -> Option<&BundleRef> {
        self.interface
            .ports()
            .iter()
            .find(|port| port.name == "bits")
            .and_then(|port| port.bundle())
    }

    pub fn fire(&self) -> Expr {
        self.valid() & self.ready()
    }
}

pub fn rv_scalar(ty: impl Into<BitType>) -> InterfacePortType {
    InterfacePortType::Scalar(ty.into())
}

pub fn rv_bundle(bundle: BundleType) -> InterfacePortType {
    InterfacePortType::Bundle(bundle)
}

pub fn rv_source(name: impl Into<String>, payload: impl Into<InterfacePortType>) -> InterfaceType {
    ready_valid_type(name, payload.into(), ReadyValidRole::Source)
}

pub fn rv_sink(name: impl Into<String>, payload: impl Into<InterfacePortType>) -> InterfaceType {
    ready_valid_type(name, payload.into(), ReadyValidRole::Sink)
}

pub fn validate_ready_valid_type(
    interface_type: &InterfaceType,
    role: ReadyValidRole,
) -> Result<InterfacePortType, ErrorReport> {
    let mut diagnostics = Vec::new();
    let ports = interface_type
        .ports
        .iter()
        .map(|port| (port.name.as_str(), port))
        .collect::<HashMap<_, _>>();

    for expected in ["valid", "ready", "bits"] {
        if !ports.contains_key(expected) {
            diagnostics.push(Diagnostic::new(
                "E_RV_PORT_MISSING",
                format!(
                    "ready/valid interface `{}` is missing port `{expected}`",
                    interface_type.name
                ),
            ));
        }
    }

    for port in &interface_type.ports {
        if !matches!(port.name.as_str(), "valid" | "ready" | "bits") {
            diagnostics.push(Diagnostic::new(
                "E_RV_PORT_EXTRA",
                format!(
                    "ready/valid interface `{}` has extra port `{}`",
                    interface_type.name, port.name
                ),
            ));
        }
    }

    validate_ready_valid_scalar_port(interface_type, &ports, "valid", role, &mut diagnostics);
    validate_ready_valid_scalar_port(interface_type, &ports, "ready", role, &mut diagnostics);

    let payload = ports.get("bits").map(|port| port.ty.clone());
    if let Some(bits) = ports.get("bits") {
        let expected = ready_valid_direction(role, "bits");
        if bits.direction != expected {
            diagnostics.push(Diagnostic::new(
                "E_RV_PORT_DIRECTION",
                format!(
                    "ready/valid interface `{}` port `bits` has direction {:?}, expected {:?}",
                    interface_type.name, bits.direction, expected
                ),
            ));
        }
        if let InterfacePortType::Scalar(ty) = bits.ty {
            if ty.width == 0 {
                diagnostics.push(Diagnostic::new(
                    "E_RV_BITS_TYPE",
                    format!(
                        "ready/valid interface `{}` scalar bits payload must have non-zero width",
                        interface_type.name
                    ),
                ));
            }
        }
    }

    if diagnostics.is_empty() {
        Ok(payload.expect("bits port presence checked"))
    } else {
        Err(ErrorReport::new(diagnostics))
    }
}

fn ready_valid_type(
    name: impl Into<String>,
    payload: InterfacePortType,
    role: ReadyValidRole,
) -> InterfaceType {
    let name = name.into();
    let (valid, ready, bits) = match role {
        ReadyValidRole::Source => (
            iface_output("valid", uint(1)),
            iface_input("ready", uint(1)),
            iface_output("bits", payload),
        ),
        ReadyValidRole::Sink => (
            iface_input("valid", uint(1)),
            iface_output("ready", uint(1)),
            iface_input("bits", payload),
        ),
    };
    interface_type(name, [valid, ready, bits])
}

fn validate_ready_valid_scalar_port(
    interface_type: &InterfaceType,
    ports: &HashMap<&str, &InterfacePort>,
    name: &str,
    role: ReadyValidRole,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let Some(port) = ports.get(name) else {
        return;
    };
    let expected = ready_valid_direction(role, name);
    if port.direction != expected {
        diagnostics.push(Diagnostic::new(
            "E_RV_PORT_DIRECTION",
            format!(
                "ready/valid interface `{}` port `{name}` has direction {:?}, expected {:?}",
                interface_type.name, port.direction, expected
            ),
        ));
    }
    match port.ty {
        InterfacePortType::Scalar(ty) if ty == uint(1) => {}
        InterfacePortType::Scalar(ty) => diagnostics.push(Diagnostic::new(
            "E_RV_CONTROL_TYPE",
            format!(
                "ready/valid interface `{}` port `{name}` must be unsigned 1 bit, found {:?}",
                interface_type.name, ty
            ),
        )),
        InterfacePortType::Bundle(_) => diagnostics.push(Diagnostic::new(
            "E_RV_CONTROL_TYPE",
            format!(
                "ready/valid interface `{}` port `{name}` must be scalar unsigned 1 bit",
                interface_type.name
            ),
        )),
    }
}

fn ready_valid_direction(role: ReadyValidRole, port: &str) -> InterfaceDirection {
    match (role, port) {
        (ReadyValidRole::Source, "ready") | (ReadyValidRole::Sink, "valid" | "bits") => {
            InterfaceDirection::Input
        }
        (ReadyValidRole::Source, "valid" | "bits") | (ReadyValidRole::Sink, "ready") => {
            InterfaceDirection::Output
        }
        _ => InterfaceDirection::Input,
    }
}

fn validate_rv_slice_bundle_payloads(
    input: &BundleRef,
    output: &BundleRef,
    module: &Module,
) -> Result<Vec<(Vec<String>, Signal, Signal, BitType)>, Vec<Diagnostic>> {
    let mut diagnostics = Vec::new();
    let input_fields = input
        .fields()
        .iter()
        .map(|field| (field.path.clone(), field))
        .collect::<HashMap<_, _>>();
    let output_fields = output
        .fields()
        .iter()
        .map(|field| (field.path.clone(), field))
        .collect::<HashMap<_, _>>();

    for output_field in output.fields() {
        if !input_fields.contains_key(&output_field.path) {
            diagnostics.push(
                Diagnostic::new(
                    "E_RV_SLICE_FIELD_MISSING",
                    format!(
                        "ready/valid register slice input payload is missing field `{}`",
                        output_field.path.join(".")
                    ),
                )
                .with_module(module.name.clone()),
            );
        }
    }

    for input_field in input.fields() {
        if !output_fields.contains_key(&input_field.path) {
            diagnostics.push(
                Diagnostic::new(
                    "E_RV_SLICE_FIELD_EXTRA",
                    format!(
                        "ready/valid register slice input payload has extra field `{}`",
                        input_field.path.join(".")
                    ),
                )
                .with_module(module.name.clone()),
            );
        }
    }

    let mut fields = Vec::new();
    for output_field in output.fields() {
        let Some(input_field) = input_fields.get(&output_field.path) else {
            continue;
        };
        if input_field.ty != output_field.ty {
            diagnostics.push(
                Diagnostic::new(
                    "E_RV_SLICE_PAYLOAD_TYPE",
                    format!(
                        "ready/valid register slice field `{}` type mismatch: input is {:?}, output is {:?}",
                        output_field.path.join("."),
                        input_field.ty,
                        output_field.ty
                    ),
                )
                .with_module(module.name.clone()),
            );
            continue;
        }
        fields.push((
            output_field.path.clone(),
            input_field.signal,
            output_field.signal,
            output_field.ty,
        ));
    }

    if diagnostics.is_empty() {
        Ok(fields)
    } else {
        Err(diagnostics)
    }
}

fn validate_rv_connect_bundle_payloads(
    input: &BundleRef,
    output: &BundleRef,
    module: &Module,
) -> Result<Vec<(Signal, Signal)>, Vec<Diagnostic>> {
    let mut diagnostics = Vec::new();
    let input_fields = input
        .fields()
        .iter()
        .map(|field| (field.path.clone(), field))
        .collect::<HashMap<_, _>>();
    let output_fields = output
        .fields()
        .iter()
        .map(|field| (field.path.clone(), field))
        .collect::<HashMap<_, _>>();

    for output_field in output.fields() {
        if !input_fields.contains_key(&output_field.path) {
            diagnostics.push(
                Diagnostic::new(
                    "E_RV_CONNECT_PAYLOAD_TYPE",
                    format!(
                        "ready/valid connect input payload is missing field `{}`",
                        output_field.path.join(".")
                    ),
                )
                .with_module(module.name.clone()),
            );
        }
    }

    for input_field in input.fields() {
        if !output_fields.contains_key(&input_field.path) {
            diagnostics.push(
                Diagnostic::new(
                    "E_RV_CONNECT_PAYLOAD_TYPE",
                    format!(
                        "ready/valid connect input payload has extra field `{}`",
                        input_field.path.join(".")
                    ),
                )
                .with_module(module.name.clone()),
            );
        }
    }

    let mut fields = Vec::new();
    for output_field in output.fields() {
        let Some(input_field) = input_fields.get(&output_field.path) else {
            continue;
        };
        if input_field.ty != output_field.ty {
            diagnostics.push(
                Diagnostic::new(
                    "E_RV_CONNECT_PAYLOAD_TYPE",
                    format!(
                        "ready/valid connect field `{}` type mismatch: input is {:?}, output is {:?}",
                        output_field.path.join("."),
                        input_field.ty,
                        output_field.ty
                    ),
                )
                .with_module(module.name.clone()),
            );
            continue;
        }
        fields.push((input_field.signal, output_field.signal));
    }

    if diagnostics.is_empty() {
        Ok(fields)
    } else {
        Err(diagnostics)
    }
}

fn validate_rv_skid_bundle_payloads(
    input: &BundleRef,
    output: &BundleRef,
    module: &Module,
) -> Result<Vec<(Vec<String>, Signal, Signal, BitType)>, Vec<Diagnostic>> {
    let mut diagnostics = Vec::new();
    let input_fields = input
        .fields()
        .iter()
        .map(|field| (field.path.clone(), field))
        .collect::<HashMap<_, _>>();
    let output_fields = output
        .fields()
        .iter()
        .map(|field| (field.path.clone(), field))
        .collect::<HashMap<_, _>>();

    for output_field in output.fields() {
        if !input_fields.contains_key(&output_field.path) {
            diagnostics.push(
                Diagnostic::new(
                    "E_RV_SKID_FIELD_MISSING",
                    format!(
                        "ready/valid skid buffer input payload is missing field `{}`",
                        output_field.path.join(".")
                    ),
                )
                .with_module(module.name.clone()),
            );
        }
    }

    for input_field in input.fields() {
        if !output_fields.contains_key(&input_field.path) {
            diagnostics.push(
                Diagnostic::new(
                    "E_RV_SKID_FIELD_EXTRA",
                    format!(
                        "ready/valid skid buffer input payload has extra field `{}`",
                        input_field.path.join(".")
                    ),
                )
                .with_module(module.name.clone()),
            );
        }
    }

    let mut fields = Vec::new();
    for output_field in output.fields() {
        let Some(input_field) = input_fields.get(&output_field.path) else {
            continue;
        };
        if input_field.ty != output_field.ty {
            diagnostics.push(
                Diagnostic::new(
                    "E_RV_SKID_PAYLOAD_TYPE",
                    format!(
                        "ready/valid skid buffer field `{}` type mismatch: input is {:?}, output is {:?}",
                        output_field.path.join("."),
                        input_field.ty,
                        output_field.ty
                    ),
                )
                .with_module(module.name.clone()),
            );
            continue;
        }
        fields.push((
            output_field.path.clone(),
            input_field.signal,
            output_field.signal,
            output_field.ty,
        ));
    }

    if diagnostics.is_empty() {
        Ok(fields)
    } else {
        Err(diagnostics)
    }
}

fn validate_rv_fifo_bundle_payloads(
    input: &BundleRef,
    output: &BundleRef,
    module: &Module,
) -> Result<Vec<(Vec<String>, Signal, Signal, BitType)>, Vec<Diagnostic>> {
    let mut diagnostics = Vec::new();
    let input_fields = input
        .fields()
        .iter()
        .map(|field| (field.path.clone(), field))
        .collect::<HashMap<_, _>>();
    let output_fields = output
        .fields()
        .iter()
        .map(|field| (field.path.clone(), field))
        .collect::<HashMap<_, _>>();

    for output_field in output.fields() {
        if !input_fields.contains_key(&output_field.path) {
            diagnostics.push(
                Diagnostic::new(
                    "E_RV_FIFO_FIELD_MISSING",
                    format!(
                        "ready/valid FIFO input payload is missing field `{}`",
                        output_field.path.join(".")
                    ),
                )
                .with_module(module.name.clone()),
            );
        }
    }

    for input_field in input.fields() {
        if !output_fields.contains_key(&input_field.path) {
            diagnostics.push(
                Diagnostic::new(
                    "E_RV_FIFO_FIELD_EXTRA",
                    format!(
                        "ready/valid FIFO input payload has extra field `{}`",
                        input_field.path.join(".")
                    ),
                )
                .with_module(module.name.clone()),
            );
        }
    }

    let mut fields = Vec::new();
    for output_field in output.fields() {
        let Some(input_field) = input_fields.get(&output_field.path) else {
            continue;
        };
        if input_field.ty != output_field.ty {
            diagnostics.push(
                Diagnostic::new(
                    "E_RV_FIFO_PAYLOAD_TYPE",
                    format!(
                        "ready/valid FIFO field `{}` type mismatch: input is {:?}, output is {:?}",
                        output_field.path.join("."),
                        input_field.ty,
                        output_field.ty
                    ),
                )
                .with_module(module.name.clone()),
            );
            continue;
        }
        fields.push((
            output_field.path.clone(),
            input_field.signal,
            output_field.signal,
            output_field.ty,
        ));
    }

    if diagnostics.is_empty() {
        Ok(fields)
    } else {
        Err(diagnostics)
    }
}

fn validate_rv_mem_fifo_bundle_payloads(
    input: &BundleRef,
    output: &BundleRef,
    module: &Module,
) -> Result<Vec<(Vec<String>, Signal, Signal, BitType)>, Vec<Diagnostic>> {
    let mut diagnostics = Vec::new();
    let input_fields = input
        .fields()
        .iter()
        .map(|field| (field.path.clone(), field))
        .collect::<HashMap<_, _>>();
    let output_fields = output
        .fields()
        .iter()
        .map(|field| (field.path.clone(), field))
        .collect::<HashMap<_, _>>();

    for output_field in output.fields() {
        if !input_fields.contains_key(&output_field.path) {
            diagnostics.push(
                Diagnostic::new(
                    "E_RV_MEM_FIFO_FIELD_MISSING",
                    format!(
                        "ready/valid memory FIFO input payload is missing field `{}`",
                        output_field.path.join(".")
                    ),
                )
                .with_module(module.name.clone()),
            );
        }
    }

    for input_field in input.fields() {
        if !output_fields.contains_key(&input_field.path) {
            diagnostics.push(
                Diagnostic::new(
                    "E_RV_MEM_FIFO_FIELD_EXTRA",
                    format!(
                        "ready/valid memory FIFO input payload has extra field `{}`",
                        input_field.path.join(".")
                    ),
                )
                .with_module(module.name.clone()),
            );
        }
    }

    let mut fields = Vec::new();
    for output_field in output.fields() {
        let Some(input_field) = input_fields.get(&output_field.path) else {
            continue;
        };
        if input_field.ty != output_field.ty {
            diagnostics.push(
                Diagnostic::new(
                    "E_RV_MEM_FIFO_PAYLOAD_TYPE",
                    format!(
                        "ready/valid memory FIFO field `{}` type mismatch: input is {:?}, output is {:?}",
                        output_field.path.join("."),
                        input_field.ty,
                        output_field.ty
                    ),
                )
                .with_module(module.name.clone()),
            );
            continue;
        }
        fields.push((
            output_field.path.clone(),
            input_field.signal,
            output_field.signal,
            output_field.ty,
        ));
    }

    if diagnostics.is_empty() {
        Ok(fields)
    } else {
        Err(diagnostics)
    }
}

fn ceil_log2_usize(value: usize) -> Width {
    usize::BITS - (value - 1).leading_zeros()
}

fn fifo_ptr_increment(ptr: Signal, width: Width, depth: usize) -> Expr {
    mux(
        ptr.value().eq_expr(lit_u((depth - 1) as u128, width)),
        lit_u(0, width),
        ptr + lit_u(1, width),
    )
}

fn mux_fifo_entry(selector: Signal, selector_width: Width, entries: &[Signal]) -> Expr {
    let mut expr = entries[0].value();
    for (index, entry) in entries.iter().enumerate().skip(1) {
        expr = mux(
            selector
                .value()
                .eq_expr(lit_u(index as u128, selector_width)),
            *entry,
            expr,
        );
    }
    expr
}

fn bundle_packed_width(fields: &[(Vec<String>, Signal, Signal, BitType)]) -> Width {
    fields.iter().map(|(_, _, _, ty)| ty.width).sum()
}

fn unpack_bundle_field(
    read_data: Signal,
    fields: &[(Vec<String>, Signal, Signal, BitType)],
    index: usize,
) -> Expr {
    let lsb = fields[index + 1..]
        .iter()
        .map(|(_, _, _, ty)| ty.width)
        .sum();
    let ty = fields[index].3;
    let slice = read_data.value().slice(lsb, ty.width);
    match ty.signedness {
        Signedness::Unsigned => slice,
        Signedness::Signed => slice.as_sint(),
    }
}

pub fn state_type(
    name: impl Into<String>,
    ty: impl Into<BitType>,
    variants: impl IntoIterator<Item = (impl Into<String>, i128)>,
) -> StateType {
    StateType {
        name: name.into(),
        ty: ty.into(),
        variants: variants
            .into_iter()
            .map(|(name, value)| StateVariant {
                name: name.into(),
                value,
            })
            .collect(),
    }
}

impl Design {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_ir(ir: rrtl_ir::Design) -> Self {
        Self { ir }
    }

    pub fn module(&mut self, name: impl Into<String>) -> ModuleBuilder<'_> {
        let id = self.push_module(name, false);
        ModuleBuilder {
            design: self,
            module: id,
        }
    }

    pub fn extern_module(&mut self, name: impl Into<String>) -> ExternModuleBuilder<'_> {
        let id = self.push_module(name, true);
        ExternModuleBuilder {
            design: self,
            module: id,
        }
    }

    fn push_module(&mut self, name: impl Into<String>, is_external: bool) -> ModuleId {
        let id = ModuleId(self.ir.modules.len());
        self.ir.modules.push(Module {
            id,
            name: name.into(),
            is_external,
            signals: Vec::new(),
            assignments: Vec::new(),
            memory_writes: Vec::new(),
            initial_register_values: Vec::new(),
            initial_memory_values: Vec::new(),
            assertions: Vec::new(),
            cover_points: Vec::new(),
            instances: Vec::new(),
            state_types: Vec::new(),
            state_signals: Vec::new(),
            bundle_types: Vec::new(),
            bundle_signals: Vec::new(),
            interface_types: Vec::new(),
            interface_signals: Vec::new(),
            builder_diagnostics: Vec::new(),
        });
        id
    }

    pub fn validate(&self) -> Result<(), ErrorReport> {
        validate_design(&self.ir)
    }

    pub fn compile(&self) -> Result<CompiledDesign, ErrorReport> {
        compile(self)
    }

    pub fn ir(&self) -> &rrtl_ir::Design {
        &self.ir
    }

    pub fn find_signal(&self, module: &str, signal: &str) -> Option<Signal> {
        self.ir
            .find_module(module)?
            .signals
            .iter()
            .find(|s| s.name == signal)
            .map(|s| s.handle)
    }
}

pub fn compile(design: &Design) -> Result<CompiledDesign, ErrorReport> {
    compile_ir(&design.ir)
}

pub fn compile_ir(design: &rrtl_ir::Design) -> Result<CompiledDesign, ErrorReport> {
    validate_design(design)?;

    let modules = design
        .modules
        .iter()
        .map(|module| compile_module(design, module))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(CompiledDesign { modules })
}

pub struct ModuleBuilder<'a> {
    design: &'a mut Design,
    module: ModuleId,
}

pub struct ExternModuleBuilder<'a> {
    design: &'a mut Design,
    module: ModuleId,
}

impl<'a> ModuleBuilder<'a> {
    pub fn input(&mut self, name: impl Into<String>, ty: impl Into<BitType>) -> Signal {
        self.add_signal(name, ty.into(), SignalKind::Input)
    }

    pub fn input_bundle(&mut self, name: impl Into<String>, ty: BundleType) -> BundleRef {
        self.add_bundle(name, ty, SignalKind::Input)
    }

    pub fn output(&mut self, name: impl Into<String>, ty: impl Into<BitType>) -> Signal {
        self.add_signal(name, ty.into(), SignalKind::Output)
    }

    pub fn inout(&mut self, name: impl Into<String>, ty: impl Into<BitType>) -> Signal {
        self.add_signal(name, ty.into(), SignalKind::Inout)
    }

    pub fn output_bundle(&mut self, name: impl Into<String>, ty: BundleType) -> BundleRef {
        self.add_bundle(name, ty, SignalKind::Output)
    }

    pub fn wire(&mut self, name: impl Into<String>, ty: impl Into<BitType>) -> Signal {
        self.add_signal(name, ty.into(), SignalKind::Wire)
    }

    pub fn wire_bundle(&mut self, name: impl Into<String>, ty: BundleType) -> BundleRef {
        self.add_bundle(name, ty, SignalKind::Wire)
    }

    pub fn interface(&mut self, name: impl Into<String>, ty: InterfaceType) -> InterfaceRef {
        self.add_interface(name, ty)
    }

    pub fn rv_source(
        &mut self,
        name: impl Into<String>,
        payload: impl Into<InterfacePortType>,
    ) -> ReadyValidRef {
        let name = name.into();
        self.add_ready_valid(
            name.clone(),
            rv_source(name, payload),
            ReadyValidRole::Source,
        )
    }

    pub fn rv_sink(
        &mut self,
        name: impl Into<String>,
        payload: impl Into<InterfacePortType>,
    ) -> ReadyValidRef {
        let name = name.into();
        self.add_ready_valid(name.clone(), rv_sink(name, payload), ReadyValidRole::Sink)
    }

    pub fn rv_connect(&mut self, input: &ReadyValidRef, output: &ReadyValidRef) {
        if input.role() != ReadyValidRole::Sink {
            self.push_builder_diagnostic(
                Diagnostic::new(
                    "E_RV_CONNECT_ROLE",
                    "ready/valid connect input must be a sink channel",
                )
                .with_module(self.module_ref().name.clone()),
            );
            return;
        }
        if output.role() != ReadyValidRole::Source {
            self.push_builder_diagnostic(
                Diagnostic::new(
                    "E_RV_CONNECT_ROLE",
                    "ready/valid connect output must be a source channel",
                )
                .with_module(self.module_ref().name.clone()),
            );
            return;
        }

        match (input.payload(), output.payload()) {
            (InterfacePortType::Scalar(input_ty), InterfacePortType::Scalar(output_ty)) => {
                if input_ty != output_ty {
                    self.push_builder_diagnostic(
                        Diagnostic::new(
                            "E_RV_CONNECT_PAYLOAD_TYPE",
                            format!(
                                "ready/valid connect scalar payload mismatch: input is {:?}, output is {:?}",
                                input_ty, output_ty
                            ),
                        )
                        .with_module(self.module_ref().name.clone()),
                    );
                    return;
                }
                let Some(input_bits) = input.bits_signal() else {
                    self.push_builder_diagnostic(
                        Diagnostic::new(
                            "E_RV_CONNECT_PAYLOAD_KIND",
                            "ready/valid connect input bits are not scalar",
                        )
                        .with_module(self.module_ref().name.clone()),
                    );
                    return;
                };
                let Some(output_bits) = output.bits_signal() else {
                    self.push_builder_diagnostic(
                        Diagnostic::new(
                            "E_RV_CONNECT_PAYLOAD_KIND",
                            "ready/valid connect output bits are not scalar",
                        )
                        .with_module(self.module_ref().name.clone()),
                    );
                    return;
                };
                self.assign(output.valid(), input.valid());
                self.assign(input.ready(), output.ready());
                self.assign(output_bits, input_bits);
            }
            (InterfacePortType::Bundle(_), InterfacePortType::Bundle(_)) => {
                let Some(input_bits) = input.bits_bundle() else {
                    self.push_builder_diagnostic(
                        Diagnostic::new(
                            "E_RV_CONNECT_PAYLOAD_KIND",
                            "ready/valid connect input bits are not a bundle",
                        )
                        .with_module(self.module_ref().name.clone()),
                    );
                    return;
                };
                let Some(output_bits) = output.bits_bundle() else {
                    self.push_builder_diagnostic(
                        Diagnostic::new(
                            "E_RV_CONNECT_PAYLOAD_KIND",
                            "ready/valid connect output bits are not a bundle",
                        )
                        .with_module(self.module_ref().name.clone()),
                    );
                    return;
                };
                let fields = match validate_rv_connect_bundle_payloads(
                    input_bits,
                    output_bits,
                    self.module_ref(),
                ) {
                    Ok(fields) => fields,
                    Err(diagnostics) => {
                        self.module_mut().builder_diagnostics.extend(diagnostics);
                        return;
                    }
                };
                self.assign(output.valid(), input.valid());
                self.assign(input.ready(), output.ready());
                for (input_signal, output_signal) in fields {
                    self.assign(output_signal, input_signal);
                }
            }
            _ => self.push_builder_diagnostic(
                Diagnostic::new(
                    "E_RV_CONNECT_PAYLOAD_KIND",
                    "ready/valid connect payloads must both be scalar or both be bundles",
                )
                .with_module(self.module_ref().name.clone()),
            ),
        }
    }

    pub fn rv_register_slice(
        &mut self,
        name: impl Into<String>,
        input: &ReadyValidRef,
        output: &ReadyValidRef,
        clock: Signal,
        reset: Signal,
    ) {
        let name = name.into();
        if input.role() != ReadyValidRole::Sink {
            self.push_builder_diagnostic(
                Diagnostic::new(
                    "E_RV_SLICE_ROLE",
                    "ready/valid register slice input must be a sink channel",
                )
                .with_module(self.module_ref().name.clone()),
            );
            return;
        }
        if output.role() != ReadyValidRole::Source {
            self.push_builder_diagnostic(
                Diagnostic::new(
                    "E_RV_SLICE_ROLE",
                    "ready/valid register slice output must be a source channel",
                )
                .with_module(self.module_ref().name.clone()),
            );
            return;
        }

        match (input.payload(), output.payload()) {
            (InterfacePortType::Scalar(input_ty), InterfacePortType::Scalar(output_ty)) => {
                if input_ty != output_ty {
                    self.push_builder_diagnostic(
                        Diagnostic::new(
                            "E_RV_SLICE_PAYLOAD_TYPE",
                            format!(
                                "ready/valid register slice scalar payload mismatch: input is {:?}, output is {:?}",
                                input_ty, output_ty
                            ),
                        )
                        .with_module(self.module_ref().name.clone()),
                    );
                    return;
                }
                self.emit_rv_register_slice_common(&name, input, output, clock, reset);
                let Some(input_bits) = input.bits_signal() else {
                    self.push_builder_diagnostic(
                        Diagnostic::new(
                            "E_RV_SLICE_PAYLOAD_KIND",
                            "ready/valid register slice input bits are not scalar",
                        )
                        .with_module(self.module_ref().name.clone()),
                    );
                    return;
                };
                let Some(output_bits) = output.bits_signal() else {
                    self.push_builder_diagnostic(
                        Diagnostic::new(
                            "E_RV_SLICE_PAYLOAD_KIND",
                            "ready/valid register slice output bits are not scalar",
                        )
                        .with_module(self.module_ref().name.clone()),
                    );
                    return;
                };
                let bits_q = self.reg(format!("{name}_bits_q"), *input_ty);
                self.clock(bits_q, clock);
                self.reset(bits_q, reset, 0);
                self.next(bits_q, input_bits);
                self.assign(output_bits, bits_q);
            }
            (InterfacePortType::Bundle(_), InterfacePortType::Bundle(_)) => {
                let Some(input_bits) = input.bits_bundle() else {
                    self.push_builder_diagnostic(
                        Diagnostic::new(
                            "E_RV_SLICE_PAYLOAD_KIND",
                            "ready/valid register slice input bits are not a bundle",
                        )
                        .with_module(self.module_ref().name.clone()),
                    );
                    return;
                };
                let Some(output_bits) = output.bits_bundle() else {
                    self.push_builder_diagnostic(
                        Diagnostic::new(
                            "E_RV_SLICE_PAYLOAD_KIND",
                            "ready/valid register slice output bits are not a bundle",
                        )
                        .with_module(self.module_ref().name.clone()),
                    );
                    return;
                };
                let fields = match validate_rv_slice_bundle_payloads(
                    input_bits,
                    output_bits,
                    self.module_ref(),
                ) {
                    Ok(fields) => fields,
                    Err(diagnostics) => {
                        self.module_mut().builder_diagnostics.extend(diagnostics);
                        return;
                    }
                };
                self.emit_rv_register_slice_common(&name, input, output, clock, reset);
                for (path, input_signal, output_signal, ty) in fields {
                    let bits_q = self.reg(format!("{}_bits_{}_q", name, path.join("_")), ty);
                    self.clock(bits_q, clock);
                    self.reset(bits_q, reset, 0);
                    self.next(bits_q, input_signal);
                    self.assign(output_signal, bits_q);
                }
            }
            _ => self.push_builder_diagnostic(
                Diagnostic::new(
                    "E_RV_SLICE_PAYLOAD_KIND",
                    "ready/valid register slice payloads must both be scalar or both be bundles",
                )
                .with_module(self.module_ref().name.clone()),
            ),
        }
    }

    pub fn rv_skid_buffer(
        &mut self,
        name: impl Into<String>,
        input: &ReadyValidRef,
        output: &ReadyValidRef,
        clock: Signal,
        reset: Signal,
    ) {
        let name = name.into();
        if input.role() != ReadyValidRole::Sink {
            self.push_builder_diagnostic(
                Diagnostic::new(
                    "E_RV_SKID_ROLE",
                    "ready/valid skid buffer input must be a sink channel",
                )
                .with_module(self.module_ref().name.clone()),
            );
            return;
        }
        if output.role() != ReadyValidRole::Source {
            self.push_builder_diagnostic(
                Diagnostic::new(
                    "E_RV_SKID_ROLE",
                    "ready/valid skid buffer output must be a source channel",
                )
                .with_module(self.module_ref().name.clone()),
            );
            return;
        }

        match (input.payload(), output.payload()) {
            (InterfacePortType::Scalar(input_ty), InterfacePortType::Scalar(output_ty)) => {
                if input_ty != output_ty {
                    self.push_builder_diagnostic(
                        Diagnostic::new(
                            "E_RV_SKID_PAYLOAD_TYPE",
                            format!(
                                "ready/valid skid buffer scalar payload mismatch: input is {:?}, output is {:?}",
                                input_ty, output_ty
                            ),
                        )
                        .with_module(self.module_ref().name.clone()),
                    );
                    return;
                }
                let Some(input_bits) = input.bits_signal() else {
                    self.push_builder_diagnostic(
                        Diagnostic::new(
                            "E_RV_SKID_PAYLOAD_KIND",
                            "ready/valid skid buffer input bits are not scalar",
                        )
                        .with_module(self.module_ref().name.clone()),
                    );
                    return;
                };
                let Some(output_bits) = output.bits_signal() else {
                    self.push_builder_diagnostic(
                        Diagnostic::new(
                            "E_RV_SKID_PAYLOAD_KIND",
                            "ready/valid skid buffer output bits are not scalar",
                        )
                        .with_module(self.module_ref().name.clone()),
                    );
                    return;
                };

                let full_q = self.emit_rv_skid_buffer_common(&name, input, output, clock, reset);
                let bits_q = self.reg(format!("{name}_bits_q"), *input_ty);
                self.clock(bits_q, clock);
                self.reset(bits_q, reset, 0);
                self.next(
                    bits_q,
                    mux(
                        input.valid()
                            & (((!full_q.value()) & (!output.ready().value()))
                                | (full_q & output.ready())),
                        input_bits,
                        bits_q,
                    ),
                );
                self.assign(output_bits, mux(full_q, bits_q, input_bits));
            }
            (InterfacePortType::Bundle(_), InterfacePortType::Bundle(_)) => {
                let Some(input_bits) = input.bits_bundle() else {
                    self.push_builder_diagnostic(
                        Diagnostic::new(
                            "E_RV_SKID_PAYLOAD_KIND",
                            "ready/valid skid buffer input bits are not a bundle",
                        )
                        .with_module(self.module_ref().name.clone()),
                    );
                    return;
                };
                let Some(output_bits) = output.bits_bundle() else {
                    self.push_builder_diagnostic(
                        Diagnostic::new(
                            "E_RV_SKID_PAYLOAD_KIND",
                            "ready/valid skid buffer output bits are not a bundle",
                        )
                        .with_module(self.module_ref().name.clone()),
                    );
                    return;
                };
                let fields = match validate_rv_skid_bundle_payloads(
                    input_bits,
                    output_bits,
                    self.module_ref(),
                ) {
                    Ok(fields) => fields,
                    Err(diagnostics) => {
                        self.module_mut().builder_diagnostics.extend(diagnostics);
                        return;
                    }
                };

                let full_q = self.emit_rv_skid_buffer_common(&name, input, output, clock, reset);
                for (path, input_signal, output_signal, ty) in fields {
                    let bits_q = self.reg(format!("{}_bits_{}_q", name, path.join("_")), ty);
                    self.clock(bits_q, clock);
                    self.reset(bits_q, reset, 0);
                    self.next(
                        bits_q,
                        mux(
                            input.valid()
                                & (((!full_q.value()) & (!output.ready().value()))
                                    | (full_q & output.ready())),
                            input_signal,
                            bits_q,
                        ),
                    );
                    self.assign(output_signal, mux(full_q, bits_q, input_signal));
                }
            }
            _ => self.push_builder_diagnostic(
                Diagnostic::new(
                    "E_RV_SKID_PAYLOAD_KIND",
                    "ready/valid skid buffer payloads must both be scalar or both be bundles",
                )
                .with_module(self.module_ref().name.clone()),
            ),
        }
    }

    pub fn rv_fifo(
        &mut self,
        name: impl Into<String>,
        input: &ReadyValidRef,
        output: &ReadyValidRef,
        clock: Signal,
        reset: Signal,
        depth: usize,
    ) {
        let name = name.into();
        if depth < 2 {
            self.push_builder_diagnostic(
                Diagnostic::new(
                    "E_RV_FIFO_DEPTH",
                    "ready/valid FIFO depth must be at least 2",
                )
                .with_module(self.module_ref().name.clone()),
            );
            return;
        }
        let Some(count_values) = depth.checked_add(1) else {
            self.push_builder_diagnostic(
                Diagnostic::new("E_RV_FIFO_DEPTH", "ready/valid FIFO depth is too large")
                    .with_module(self.module_ref().name.clone()),
            );
            return;
        };
        if input.role() != ReadyValidRole::Sink {
            self.push_builder_diagnostic(
                Diagnostic::new(
                    "E_RV_FIFO_ROLE",
                    "ready/valid FIFO input must be a sink channel",
                )
                .with_module(self.module_ref().name.clone()),
            );
            return;
        }
        if output.role() != ReadyValidRole::Source {
            self.push_builder_diagnostic(
                Diagnostic::new(
                    "E_RV_FIFO_ROLE",
                    "ready/valid FIFO output must be a source channel",
                )
                .with_module(self.module_ref().name.clone()),
            );
            return;
        }

        match (input.payload(), output.payload()) {
            (InterfacePortType::Scalar(input_ty), InterfacePortType::Scalar(output_ty)) => {
                if input_ty != output_ty {
                    self.push_builder_diagnostic(
                        Diagnostic::new(
                            "E_RV_FIFO_PAYLOAD_TYPE",
                            format!(
                                "ready/valid FIFO scalar payload mismatch: input is {:?}, output is {:?}",
                                input_ty, output_ty
                            ),
                        )
                        .with_module(self.module_ref().name.clone()),
                    );
                    return;
                }
                let Some(input_bits) = input.bits_signal() else {
                    self.push_builder_diagnostic(
                        Diagnostic::new(
                            "E_RV_FIFO_PAYLOAD_KIND",
                            "ready/valid FIFO input bits are not scalar",
                        )
                        .with_module(self.module_ref().name.clone()),
                    );
                    return;
                };
                let Some(output_bits) = output.bits_signal() else {
                    self.push_builder_diagnostic(
                        Diagnostic::new(
                            "E_RV_FIFO_PAYLOAD_KIND",
                            "ready/valid FIFO output bits are not scalar",
                        )
                        .with_module(self.module_ref().name.clone()),
                    );
                    return;
                };

                let (wptr_q, rptr_q, write_fire, _, ptr_width) = self.emit_rv_fifo_common(
                    &name,
                    input,
                    output,
                    clock,
                    reset,
                    depth,
                    count_values,
                );
                let mut entries = Vec::new();
                for index in 0..depth {
                    let data_q = self.reg(format!("{name}_data_{index}_q"), *input_ty);
                    self.clock(data_q, clock);
                    self.reset(data_q, reset, 0);
                    self.next(
                        data_q,
                        mux(
                            write_fire.clone()
                                & wptr_q.value().eq_expr(lit_u(index as u128, ptr_width)),
                            input_bits,
                            data_q,
                        ),
                    );
                    entries.push(data_q);
                }
                self.assign(output_bits, mux_fifo_entry(rptr_q, ptr_width, &entries));
            }
            (InterfacePortType::Bundle(_), InterfacePortType::Bundle(_)) => {
                let Some(input_bits) = input.bits_bundle() else {
                    self.push_builder_diagnostic(
                        Diagnostic::new(
                            "E_RV_FIFO_PAYLOAD_KIND",
                            "ready/valid FIFO input bits are not a bundle",
                        )
                        .with_module(self.module_ref().name.clone()),
                    );
                    return;
                };
                let Some(output_bits) = output.bits_bundle() else {
                    self.push_builder_diagnostic(
                        Diagnostic::new(
                            "E_RV_FIFO_PAYLOAD_KIND",
                            "ready/valid FIFO output bits are not a bundle",
                        )
                        .with_module(self.module_ref().name.clone()),
                    );
                    return;
                };
                let fields = match validate_rv_fifo_bundle_payloads(
                    input_bits,
                    output_bits,
                    self.module_ref(),
                ) {
                    Ok(fields) => fields,
                    Err(diagnostics) => {
                        self.module_mut().builder_diagnostics.extend(diagnostics);
                        return;
                    }
                };

                let (wptr_q, rptr_q, write_fire, _, ptr_width) = self.emit_rv_fifo_common(
                    &name,
                    input,
                    output,
                    clock,
                    reset,
                    depth,
                    count_values,
                );
                for (path, input_signal, output_signal, ty) in fields {
                    let mut entries = Vec::new();
                    for index in 0..depth {
                        let data_q =
                            self.reg(format!("{}_data_{}_{}_q", name, index, path.join("_")), ty);
                        self.clock(data_q, clock);
                        self.reset(data_q, reset, 0);
                        self.next(
                            data_q,
                            mux(
                                write_fire.clone()
                                    & wptr_q.value().eq_expr(lit_u(index as u128, ptr_width)),
                                input_signal,
                                data_q,
                            ),
                        );
                        entries.push(data_q);
                    }
                    self.assign(output_signal, mux_fifo_entry(rptr_q, ptr_width, &entries));
                }
            }
            _ => self.push_builder_diagnostic(
                Diagnostic::new(
                    "E_RV_FIFO_PAYLOAD_KIND",
                    "ready/valid FIFO payloads must both be scalar or both be bundles",
                )
                .with_module(self.module_ref().name.clone()),
            ),
        }
    }

    pub fn rv_mem_fifo(
        &mut self,
        name: impl Into<String>,
        input: &ReadyValidRef,
        output: &ReadyValidRef,
        clock: Signal,
        reset: Signal,
        depth: usize,
    ) {
        let name = name.into();
        if depth < 2 {
            self.push_builder_diagnostic(
                Diagnostic::new(
                    "E_RV_MEM_FIFO_DEPTH",
                    "ready/valid memory FIFO depth must be at least 2",
                )
                .with_module(self.module_ref().name.clone()),
            );
            return;
        }
        let Some(count_values) = depth.checked_add(1) else {
            self.push_builder_diagnostic(
                Diagnostic::new(
                    "E_RV_MEM_FIFO_DEPTH",
                    "ready/valid memory FIFO depth is too large",
                )
                .with_module(self.module_ref().name.clone()),
            );
            return;
        };
        if input.role() != ReadyValidRole::Sink {
            self.push_builder_diagnostic(
                Diagnostic::new(
                    "E_RV_MEM_FIFO_ROLE",
                    "ready/valid memory FIFO input must be a sink channel",
                )
                .with_module(self.module_ref().name.clone()),
            );
            return;
        }
        if output.role() != ReadyValidRole::Source {
            self.push_builder_diagnostic(
                Diagnostic::new(
                    "E_RV_MEM_FIFO_ROLE",
                    "ready/valid memory FIFO output must be a source channel",
                )
                .with_module(self.module_ref().name.clone()),
            );
            return;
        }

        match (input.payload(), output.payload()) {
            (InterfacePortType::Scalar(input_ty), InterfacePortType::Scalar(output_ty)) => {
                if input_ty != output_ty {
                    self.push_builder_diagnostic(
                        Diagnostic::new(
                            "E_RV_MEM_FIFO_PAYLOAD_TYPE",
                            format!(
                                "ready/valid memory FIFO scalar payload mismatch: input is {:?}, output is {:?}",
                                input_ty, output_ty
                            ),
                        )
                        .with_module(self.module_ref().name.clone()),
                    );
                    return;
                }
                let Some(input_bits) = input.bits_signal() else {
                    self.push_builder_diagnostic(
                        Diagnostic::new(
                            "E_RV_MEM_FIFO_PAYLOAD_KIND",
                            "ready/valid memory FIFO input bits are not scalar",
                        )
                        .with_module(self.module_ref().name.clone()),
                    );
                    return;
                };
                let Some(output_bits) = output.bits_signal() else {
                    self.push_builder_diagnostic(
                        Diagnostic::new(
                            "E_RV_MEM_FIFO_PAYLOAD_KIND",
                            "ready/valid memory FIFO output bits are not scalar",
                        )
                        .with_module(self.module_ref().name.clone()),
                    );
                    return;
                };

                let (wptr_q, rptr_q, write_fire, _, ptr_width) = self.emit_rv_fifo_common(
                    &name,
                    input,
                    output,
                    clock,
                    reset,
                    depth,
                    count_values,
                );
                let mem = self.mem(format!("{name}_mem"), ptr_width, *input_ty, depth);
                let read_data = self.wire(format!("{name}_read_data"), *input_ty);
                self.mem_write(mem, clock, write_fire, wptr_q, input_bits);
                let read_expr = self.mem_read(mem, rptr_q);
                self.assign(read_data, read_expr);
                self.assign(output_bits, read_data);
            }
            (InterfacePortType::Bundle(_), InterfacePortType::Bundle(_)) => {
                let Some(input_bits) = input.bits_bundle() else {
                    self.push_builder_diagnostic(
                        Diagnostic::new(
                            "E_RV_MEM_FIFO_PAYLOAD_KIND",
                            "ready/valid memory FIFO input bits are not a bundle",
                        )
                        .with_module(self.module_ref().name.clone()),
                    );
                    return;
                };
                let Some(output_bits) = output.bits_bundle() else {
                    self.push_builder_diagnostic(
                        Diagnostic::new(
                            "E_RV_MEM_FIFO_PAYLOAD_KIND",
                            "ready/valid memory FIFO output bits are not a bundle",
                        )
                        .with_module(self.module_ref().name.clone()),
                    );
                    return;
                };
                let fields = match validate_rv_mem_fifo_bundle_payloads(
                    input_bits,
                    output_bits,
                    self.module_ref(),
                ) {
                    Ok(fields) => fields,
                    Err(diagnostics) => {
                        self.module_mut().builder_diagnostics.extend(diagnostics);
                        return;
                    }
                };

                let (wptr_q, rptr_q, write_fire, _, ptr_width) = self.emit_rv_fifo_common(
                    &name,
                    input,
                    output,
                    clock,
                    reset,
                    depth,
                    count_values,
                );
                let packed_width = bundle_packed_width(&fields);
                let mem = self.mem(format!("{name}_mem"), ptr_width, uint(packed_width), depth);
                let read_data = self.wire(format!("{name}_read_data"), uint(packed_width));
                let packed = concat(
                    fields
                        .iter()
                        .map(|(_, input_signal, _, _)| input_signal.value())
                        .collect::<Vec<_>>(),
                );
                self.mem_write(mem, clock, write_fire, wptr_q, packed);
                let read_expr = self.mem_read(mem, rptr_q);
                self.assign(read_data, read_expr);
                for (index, (_, _, output_signal, _)) in fields.iter().enumerate() {
                    self.assign(
                        *output_signal,
                        unpack_bundle_field(read_data, &fields, index),
                    );
                }
            }
            _ => self.push_builder_diagnostic(
                Diagnostic::new(
                    "E_RV_MEM_FIFO_PAYLOAD_KIND",
                    "ready/valid memory FIFO payloads must both be scalar or both be bundles",
                )
                .with_module(self.module_ref().name.clone()),
            ),
        }
    }

    fn emit_rv_register_slice_common(
        &mut self,
        name: &str,
        input: &ReadyValidRef,
        output: &ReadyValidRef,
        clock: Signal,
        reset: Signal,
    ) {
        self.assign(input.ready(), output.ready());
        let valid_q = self.reg(format!("{name}_valid_q"), uint(1));
        self.clock(valid_q, clock);
        self.reset(valid_q, reset, 0);
        self.next(valid_q, input.valid());
        self.assign(output.valid(), valid_q);
    }

    fn emit_rv_skid_buffer_common(
        &mut self,
        name: &str,
        input: &ReadyValidRef,
        output: &ReadyValidRef,
        clock: Signal,
        reset: Signal,
    ) -> Signal {
        let full_q = self.reg(format!("{name}_full_q"), uint(1));
        self.clock(full_q, clock);
        self.reset(full_q, reset, 0);
        self.next(
            full_q,
            mux(
                output.ready(),
                full_q & input.valid(),
                full_q | input.valid(),
            ),
        );
        self.assign(input.ready(), (!full_q.value()) | output.ready());
        self.assign(output.valid(), full_q | input.valid());
        full_q
    }

    fn emit_rv_fifo_common(
        &mut self,
        name: &str,
        input: &ReadyValidRef,
        output: &ReadyValidRef,
        clock: Signal,
        reset: Signal,
        depth: usize,
        count_values: usize,
    ) -> (Signal, Signal, Expr, Expr, Width) {
        let ptr_width = ceil_log2_usize(depth);
        let count_width = ceil_log2_usize(count_values);
        let count_q = self.reg(format!("{name}_count_q"), uint(count_width));
        let wptr_q = self.reg(format!("{name}_wptr_q"), uint(ptr_width));
        let rptr_q = self.reg(format!("{name}_rptr_q"), uint(ptr_width));
        for reg in [count_q, wptr_q, rptr_q] {
            self.clock(reg, clock);
            self.reset(reg, reset, 0);
        }

        let count_not_full = count_q.value().ne_expr(lit_u(depth as u128, count_width));
        let count_non_empty = count_q.value().ne_expr(lit_u(0, count_width));
        let write_fire = input.valid() & count_not_full.clone();
        let read_fire = count_non_empty.clone() & output.ready();

        self.assign(input.ready(), count_not_full);
        self.assign(output.valid(), count_non_empty);

        self.next(
            count_q,
            mux(
                write_fire.clone() & (!read_fire.clone()),
                count_q + lit_u(1, count_width),
                mux(
                    read_fire.clone() & (!write_fire.clone()),
                    count_q - lit_u(1, count_width),
                    count_q,
                ),
            ),
        );

        self.next(
            wptr_q,
            mux(
                write_fire.clone(),
                fifo_ptr_increment(wptr_q, ptr_width, depth),
                wptr_q,
            ),
        );
        self.next(
            rptr_q,
            mux(
                read_fire.clone(),
                fifo_ptr_increment(rptr_q, ptr_width, depth),
                rptr_q,
            ),
        );

        (wptr_q, rptr_q, write_fire, read_fire, ptr_width)
    }

    pub fn reg(&mut self, name: impl Into<String>, ty: impl Into<BitType>) -> Signal {
        self.add_signal(
            name,
            ty.into(),
            SignalKind::Reg {
                clock: None,
                reset: None,
                next: None,
            },
        )
    }

    pub fn state_reg(
        &mut self,
        name: impl Into<String>,
        state_type: StateType,
        clock: Signal,
        reset: Signal,
        reset_variant: &str,
    ) -> StateReg {
        self.register_state_type(state_type.clone());
        let signal = self.reg(name, state_type.ty);
        self.clock(signal, clock);
        match state_type.value_of(reset_variant) {
            Some(value) => self.reset(signal, reset, value),
            None => self.push_builder_diagnostic(
                Diagnostic::new(
                    "E_STATE_RESET_VARIANT",
                    format!(
                        "state type `{}` has no reset variant `{}`",
                        state_type.name, reset_variant
                    ),
                )
                .with_module(self.module_ref().name.clone()),
            ),
        }
        self.module_mut().state_signals.push(StateSignal {
            signal,
            state_type: state_type.clone(),
            reset_variant: reset_variant.to_string(),
        });
        StateReg { signal, state_type }
    }

    pub fn state_reg_reset_low(
        &mut self,
        name: impl Into<String>,
        state_type: StateType,
        clock: Signal,
        reset: Signal,
        reset_variant: &str,
    ) -> StateReg {
        self.register_state_type(state_type.clone());
        let signal = self.reg(name, state_type.ty);
        self.clock(signal, clock);
        match state_type.value_of(reset_variant) {
            Some(value) => self.reset_low(signal, reset, value),
            None => self.push_builder_diagnostic(
                Diagnostic::new(
                    "E_STATE_RESET_VARIANT",
                    format!(
                        "state type `{}` has no reset variant `{}`",
                        state_type.name, reset_variant
                    ),
                )
                .with_module(self.module_ref().name.clone()),
            ),
        }
        self.module_mut().state_signals.push(StateSignal {
            signal,
            state_type: state_type.clone(),
            reset_variant: reset_variant.to_string(),
        });
        StateReg { signal, state_type }
    }

    pub fn state_reg_async_reset(
        &mut self,
        name: impl Into<String>,
        state_type: StateType,
        clock: Signal,
        reset: Signal,
        reset_variant: &str,
    ) -> StateReg {
        self.register_state_type(state_type.clone());
        let signal = self.reg(name, state_type.ty);
        self.clock(signal, clock);
        match state_type.value_of(reset_variant) {
            Some(value) => self.async_reset(signal, reset, value),
            None => self.push_builder_diagnostic(
                Diagnostic::new(
                    "E_STATE_RESET_VARIANT",
                    format!(
                        "state type `{}` has no reset variant `{}`",
                        state_type.name, reset_variant
                    ),
                )
                .with_module(self.module_ref().name.clone()),
            ),
        }
        self.module_mut().state_signals.push(StateSignal {
            signal,
            state_type: state_type.clone(),
            reset_variant: reset_variant.to_string(),
        });
        StateReg { signal, state_type }
    }

    pub fn state_reg_async_reset_low(
        &mut self,
        name: impl Into<String>,
        state_type: StateType,
        clock: Signal,
        reset: Signal,
        reset_variant: &str,
    ) -> StateReg {
        self.register_state_type(state_type.clone());
        let signal = self.reg(name, state_type.ty);
        self.clock(signal, clock);
        match state_type.value_of(reset_variant) {
            Some(value) => self.async_reset_low(signal, reset, value),
            None => self.push_builder_diagnostic(
                Diagnostic::new(
                    "E_STATE_RESET_VARIANT",
                    format!(
                        "state type `{}` has no reset variant `{}`",
                        state_type.name, reset_variant
                    ),
                )
                .with_module(self.module_ref().name.clone()),
            ),
        }
        self.module_mut().state_signals.push(StateSignal {
            signal,
            state_type: state_type.clone(),
            reset_variant: reset_variant.to_string(),
        });
        StateReg { signal, state_type }
    }

    pub fn state_next(
        &mut self,
        state: &StateReg,
        default_variant: &str,
        transitions: impl IntoIterator<Item = (impl Into<Expr>, impl Into<String>)>,
    ) {
        let Some(mut expr) = state.lit(default_variant) else {
            self.push_builder_diagnostic(
                Diagnostic::new(
                    "E_STATE_NEXT_VARIANT",
                    format!(
                        "state type `{}` has no default variant `{}`",
                        state.state_type.name, default_variant
                    ),
                )
                .with_module(self.module_ref().name.clone()),
            );
            return;
        };

        let transitions = transitions
            .into_iter()
            .map(|(cond, variant)| (cond.into(), variant.into()))
            .collect::<Vec<(Expr, String)>>();
        for (cond, variant) in transitions.into_iter().rev() {
            let Some(next) = state.lit(&variant) else {
                self.push_builder_diagnostic(
                    Diagnostic::new(
                        "E_STATE_NEXT_VARIANT",
                        format!(
                            "state type `{}` has no transition variant `{}`",
                            state.state_type.name, variant
                        ),
                    )
                    .with_module(self.module_ref().name.clone()),
                );
                return;
            };
            expr = mux(cond, next, expr);
        }
        self.next(state.signal, expr);
    }

    pub fn state_next_hold(
        &mut self,
        state: &StateReg,
        transitions: impl IntoIterator<Item = (impl Into<Expr>, impl Into<String>)>,
    ) {
        let mut expr = state.value();
        let transitions = transitions
            .into_iter()
            .map(|(cond, variant)| (cond.into(), variant.into()))
            .collect::<Vec<(Expr, String)>>();
        for (cond, variant) in transitions.into_iter().rev() {
            let Some(next) = state.lit(&variant) else {
                self.push_builder_diagnostic(
                    Diagnostic::new(
                        "E_STATE_NEXT_VARIANT",
                        format!(
                            "state type `{}` has no transition variant `{}`",
                            state.state_type.name, variant
                        ),
                    )
                    .with_module(self.module_ref().name.clone()),
                );
                return;
            };
            expr = mux(cond, next, expr);
        }
        self.next(state.signal, expr);
    }

    pub fn mem(
        &mut self,
        name: impl Into<String>,
        addr_width: Width,
        data_ty: impl Into<BitType>,
        depth: usize,
    ) -> Signal {
        let data_ty = data_ty.into();
        self.add_signal(
            name,
            data_ty,
            SignalKind::Mem {
                addr_width,
                data_width: data_ty.width,
                depth,
            },
        )
    }

    pub fn mem_read(&mut self, mem: Signal, addr: impl Into<Expr>) -> Expr {
        mem_read(mem, addr)
    }

    pub fn mem_write(
        &mut self,
        mem: Signal,
        clock: Signal,
        enable: impl Into<Expr>,
        addr: impl Into<Expr>,
        data: impl Into<Expr>,
    ) {
        self.module_mut().memory_writes.push(MemoryWrite {
            mem,
            clock,
            enable: enable.into(),
            addr: addr.into(),
            data: data.into(),
        });
    }

    pub fn initial(&mut self, reg: Signal, value: i128) {
        self.module_mut()
            .initial_register_values
            .push(InitialRegisterValue { signal: reg, value });
    }

    pub fn mem_init(&mut self, mem: Signal, addr: usize, value: i128) {
        self.module_mut()
            .initial_memory_values
            .push(InitialMemoryValue { mem, addr, value });
    }

    pub fn assign(&mut self, dst: Signal, expr: impl Into<Expr>) {
        self.module_mut().assignments.push(Assignment {
            dst,
            expr: expr.into(),
        });
    }

    pub fn assign_bundle(&mut self, dst: &BundleRef, src: &BundleRef) {
        let mut diagnostics = Vec::new();
        let assignments =
            expand_bundle_assignments(self.module_ref(), dst, src, None, &mut diagnostics);
        if diagnostics.is_empty() {
            self.module_mut().assignments.extend(assignments);
        } else {
            self.module_mut().builder_diagnostics.extend(diagnostics);
        }
    }

    pub fn assign_bundle_when(
        &mut self,
        dst: &BundleRef,
        enable: impl Into<Expr>,
        src: &BundleRef,
    ) {
        let mut diagnostics = Vec::new();
        let assignments = expand_bundle_assignments(
            self.module_ref(),
            dst,
            src,
            Some(enable.into()),
            &mut diagnostics,
        );
        if diagnostics.is_empty() {
            self.module_mut().assignments.extend(assignments);
        } else {
            self.module_mut().builder_diagnostics.extend(diagnostics);
        }
    }

    pub fn assert(&mut self, name: impl Into<String>, condition: impl Into<Expr>) {
        self.module_mut().assertions.push(Assertion {
            name: name.into(),
            clock: None,
            enable: None,
            condition: condition.into(),
            message: None,
        });
    }

    pub fn assert_msg(
        &mut self,
        name: impl Into<String>,
        condition: impl Into<Expr>,
        message: impl Into<String>,
    ) {
        self.module_mut().assertions.push(Assertion {
            name: name.into(),
            clock: None,
            enable: None,
            condition: condition.into(),
            message: Some(message.into()),
        });
    }

    pub fn assert_when(
        &mut self,
        name: impl Into<String>,
        enable: impl Into<Expr>,
        condition: impl Into<Expr>,
    ) {
        self.module_mut().assertions.push(Assertion {
            name: name.into(),
            clock: None,
            enable: Some(enable.into()),
            condition: condition.into(),
            message: None,
        });
    }

    pub fn assert_when_msg(
        &mut self,
        name: impl Into<String>,
        enable: impl Into<Expr>,
        condition: impl Into<Expr>,
        message: impl Into<String>,
    ) {
        self.module_mut().assertions.push(Assertion {
            name: name.into(),
            clock: None,
            enable: Some(enable.into()),
            condition: condition.into(),
            message: Some(message.into()),
        });
    }

    pub fn assert_clocked(
        &mut self,
        name: impl Into<String>,
        clock: Signal,
        condition: impl Into<Expr>,
    ) {
        self.module_mut().assertions.push(Assertion {
            name: name.into(),
            clock: Some(clock),
            enable: None,
            condition: condition.into(),
            message: None,
        });
    }

    pub fn assert_clocked_msg(
        &mut self,
        name: impl Into<String>,
        clock: Signal,
        condition: impl Into<Expr>,
        message: impl Into<String>,
    ) {
        self.module_mut().assertions.push(Assertion {
            name: name.into(),
            clock: Some(clock),
            enable: None,
            condition: condition.into(),
            message: Some(message.into()),
        });
    }

    pub fn cover(&mut self, name: impl Into<String>, condition: impl Into<Expr>) {
        self.module_mut().cover_points.push(CoverPoint {
            name: name.into(),
            clock: None,
            enable: None,
            condition: condition.into(),
            message: None,
        });
    }

    pub fn cover_msg(
        &mut self,
        name: impl Into<String>,
        condition: impl Into<Expr>,
        message: impl Into<String>,
    ) {
        self.module_mut().cover_points.push(CoverPoint {
            name: name.into(),
            clock: None,
            enable: None,
            condition: condition.into(),
            message: Some(message.into()),
        });
    }

    pub fn cover_when(
        &mut self,
        name: impl Into<String>,
        enable: impl Into<Expr>,
        condition: impl Into<Expr>,
    ) {
        self.module_mut().cover_points.push(CoverPoint {
            name: name.into(),
            clock: None,
            enable: Some(enable.into()),
            condition: condition.into(),
            message: None,
        });
    }

    pub fn cover_when_msg(
        &mut self,
        name: impl Into<String>,
        enable: impl Into<Expr>,
        condition: impl Into<Expr>,
        message: impl Into<String>,
    ) {
        self.module_mut().cover_points.push(CoverPoint {
            name: name.into(),
            clock: None,
            enable: Some(enable.into()),
            condition: condition.into(),
            message: Some(message.into()),
        });
    }

    pub fn cover_clocked(
        &mut self,
        name: impl Into<String>,
        clock: Signal,
        condition: impl Into<Expr>,
    ) {
        self.module_mut().cover_points.push(CoverPoint {
            name: name.into(),
            clock: Some(clock),
            enable: None,
            condition: condition.into(),
            message: None,
        });
    }

    pub fn cover_clocked_msg(
        &mut self,
        name: impl Into<String>,
        clock: Signal,
        condition: impl Into<Expr>,
        message: impl Into<String>,
    ) {
        self.module_mut().cover_points.push(CoverPoint {
            name: name.into(),
            clock: Some(clock),
            enable: None,
            condition: condition.into(),
            message: Some(message.into()),
        });
    }

    pub fn clock(&mut self, reg: Signal, clock: Signal) {
        let module_name = self.module_ref().name.clone();
        let Some(info) = self.module_mut().signal_mut(reg) else {
            self.push_builder_diagnostic(
                Diagnostic::new(
                    "E_BUILDER_REG_HANDLE",
                    "clock target is not a signal in this module",
                )
                .with_module(module_name),
            );
            return;
        };
        if let SignalKind::Reg { clock: c, .. } = &mut info.kind {
            *c = Some(clock);
        } else {
            let signal_name = info.name.clone();
            self.push_builder_diagnostic(
                Diagnostic::new("E_BUILDER_REG_KIND", "clock target is not a register")
                    .with_module(module_name)
                    .with_signal(signal_name),
            );
        }
    }

    pub fn reset(&mut self, reg: Signal, reset: Signal, value: i128) {
        self.set_reset(
            reg,
            reset,
            value,
            ResetKind::Sync,
            ResetPolarity::ActiveHigh,
        );
    }

    pub fn reset_low(&mut self, reg: Signal, reset: Signal, value: i128) {
        self.set_reset(reg, reset, value, ResetKind::Sync, ResetPolarity::ActiveLow);
    }

    pub fn async_reset(&mut self, reg: Signal, reset: Signal, value: i128) {
        self.set_reset(
            reg,
            reset,
            value,
            ResetKind::Async,
            ResetPolarity::ActiveHigh,
        );
    }

    pub fn async_reset_low(&mut self, reg: Signal, reset: Signal, value: i128) {
        self.set_reset(
            reg,
            reset,
            value,
            ResetKind::Async,
            ResetPolarity::ActiveLow,
        );
    }

    fn set_reset(
        &mut self,
        reg: Signal,
        reset: Signal,
        value: i128,
        kind: ResetKind,
        polarity: ResetPolarity,
    ) {
        let module_name = self.module_ref().name.clone();
        let Some(info) = self.module_mut().signal_mut(reg) else {
            self.push_builder_diagnostic(
                Diagnostic::new(
                    "E_BUILDER_REG_HANDLE",
                    "reset target is not a signal in this module",
                )
                .with_module(module_name),
            );
            return;
        };
        let mut diagnostic = None;
        if let SignalKind::Reg { reset: r, .. } = &mut info.kind {
            if let Some(existing) = r
                .as_ref()
                .filter(|existing| existing.kind != kind || existing.polarity != polarity)
            {
                diagnostic = Some(
                    Diagnostic::new(
                        "E_BUILDER_RESET_KIND",
                        format!(
                            "register reset changed from {:?} {:?} to {:?} {:?}",
                            existing.kind, existing.polarity, kind, polarity
                        ),
                    )
                    .with_module(module_name.clone())
                    .with_signal(info.name.clone()),
                );
            }
            *r = Some(Reset {
                signal: reset,
                value,
                kind,
                polarity,
            });
        } else {
            let signal_name = info.name.clone();
            self.push_builder_diagnostic(
                Diagnostic::new("E_BUILDER_REG_KIND", "reset target is not a register")
                    .with_module(module_name)
                    .with_signal(signal_name),
            );
        }
        if let Some(diagnostic) = diagnostic {
            self.push_builder_diagnostic(diagnostic);
        }
    }

    pub fn next(&mut self, reg: Signal, expr: impl Into<Expr>) {
        let module_name = self.module_ref().name.clone();
        let Some(info) = self.module_mut().signal_mut(reg) else {
            self.push_builder_diagnostic(
                Diagnostic::new(
                    "E_BUILDER_REG_HANDLE",
                    "next target is not a signal in this module",
                )
                .with_module(module_name),
            );
            return;
        };
        if let SignalKind::Reg { next, .. } = &mut info.kind {
            *next = Some(expr.into());
        } else {
            let signal_name = info.name.clone();
            self.push_builder_diagnostic(
                Diagnostic::new("E_BUILDER_REG_KIND", "next target is not a register")
                    .with_module(module_name)
                    .with_signal(signal_name),
            );
        }
    }

    pub fn instance(
        &mut self,
        name: impl Into<String>,
        module: impl Into<String>,
        connections: impl IntoIterator<Item = (impl Into<String>, Signal)>,
    ) {
        self.module_mut().instances.push(Instance {
            name: name.into(),
            module: module.into(),
            connections: connections
                .into_iter()
                .map(|(port, signal)| (port.into(), signal))
                .collect(),
        });
    }

    pub fn instance_bundles<N, M, I, P>(&mut self, name: N, module: M, bundles: I)
    where
        N: Into<String>,
        M: Into<String>,
        I: IntoIterator<Item = (P, BundleRef)>,
        P: Into<String>,
    {
        let instance_name = name.into();
        let module_name = module.into();
        let bundles = bundles
            .into_iter()
            .map(|(port, bundle)| (port.into(), bundle))
            .collect::<Vec<_>>();
        let Some(target) = self.design.ir.find_module(&module_name).cloned() else {
            self.push_builder_diagnostic(
                Diagnostic::new(
                    "E_BUNDLE_INSTANCE_TARGET",
                    format!("bundle instance target module `{module_name}` does not exist"),
                )
                .with_module(self.module_ref().name.clone()),
            );
            return;
        };

        let mut diagnostics = Vec::new();
        let mut connections = Vec::new();
        for (port_name, bundle_ref) in bundles {
            let Some(target_bundle) = target
                .bundle_signals
                .iter()
                .find(|bundle| bundle.name == port_name)
            else {
                diagnostics.push(
                    Diagnostic::new(
                        "E_BUNDLE_INSTANCE_PORT",
                        format!(
                            "target module `{}` has no bundle port named `{}`",
                            target.name, port_name
                        ),
                    )
                    .with_module(self.module_ref().name.clone()),
                );
                continue;
            };

            expand_bundle_connection(
                self.module_ref(),
                &target,
                target_bundle,
                &bundle_ref,
                &mut connections,
                &mut diagnostics,
            );
        }

        if diagnostics.is_empty() {
            self.module_mut().instances.push(Instance {
                name: instance_name,
                module: module_name,
                connections,
            });
        } else {
            self.module_mut().builder_diagnostics.extend(diagnostics);
        }
    }

    pub fn instance_interfaces<N, M, I, P>(&mut self, name: N, module: M, interfaces: I)
    where
        N: Into<String>,
        M: Into<String>,
        I: IntoIterator<Item = (P, InterfaceRef)>,
        P: Into<String>,
    {
        let instance_name = name.into();
        let module_name = module.into();
        let interfaces = interfaces
            .into_iter()
            .map(|(port, interface)| (port.into(), interface))
            .collect::<Vec<_>>();
        let Some(target) = self.design.ir.find_module(&module_name).cloned() else {
            self.push_builder_diagnostic(
                Diagnostic::new(
                    "E_INTERFACE_INSTANCE_TARGET",
                    format!("interface instance target module `{module_name}` does not exist"),
                )
                .with_module(self.module_ref().name.clone()),
            );
            return;
        };

        let mut diagnostics = Vec::new();
        let mut connections = Vec::new();
        for (port_name, interface_ref) in interfaces {
            let Some(target_interface) = target
                .interface_signals
                .iter()
                .find(|interface| interface.name == port_name)
            else {
                diagnostics.push(
                    Diagnostic::new(
                        "E_INTERFACE_INSTANCE_PORT",
                        format!(
                            "target module `{}` has no interface port named `{}`",
                            target.name, port_name
                        ),
                    )
                    .with_module(self.module_ref().name.clone()),
                );
                continue;
            };

            expand_interface_connection(
                self.module_ref(),
                &target,
                target_interface,
                &interface_ref,
                &mut connections,
                &mut diagnostics,
            );
        }

        if diagnostics.is_empty() {
            self.module_mut().instances.push(Instance {
                name: instance_name,
                module: module_name,
                connections,
            });
        } else {
            self.module_mut().builder_diagnostics.extend(diagnostics);
        }
    }

    fn add_signal(&mut self, name: impl Into<String>, ty: BitType, kind: SignalKind) -> Signal {
        let id = SignalId(self.module_ref().signals.len());
        let handle = Signal {
            module: self.module,
            id,
        };
        self.module_mut().signals.push(SignalInfo {
            handle,
            name: name.into(),
            width: ty.width,
            ty,
            kind,
        });
        handle
    }

    fn add_bundle(
        &mut self,
        name: impl Into<String>,
        bundle_type: BundleType,
        kind: SignalKind,
    ) -> BundleRef {
        let name = name.into();
        self.register_bundle_type(bundle_type.clone());

        let mut fields = Vec::new();
        let mut metadata = Vec::new();
        for leaf in bundle_type.leaf_fields() {
            let signal_name = format!("{}_{}", name, leaf.path.join("_"));
            let signal = self.add_signal(signal_name, leaf.ty, kind.clone());
            fields.push(BundleRefField {
                path: leaf.path.clone(),
                signal,
                ty: leaf.ty,
            });
            metadata.push(BundleSignalField {
                path: leaf.path,
                signal,
            });
        }

        self.module_mut().bundle_signals.push(BundleSignal {
            name: name.clone(),
            bundle_type: bundle_type.clone(),
            fields: metadata,
        });

        BundleRef {
            name,
            bundle_type,
            fields,
        }
    }

    fn add_interface(
        &mut self,
        name: impl Into<String>,
        interface_type: InterfaceType,
    ) -> InterfaceRef {
        let name = name.into();
        self.register_interface_type(interface_type.clone());

        let mut ports = Vec::new();
        let mut metadata_ports = Vec::new();
        for port in &interface_type.ports {
            let kind = interface_direction_to_signal_kind(port.direction);
            match &port.ty {
                InterfacePortType::Scalar(ty) => {
                    let signal = self.add_signal(format!("{}_{}", name, port.name), *ty, kind);
                    ports.push(PortRef {
                        name: port.name.clone(),
                        direction: port.direction,
                        ty: port.ty.clone(),
                        signal: Some(signal),
                        bundle: None,
                    });
                    metadata_ports.push(InterfaceSignalPort {
                        name: port.name.clone(),
                        direction: port.direction,
                        signals: InterfaceSignalPortSignals::Scalar { signal },
                    });
                }
                InterfacePortType::Bundle(bundle_type) => {
                    self.register_bundle_type(bundle_type.clone());
                    let bundle = self.add_interface_bundle_port(
                        &format!("{}_{}", name, port.name),
                        bundle_type.clone(),
                        kind,
                    );
                    metadata_ports.push(InterfaceSignalPort {
                        name: port.name.clone(),
                        direction: port.direction,
                        signals: InterfaceSignalPortSignals::Bundle {
                            fields: bundle
                                .fields()
                                .iter()
                                .map(|field| BundleSignalField {
                                    path: field.path.clone(),
                                    signal: field.signal,
                                })
                                .collect(),
                        },
                    });
                    ports.push(PortRef {
                        name: port.name.clone(),
                        direction: port.direction,
                        ty: port.ty.clone(),
                        signal: None,
                        bundle: Some(bundle),
                    });
                }
            }
        }

        self.module_mut().interface_signals.push(InterfaceSignal {
            name: name.clone(),
            interface_type: interface_type.clone(),
            ports: metadata_ports,
        });

        InterfaceRef {
            name,
            interface_type,
            ports,
        }
    }

    fn add_ready_valid(
        &mut self,
        name: String,
        interface_type: InterfaceType,
        role: ReadyValidRole,
    ) -> ReadyValidRef {
        let payload = match validate_ready_valid_type(&interface_type, role) {
            Ok(payload) => payload,
            Err(err) => {
                let module_name = self.module_ref().name.clone();
                self.module_mut()
                    .builder_diagnostics
                    .extend(err.diagnostics.into_iter().map(|diagnostic| {
                        if diagnostic.module.is_none() {
                            diagnostic.with_module(module_name.clone())
                        } else {
                            diagnostic
                        }
                    }));
                InterfacePortType::Scalar(uint(1))
            }
        };
        let interface = self.add_interface(name, interface_type);
        ReadyValidRef {
            interface,
            role,
            payload,
        }
    }

    fn add_interface_bundle_port(
        &mut self,
        name: &str,
        bundle_type: BundleType,
        kind: SignalKind,
    ) -> BundleRef {
        let mut fields = Vec::new();
        for leaf in bundle_type.leaf_fields() {
            let signal_name = format!("{}_{}", name, leaf.path.join("_"));
            let signal = self.add_signal(signal_name, leaf.ty, kind.clone());
            fields.push(BundleRefField {
                path: leaf.path,
                signal,
                ty: leaf.ty,
            });
        }
        BundleRef {
            name: name.to_string(),
            bundle_type,
            fields,
        }
    }

    fn register_state_type(&mut self, state_type: StateType) {
        if !self
            .module_ref()
            .state_types
            .iter()
            .any(|existing| existing.name == state_type.name)
        {
            self.module_mut().state_types.push(state_type);
        }
    }

    fn register_bundle_type(&mut self, bundle_type: BundleType) {
        if let Some(existing) = self
            .module_ref()
            .bundle_types
            .iter()
            .find(|existing| existing.name == bundle_type.name)
        {
            if existing != &bundle_type {
                self.push_builder_diagnostic(
                    Diagnostic::new(
                        "E_BUNDLE_TYPE_CONFLICT",
                        format!(
                            "bundle type `{}` was registered with a different shape",
                            bundle_type.name
                        ),
                    )
                    .with_module(self.module_ref().name.clone()),
                );
            }
        } else {
            self.module_mut().bundle_types.push(bundle_type);
        }
    }

    fn register_interface_type(&mut self, interface_type: InterfaceType) {
        if let Some(existing) = self
            .module_ref()
            .interface_types
            .iter()
            .find(|existing| existing.name == interface_type.name)
        {
            if existing != &interface_type {
                self.push_builder_diagnostic(
                    Diagnostic::new(
                        "E_INTERFACE_TYPE_CONFLICT",
                        format!(
                            "interface type `{}` was registered with a different shape",
                            interface_type.name
                        ),
                    )
                    .with_module(self.module_ref().name.clone()),
                );
            }
        } else {
            self.module_mut().interface_types.push(interface_type);
        }
    }

    fn module_ref(&self) -> &Module {
        self.design.ir.module(self.module).expect("module exists")
    }

    fn module_mut(&mut self) -> &mut Module {
        self.design
            .ir
            .module_mut(self.module)
            .expect("module exists")
    }

    fn push_builder_diagnostic(&mut self, diagnostic: Diagnostic) {
        self.module_mut().builder_diagnostics.push(diagnostic);
    }
}

impl<'a> ExternModuleBuilder<'a> {
    pub fn input(&mut self, name: impl Into<String>, ty: impl Into<BitType>) -> Signal {
        self.add_signal(name, ty.into(), SignalKind::Input)
    }

    pub fn output(&mut self, name: impl Into<String>, ty: impl Into<BitType>) -> Signal {
        self.add_signal(name, ty.into(), SignalKind::Output)
    }

    pub fn inout(&mut self, name: impl Into<String>, ty: impl Into<BitType>) -> Signal {
        self.add_signal(name, ty.into(), SignalKind::Inout)
    }

    fn add_signal(&mut self, name: impl Into<String>, ty: BitType, kind: SignalKind) -> Signal {
        let id = SignalId(self.module_ref().signals.len());
        let handle = Signal {
            module: self.module,
            id,
        };
        self.module_mut().signals.push(SignalInfo {
            handle,
            name: name.into(),
            width: ty.width,
            ty,
            kind,
        });
        handle
    }

    fn module_ref(&self) -> &Module {
        self.design.ir.module(self.module).expect("module exists")
    }

    fn module_mut(&mut self) -> &mut Module {
        self.design
            .ir
            .module_mut(self.module)
            .expect("module exists")
    }
}

fn expand_bundle_connection(
    parent: &Module,
    target: &Module,
    target_bundle: &BundleSignal,
    parent_bundle: &BundleRef,
    connections: &mut Vec<(String, Signal)>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let target_order = target_bundle
        .fields
        .iter()
        .filter_map(|field| {
            let signal = target.signal(field.signal)?;
            Some((field.path.clone(), signal.name.clone(), signal.ty))
        })
        .collect::<Vec<_>>();
    let target_fields = target_order
        .iter()
        .map(|(path, signal_name, ty)| (path.clone(), (signal_name.clone(), *ty)))
        .collect::<HashMap<Vec<String>, (String, BitType)>>();
    let parent_fields = parent_bundle
        .fields()
        .iter()
        .map(|field| (field.path.clone(), (field.signal, field.ty)))
        .collect::<HashMap<Vec<String>, (Signal, BitType)>>();

    for (path, _, _) in &target_order {
        if !parent_fields.contains_key(path) {
            diagnostics.push(
                Diagnostic::new(
                    "E_BUNDLE_CONNECT_MISSING",
                    format!(
                        "bundle `{}` is missing field `{}` required by target bundle `{}`",
                        parent_bundle.name,
                        path.join("."),
                        target_bundle.name
                    ),
                )
                .with_module(parent.name.clone()),
            );
        }
    }

    for path in parent_fields.keys() {
        if !target_fields.contains_key(path) {
            diagnostics.push(
                Diagnostic::new(
                    "E_BUNDLE_CONNECT_EXTRA",
                    format!(
                        "bundle `{}` has extra field `{}` not present in target bundle `{}`",
                        parent_bundle.name,
                        path.join("."),
                        target_bundle.name
                    ),
                )
                .with_module(parent.name.clone()),
            );
        }
    }

    for (path, target_signal, target_ty) in target_order {
        let Some((parent_signal, parent_ty)) = parent_fields.get(&path) else {
            continue;
        };
        if *parent_ty != target_ty {
            diagnostics.push(
                Diagnostic::new(
                    "E_BUNDLE_CONNECT_TYPE",
                    format!(
                        "bundle field `{}` type mismatch: target is {:?}, parent is {:?}",
                        path.join("."),
                        target_ty,
                        parent_ty
                    ),
                )
                .with_module(parent.name.clone()),
            );
            continue;
        }
        connections.push((target_signal, *parent_signal));
    }
}

fn expand_bundle_assignments(
    module: &Module,
    dst: &BundleRef,
    src: &BundleRef,
    enable: Option<Expr>,
    diagnostics: &mut Vec<Diagnostic>,
) -> Vec<Assignment> {
    let dst_order = dst
        .fields()
        .iter()
        .map(|field| (field.path.clone(), field.signal, field.ty))
        .collect::<Vec<_>>();
    let dst_fields = dst_order
        .iter()
        .map(|(path, signal, ty)| (path.clone(), (*signal, *ty)))
        .collect::<HashMap<Vec<String>, (Signal, BitType)>>();
    let src_fields = src
        .fields()
        .iter()
        .map(|field| (field.path.clone(), (field.signal, field.ty)))
        .collect::<HashMap<Vec<String>, (Signal, BitType)>>();

    for (path, _, _) in &dst_order {
        if !src_fields.contains_key(path) {
            diagnostics.push(
                Diagnostic::new(
                    "E_BUNDLE_ASSIGN_MISSING",
                    format!(
                        "source bundle `{}` is missing field `{}` required by destination bundle `{}`",
                        src.name,
                        path.join("."),
                        dst.name
                    ),
                )
                .with_module(module.name.clone()),
            );
        }
    }

    for path in src_fields.keys() {
        if !dst_fields.contains_key(path) {
            diagnostics.push(
                Diagnostic::new(
                    "E_BUNDLE_ASSIGN_EXTRA",
                    format!(
                        "source bundle `{}` has extra field `{}` not present in destination bundle `{}`",
                        src.name,
                        path.join("."),
                        dst.name
                    ),
                )
                .with_module(module.name.clone()),
            );
        }
    }

    let mut assignments = Vec::new();
    for (path, dst_signal, dst_ty) in dst_order {
        let Some((src_signal, src_ty)) = src_fields.get(&path) else {
            continue;
        };
        if *src_ty != dst_ty {
            diagnostics.push(
                Diagnostic::new(
                    "E_BUNDLE_ASSIGN_TYPE",
                    format!(
                        "bundle field `{}` type mismatch: destination is {:?}, source is {:?}",
                        path.join("."),
                        dst_ty,
                        src_ty
                    ),
                )
                .with_module(module.name.clone()),
            );
            continue;
        }

        let expr = match &enable {
            Some(enable) => mux(enable.clone(), *src_signal, dst_signal.value()),
            None => (*src_signal).into(),
        };
        assignments.push(Assignment {
            dst: dst_signal,
            expr,
        });
    }

    assignments
}

fn expand_interface_connection(
    parent: &Module,
    target: &Module,
    target_interface: &InterfaceSignal,
    parent_interface: &InterfaceRef,
    connections: &mut Vec<(String, Signal)>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let target_ports = target_interface
        .ports
        .iter()
        .map(|port| (port.name.clone(), port))
        .collect::<HashMap<_, _>>();
    let parent_ports = parent_interface
        .ports()
        .iter()
        .map(|port| (port.name.clone(), port))
        .collect::<HashMap<_, _>>();

    for target_port in &target_interface.ports {
        if !parent_ports.contains_key(&target_port.name) {
            diagnostics.push(
                Diagnostic::new(
                    "E_INTERFACE_CONNECT_MISSING",
                    format!(
                        "interface `{}` is missing port `{}` required by target interface `{}`",
                        parent_interface.name, target_port.name, target_interface.name
                    ),
                )
                .with_module(parent.name.clone()),
            );
        }
    }

    for parent_port in parent_interface.ports() {
        if !target_ports.contains_key(&parent_port.name) {
            diagnostics.push(
                Diagnostic::new(
                    "E_INTERFACE_CONNECT_EXTRA",
                    format!(
                        "interface `{}` has extra port `{}` not present in target interface `{}`",
                        parent_interface.name, parent_port.name, target_interface.name
                    ),
                )
                .with_module(parent.name.clone()),
            );
        }
    }

    for target_port in &target_interface.ports {
        let Some(parent_port) = parent_ports.get(&target_port.name) else {
            continue;
        };
        if parent_port.direction != target_port.direction {
            diagnostics.push(
                Diagnostic::new(
                    "E_INTERFACE_CONNECT_DIRECTION",
                    format!(
                        "interface port `{}` direction mismatch: target is {:?}, parent is {:?}",
                        target_port.name, target_port.direction, parent_port.direction
                    ),
                )
                .with_module(parent.name.clone()),
            );
            continue;
        }

        match (
            &target_port.signals,
            parent_port.signal(),
            parent_port.bundle(),
        ) {
            (InterfaceSignalPortSignals::Scalar { signal }, Some(parent_signal), None) => {
                let Some(target_signal) = target.signal(*signal) else {
                    continue;
                };
                if let Some(parent_ty) = signal_type(parent, parent_signal, diagnostics) {
                    if parent_ty != target_signal.ty {
                        diagnostics.push(
                            Diagnostic::new(
                                "E_INTERFACE_CONNECT_TYPE",
                                format!(
                                    "interface port `{}` type mismatch: target is {:?}, parent is {:?}",
                                    target_port.name, target_signal.ty, parent_ty
                                ),
                            )
                            .with_module(parent.name.clone()),
                        );
                        continue;
                    }
                }
                connections.push((target_signal.name.clone(), parent_signal));
            }
            (InterfaceSignalPortSignals::Bundle { fields }, None, Some(parent_bundle)) => {
                expand_interface_bundle_connection(
                    parent,
                    target,
                    &target_port.name,
                    fields,
                    parent_bundle,
                    connections,
                    diagnostics,
                );
            }
            (InterfaceSignalPortSignals::Scalar { .. }, _, _) => diagnostics.push(
                Diagnostic::new(
                    "E_INTERFACE_CONNECT_KIND",
                    format!(
                        "interface port `{}` expects a scalar connection",
                        target_port.name
                    ),
                )
                .with_module(parent.name.clone()),
            ),
            (InterfaceSignalPortSignals::Bundle { .. }, _, _) => diagnostics.push(
                Diagnostic::new(
                    "E_INTERFACE_CONNECT_KIND",
                    format!(
                        "interface port `{}` expects a bundle connection",
                        target_port.name
                    ),
                )
                .with_module(parent.name.clone()),
            ),
        }
    }
}

fn expand_interface_bundle_connection(
    parent: &Module,
    target: &Module,
    port_name: &str,
    target_fields: &[BundleSignalField],
    parent_bundle: &BundleRef,
    connections: &mut Vec<(String, Signal)>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let target_order = target_fields
        .iter()
        .filter_map(|field| {
            let signal = target.signal(field.signal)?;
            Some((field.path.clone(), signal.name.clone(), signal.ty))
        })
        .collect::<Vec<_>>();
    let target_paths = target_order
        .iter()
        .map(|(path, _, ty)| (path.clone(), *ty))
        .collect::<HashMap<Vec<String>, BitType>>();
    let parent_fields = parent_bundle
        .fields()
        .iter()
        .map(|field| (field.path.clone(), (field.signal, field.ty)))
        .collect::<HashMap<Vec<String>, (Signal, BitType)>>();

    for (path, _, _) in &target_order {
        if !parent_fields.contains_key(path) {
            diagnostics.push(
                Diagnostic::new(
                    "E_INTERFACE_CONNECT_MISSING",
                    format!(
                        "interface bundle port `{}` is missing field `{}`",
                        port_name,
                        path.join(".")
                    ),
                )
                .with_module(parent.name.clone()),
            );
        }
    }

    for path in parent_fields.keys() {
        if !target_paths.contains_key(path) {
            diagnostics.push(
                Diagnostic::new(
                    "E_INTERFACE_CONNECT_EXTRA",
                    format!(
                        "interface bundle port `{}` has extra field `{}`",
                        port_name,
                        path.join(".")
                    ),
                )
                .with_module(parent.name.clone()),
            );
        }
    }

    for (path, target_signal, target_ty) in target_order {
        let Some((parent_signal, parent_ty)) = parent_fields.get(&path) else {
            continue;
        };
        if *parent_ty != target_ty {
            diagnostics.push(
                Diagnostic::new(
                    "E_INTERFACE_CONNECT_TYPE",
                    format!(
                        "interface bundle field `{}.{}` type mismatch: target is {:?}, parent is {:?}",
                        port_name,
                        path.join("."),
                        target_ty,
                        parent_ty
                    ),
                )
                .with_module(parent.name.clone()),
            );
            continue;
        }
        connections.push((target_signal, *parent_signal));
    }
}

fn interface_direction_to_signal_kind(direction: InterfaceDirection) -> SignalKind {
    match direction {
        InterfaceDirection::Input => SignalKind::Input,
        InterfaceDirection::Output => SignalKind::Output,
    }
}

pub fn validate_design(design: &rrtl_ir::Design) -> Result<(), ErrorReport> {
    let mut diagnostics = Vec::new();
    validate_module_names(design, &mut diagnostics);
    validate_hierarchy_cycles(design, &mut diagnostics);
    for module in &design.modules {
        validate_module(design, module, &mut diagnostics);
    }

    if diagnostics.is_empty() {
        Ok(())
    } else {
        Err(ErrorReport::new(diagnostics))
    }
}

fn compile_module(
    design: &rrtl_ir::Design,
    module: &Module,
) -> Result<CompiledModule, ErrorReport> {
    let mut diagnostics = Vec::new();
    let assignments = module
        .assignments
        .iter()
        .filter_map(|assignment| {
            Some(CompiledAssignment {
                dst: assignment.dst,
                expr: compile_expr(module, &assignment.expr, &mut diagnostics)?,
            })
        })
        .collect();

    let registers = module
        .signals
        .iter()
        .filter_map(|signal| {
            let SignalKind::Reg { clock, reset, next } = &signal.kind else {
                return None;
            };
            Some(CompiledRegister {
                signal: signal.handle,
                clock: clock.as_ref().copied()?,
                reset: reset.clone(),
                next: compile_expr(module, next.as_ref()?, &mut diagnostics)?,
            })
        })
        .collect();

    let memory_writes = module
        .memory_writes
        .iter()
        .filter_map(|write| {
            Some(CompiledMemoryWrite {
                mem: write.mem,
                clock: write.clock,
                enable: compile_expr(module, &write.enable, &mut diagnostics)?,
                addr: compile_expr(module, &write.addr, &mut diagnostics)?,
                data: compile_expr(module, &write.data, &mut diagnostics)?,
            })
        })
        .collect();

    let assertions = module
        .assertions
        .iter()
        .filter_map(|assertion| {
            Some(CompiledAssertion {
                name: assertion.name.clone(),
                clock: assertion.clock,
                enable: match &assertion.enable {
                    Some(enable) => Some(compile_expr(module, enable, &mut diagnostics)?),
                    None => None,
                },
                condition: compile_expr(module, &assertion.condition, &mut diagnostics)?,
                message: assertion.message.clone(),
            })
        })
        .collect();

    let cover_points = module
        .cover_points
        .iter()
        .filter_map(|cover| {
            Some(CompiledCoverPoint {
                name: cover.name.clone(),
                clock: cover.clock,
                enable: match &cover.enable {
                    Some(enable) => Some(compile_expr(module, enable, &mut diagnostics)?),
                    None => None,
                },
                condition: compile_expr(module, &cover.condition, &mut diagnostics)?,
                message: cover.message.clone(),
            })
        })
        .collect();

    let instances = module
        .instances
        .iter()
        .filter_map(|instance| compile_instance(design, module, instance, &mut diagnostics))
        .collect();

    if diagnostics.is_empty() {
        Ok(CompiledModule {
            id: module.id,
            name: module.name.clone(),
            is_external: module.is_external,
            signals: module.signals.clone(),
            assignments,
            registers,
            memory_writes,
            initial_register_values: module.initial_register_values.clone(),
            initial_memory_values: module.initial_memory_values.clone(),
            assertions,
            cover_points,
            instances,
            state_types: module.state_types.clone(),
            state_signals: module.state_signals.clone(),
            bundle_types: module.bundle_types.clone(),
            bundle_signals: module.bundle_signals.clone(),
            interface_types: module.interface_types.clone(),
            interface_signals: module.interface_signals.clone(),
        })
    } else {
        Err(ErrorReport::new(diagnostics))
    }
}

fn compile_instance(
    design: &rrtl_ir::Design,
    module: &Module,
    instance: &Instance,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<CompiledInstance> {
    let target = design.find_module(&instance.module)?;
    let connections = instance
        .connections
        .iter()
        .filter_map(|(port_name, signal)| {
            let port = target.signals.iter().find(|s| {
                s.name == *port_name
                    && matches!(
                        s.kind,
                        SignalKind::Input | SignalKind::Output | SignalKind::Inout
                    )
            })?;
            let direction = match port.kind {
                SignalKind::Input => PortDirection::Input,
                SignalKind::Output => PortDirection::Output,
                SignalKind::Inout => PortDirection::Inout,
                _ => return None,
            };
            if module.signal(*signal).is_none() {
                diagnostics.push(
                    Diagnostic::new("E_COMPILE_INSTANCE_SIGNAL", "instance signal is invalid")
                        .with_module(module.name.clone()),
                );
                return None;
            }
            Some(CompiledConnection {
                port: port_name.clone(),
                signal: *signal,
                direction,
                width: port.width,
                ty: port.ty,
            })
        })
        .collect();

    Some(CompiledInstance {
        name: instance.name.clone(),
        module: instance.module.clone(),
        connections,
    })
}

fn compile_expr(
    module: &Module,
    expr: &Expr,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<CompiledExpr> {
    expr_type(module, expr, diagnostics).map(|ty| CompiledExpr {
        expr: expr.clone(),
        width: ty.width,
        ty,
    })
}

fn validate_module_names(design: &rrtl_ir::Design, diagnostics: &mut Vec<Diagnostic>) {
    let mut seen = HashSet::new();
    for module in &design.modules {
        if module.name.is_empty() {
            diagnostics.push(Diagnostic::new(
                "E_MODULE_NAME_EMPTY",
                "module name is empty",
            ));
        }
        if !seen.insert(module.name.clone()) {
            diagnostics.push(
                Diagnostic::new(
                    "E_MODULE_NAME_DUP",
                    format!("duplicate module name `{}`", module.name),
                )
                .with_module(module.name.clone()),
            );
        }
    }
}

fn validate_hierarchy_cycles(design: &rrtl_ir::Design, diagnostics: &mut Vec<Diagnostic>) {
    let module_names = design
        .modules
        .iter()
        .map(|module| module.name.as_str())
        .collect::<HashSet<_>>();
    let mut graph: HashMap<&str, Vec<&str>> = HashMap::new();
    for module in &design.modules {
        if module.is_external {
            continue;
        }
        for instance in &module.instances {
            if module_names.contains(instance.module.as_str()) {
                graph
                    .entry(module.name.as_str())
                    .or_default()
                    .push(instance.module.as_str());
            }
        }
    }

    let mut visited = HashSet::new();
    for module in &design.modules {
        let mut visiting = HashSet::new();
        if has_hierarchy_cycle(module.name.as_str(), &graph, &mut visiting, &mut visited) {
            diagnostics.push(
                Diagnostic::new(
                    "E_HIERARCHY_CYCLE",
                    format!("module `{}` participates in an instance cycle", module.name),
                )
                .with_module(module.name.clone()),
            );
        }
    }
}

fn has_hierarchy_cycle<'a>(
    module: &'a str,
    graph: &HashMap<&'a str, Vec<&'a str>>,
    visiting: &mut HashSet<&'a str>,
    visited: &mut HashSet<&'a str>,
) -> bool {
    if visited.contains(module) {
        return false;
    }
    if !visiting.insert(module) {
        return true;
    }
    for dep in graph.get(module).into_iter().flatten().copied() {
        if has_hierarchy_cycle(dep, graph, visiting, visited) {
            return true;
        }
    }
    visiting.remove(module);
    visited.insert(module);
    false
}

fn validate_module(design: &rrtl_ir::Design, module: &Module, diagnostics: &mut Vec<Diagnostic>) {
    diagnostics.extend(module.builder_diagnostics.clone());
    validate_signal_names(module, diagnostics);
    validate_signal_widths(module, diagnostics);
    if module.is_external {
        validate_external_module(module, diagnostics);
        return;
    }
    validate_state_types(module, diagnostics);
    validate_bundle_types(module, diagnostics);
    validate_interface_types(module, diagnostics);
    validate_assignments(module, diagnostics);
    validate_memory_writes(module, diagnostics);
    validate_initial_values(module, diagnostics);
    validate_assertions(module, diagnostics);
    validate_cover_points(module, diagnostics);
    validate_registers(module, diagnostics);
    validate_instances(design, module, diagnostics);
    validate_comb_cycles(design, module, diagnostics);
}

fn validate_external_module(module: &Module, diagnostics: &mut Vec<Diagnostic>) {
    if !module.assertions.is_empty() {
        diagnostics.push(
            Diagnostic::new(
                "E_EXTERN_MODULE_ASSERTION",
                "external modules may not contain assertions",
            )
            .with_module(module.name.clone()),
        );
    }
    if !module.cover_points.is_empty() {
        diagnostics.push(
            Diagnostic::new(
                "E_EXTERN_MODULE_COVER",
                "external modules may not contain cover points",
            )
            .with_module(module.name.clone()),
        );
    }
    if !module.initial_register_values.is_empty() || !module.initial_memory_values.is_empty() {
        diagnostics.push(
            Diagnostic::new(
                "E_EXTERN_MODULE_INITIAL",
                "external modules may not contain initial values",
            )
            .with_module(module.name.clone()),
        );
    }
    for signal in &module.signals {
        if !matches!(
            signal.kind,
            SignalKind::Input | SignalKind::Output | SignalKind::Inout
        ) {
            diagnostics.push(
                Diagnostic::new(
                    "E_EXTERN_MODULE_SIGNAL",
                    "external modules may only contain input, output, and inout ports",
                )
                .with_module(module.name.clone())
                .with_signal(signal.name.clone()),
            );
        }
    }

    if !module.assignments.is_empty()
        || !module.memory_writes.is_empty()
        || !module.instances.is_empty()
        || !module.state_types.is_empty()
        || !module.state_signals.is_empty()
        || !module.bundle_types.is_empty()
        || !module.bundle_signals.is_empty()
        || !module.interface_types.is_empty()
        || !module.interface_signals.is_empty()
    {
        diagnostics.push(
            Diagnostic::new(
                "E_EXTERN_MODULE_BODY",
                "external modules may only declare scalar input, output, and inout ports",
            )
            .with_module(module.name.clone()),
        );
    }
}

fn validate_state_types(module: &Module, diagnostics: &mut Vec<Diagnostic>) {
    let mut type_names = HashSet::new();
    let mut sv_type_names = HashSet::new();
    for state_type in &module.state_types {
        if state_type.name.is_empty() {
            diagnostics.push(
                Diagnostic::new("E_STATE_TYPE_NAME", "state type name is empty")
                    .with_module(module.name.clone()),
            );
        }
        if !type_names.insert(state_type.name.clone()) {
            diagnostics.push(
                Diagnostic::new(
                    "E_STATE_TYPE_DUP",
                    format!("duplicate state type `{}`", state_type.name),
                )
                .with_module(module.name.clone()),
            );
        }
        let sv_type_name = state_sv_type_name(&state_type.name);
        if !sv_type_names.insert(sv_type_name.clone()) {
            diagnostics.push(
                Diagnostic::new(
                    "E_STATE_TYPE_SV_NAME_DUP",
                    format!(
                        "state type `{}` sanitizes to duplicate SystemVerilog enum type `{}`",
                        state_type.name, sv_type_name
                    ),
                )
                .with_module(module.name.clone()),
            );
        }
        if state_type.ty.width == 0 {
            diagnostics.push(
                Diagnostic::new(
                    "E_STATE_TYPE_WIDTH",
                    format!("state type `{}` must have non-zero width", state_type.name),
                )
                .with_module(module.name.clone()),
            );
        }
        let mut names = HashSet::new();
        let mut sv_names = HashSet::new();
        let mut values = HashSet::new();
        for variant in &state_type.variants {
            if variant.name.is_empty() {
                diagnostics.push(
                    Diagnostic::new(
                        "E_STATE_VARIANT_NAME",
                        format!("state type `{}` has an empty variant name", state_type.name),
                    )
                    .with_module(module.name.clone()),
                );
            }
            if !names.insert(variant.name.clone()) {
                diagnostics.push(
                    Diagnostic::new(
                        "E_STATE_VARIANT_DUP",
                        format!(
                            "state type `{}` has duplicate variant `{}`",
                            state_type.name, variant.name
                        ),
                    )
                    .with_module(module.name.clone()),
                );
            }
            let sv_variant_name = format!(
                "{}_{}",
                sanitize_sv_ident(&state_type.name),
                sanitize_sv_ident(&variant.name)
            );
            if !sv_names.insert(sv_variant_name.clone()) {
                diagnostics.push(
                    Diagnostic::new(
                        "E_STATE_VARIANT_SV_NAME_DUP",
                        format!(
                            "state variant `{}.{}` sanitizes to duplicate SystemVerilog enum literal `{}`",
                            state_type.name, variant.name, sv_variant_name
                        ),
                    )
                    .with_module(module.name.clone()),
                );
            }
            if !values.insert(variant.value) {
                diagnostics.push(
                    Diagnostic::new(
                        "E_STATE_ENCODING_DUP",
                        format!(
                            "state type `{}` has duplicate encoding `{}`",
                            state_type.name, variant.value
                        ),
                    )
                    .with_module(module.name.clone()),
                );
            }
            if !fits_type(variant.value, state_type.ty) {
                diagnostics.push(
                    Diagnostic::new(
                        "E_STATE_ENCODING_RANGE",
                        format!(
                            "state variant `{}.{}` value {} does not fit in {:?}",
                            state_type.name, variant.name, variant.value, state_type.ty
                        ),
                    )
                    .with_module(module.name.clone()),
                );
            }
        }
    }

    for state_signal in &module.state_signals {
        let Some(signal) = module.signal(state_signal.signal) else {
            diagnostics.push(
                Diagnostic::new(
                    "E_STATE_SIGNAL",
                    "state signal does not belong to this module",
                )
                .with_module(module.name.clone()),
            );
            continue;
        };
        if signal.ty != state_signal.state_type.ty {
            diagnostics.push(
                Diagnostic::new(
                    "E_STATE_SIGNAL_TYPE",
                    format!(
                        "state signal `{}` has type {:?}, expected {:?}",
                        signal.name, signal.ty, state_signal.state_type.ty
                    ),
                )
                .with_module(module.name.clone())
                .with_signal(signal.name.clone()),
            );
        }
        if state_signal
            .state_type
            .value_of(&state_signal.reset_variant)
            .is_none()
        {
            diagnostics.push(
                Diagnostic::new(
                    "E_STATE_RESET_VARIANT",
                    format!(
                        "state type `{}` has no reset variant `{}`",
                        state_signal.state_type.name, state_signal.reset_variant
                    ),
                )
                .with_module(module.name.clone())
                .with_signal(signal.name.clone()),
            );
        }
    }
}

fn state_sv_type_name(name: &str) -> String {
    format!("{}_t", sanitize_sv_ident(name))
}

fn sanitize_sv_ident(name: &str) -> String {
    name.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect()
}

fn validate_bundle_types(module: &Module, diagnostics: &mut Vec<Diagnostic>) {
    let mut type_names = HashSet::new();
    for bundle_type in &module.bundle_types {
        if bundle_type.name.is_empty() {
            diagnostics.push(
                Diagnostic::new("E_BUNDLE_TYPE_NAME", "bundle type name is empty")
                    .with_module(module.name.clone()),
            );
        }
        if !type_names.insert(bundle_type.name.clone()) {
            diagnostics.push(
                Diagnostic::new(
                    "E_BUNDLE_TYPE_DUP",
                    format!("duplicate bundle type `{}`", bundle_type.name),
                )
                .with_module(module.name.clone()),
            );
        }
        validate_bundle_type_fields(module, bundle_type, &bundle_type.name, diagnostics);
    }

    let mut bundle_signal_names = HashSet::new();
    for bundle_signal in &module.bundle_signals {
        if bundle_signal.name.is_empty() {
            diagnostics.push(
                Diagnostic::new("E_BUNDLE_SIGNAL_NAME", "bundle signal name is empty")
                    .with_module(module.name.clone()),
            );
        }
        if !bundle_signal_names.insert(bundle_signal.name.clone()) {
            diagnostics.push(
                Diagnostic::new(
                    "E_BUNDLE_SIGNAL_DUP",
                    format!("duplicate bundle signal `{}`", bundle_signal.name),
                )
                .with_module(module.name.clone()),
            );
        }

        let expected = bundle_signal
            .bundle_type
            .leaf_fields()
            .into_iter()
            .map(|leaf| (leaf.path, leaf.ty))
            .collect::<HashMap<Vec<String>, BitType>>();
        let mut seen_paths = HashSet::new();
        for field in &bundle_signal.fields {
            if !seen_paths.insert(field.path.clone()) {
                diagnostics.push(
                    Diagnostic::new(
                        "E_BUNDLE_SIGNAL_FIELD_DUP",
                        format!(
                            "bundle signal `{}` has duplicate field `{}`",
                            bundle_signal.name,
                            field.path.join(".")
                        ),
                    )
                    .with_module(module.name.clone()),
                );
            }

            let Some(expected_ty) = expected.get(&field.path) else {
                diagnostics.push(
                    Diagnostic::new(
                        "E_BUNDLE_SIGNAL_FIELD",
                        format!(
                            "bundle signal `{}` has unexpected field `{}`",
                            bundle_signal.name,
                            field.path.join(".")
                        ),
                    )
                    .with_module(module.name.clone()),
                );
                continue;
            };

            let Some(signal) = module.signal(field.signal) else {
                diagnostics.push(
                    Diagnostic::new(
                        "E_BUNDLE_SIGNAL_FIELD",
                        format!(
                            "bundle signal `{}` field `{}` references an invalid signal",
                            bundle_signal.name,
                            field.path.join(".")
                        ),
                    )
                    .with_module(module.name.clone()),
                );
                continue;
            };

            if signal.ty != *expected_ty {
                diagnostics.push(
                    Diagnostic::new(
                        "E_BUNDLE_SIGNAL_FIELD_TYPE",
                        format!(
                            "bundle signal `{}` field `{}` has type {:?}, expected {:?}",
                            bundle_signal.name,
                            field.path.join("."),
                            signal.ty,
                            expected_ty
                        ),
                    )
                    .with_module(module.name.clone())
                    .with_signal(signal.name.clone()),
                );
            }
        }

        for path in expected.keys() {
            if !seen_paths.contains(path) {
                diagnostics.push(
                    Diagnostic::new(
                        "E_BUNDLE_SIGNAL_FIELD_MISSING",
                        format!(
                            "bundle signal `{}` is missing field `{}`",
                            bundle_signal.name,
                            path.join(".")
                        ),
                    )
                    .with_module(module.name.clone()),
                );
            }
        }
    }
}

fn validate_bundle_type_fields(
    module: &Module,
    bundle_type: &BundleType,
    context: &str,
    diagnostics: &mut Vec<Diagnostic>,
) {
    if bundle_type.fields.is_empty() {
        diagnostics.push(
            Diagnostic::new(
                "E_BUNDLE_TYPE_EMPTY",
                format!("bundle type `{context}` has no fields"),
            )
            .with_module(module.name.clone()),
        );
    }

    let mut field_names = HashSet::new();
    for field in &bundle_type.fields {
        match field {
            BundleField::Leaf { name, ty } => {
                validate_bundle_field_name(module, context, name, &mut field_names, diagnostics);
                if ty.width == 0 {
                    diagnostics.push(
                        Diagnostic::new(
                            "E_BUNDLE_FIELD_WIDTH",
                            format!("bundle field `{context}.{name}` must have non-zero width"),
                        )
                        .with_module(module.name.clone()),
                    );
                }
            }
            BundleField::Nested { name, bundle } => {
                validate_bundle_field_name(module, context, name, &mut field_names, diagnostics);
                if bundle.name.is_empty() {
                    diagnostics.push(
                        Diagnostic::new(
                            "E_BUNDLE_TYPE_NAME",
                            format!(
                                "nested bundle field `{context}.{name}` has an empty type name"
                            ),
                        )
                        .with_module(module.name.clone()),
                    );
                }
                validate_bundle_type_fields(
                    module,
                    bundle,
                    &format!("{context}.{name}"),
                    diagnostics,
                );
            }
        }
    }
}

fn validate_bundle_field_name(
    module: &Module,
    context: &str,
    name: &str,
    field_names: &mut HashSet<String>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    if name.is_empty() {
        diagnostics.push(
            Diagnostic::new(
                "E_BUNDLE_FIELD_NAME",
                format!("bundle type `{context}` has an empty field name"),
            )
            .with_module(module.name.clone()),
        );
    }
    if !field_names.insert(name.to_string()) {
        diagnostics.push(
            Diagnostic::new(
                "E_BUNDLE_FIELD_DUP",
                format!("bundle type `{context}` has duplicate field `{name}`"),
            )
            .with_module(module.name.clone()),
        );
    }
}

fn validate_interface_types(module: &Module, diagnostics: &mut Vec<Diagnostic>) {
    let mut type_names = HashSet::new();
    for interface_type in &module.interface_types {
        if interface_type.name.is_empty() {
            diagnostics.push(
                Diagnostic::new("E_INTERFACE_TYPE_NAME", "interface type name is empty")
                    .with_module(module.name.clone()),
            );
        }
        if !type_names.insert(interface_type.name.clone()) {
            diagnostics.push(
                Diagnostic::new(
                    "E_INTERFACE_TYPE_DUP",
                    format!("duplicate interface type `{}`", interface_type.name),
                )
                .with_module(module.name.clone()),
            );
        }
        if interface_type.ports.is_empty() {
            diagnostics.push(
                Diagnostic::new(
                    "E_INTERFACE_TYPE_EMPTY",
                    format!("interface type `{}` has no ports", interface_type.name),
                )
                .with_module(module.name.clone()),
            );
        }

        let mut port_names = HashSet::new();
        for port in &interface_type.ports {
            validate_interface_port(module, interface_type, port, &mut port_names, diagnostics);
        }
    }

    let mut interface_signal_names = HashSet::new();
    for interface_signal in &module.interface_signals {
        if interface_signal.name.is_empty() {
            diagnostics.push(
                Diagnostic::new("E_INTERFACE_SIGNAL_NAME", "interface signal name is empty")
                    .with_module(module.name.clone()),
            );
        }
        if !interface_signal_names.insert(interface_signal.name.clone()) {
            diagnostics.push(
                Diagnostic::new(
                    "E_INTERFACE_SIGNAL_DUP",
                    format!("duplicate interface signal `{}`", interface_signal.name),
                )
                .with_module(module.name.clone()),
            );
        }

        let expected = interface_signal
            .interface_type
            .ports
            .iter()
            .map(|port| (port.name.clone(), port))
            .collect::<HashMap<_, _>>();
        let mut seen_ports = HashSet::new();
        for port_signal in &interface_signal.ports {
            if !seen_ports.insert(port_signal.name.clone()) {
                diagnostics.push(
                    Diagnostic::new(
                        "E_INTERFACE_SIGNAL_PORT_DUP",
                        format!(
                            "interface signal `{}` has duplicate port `{}`",
                            interface_signal.name, port_signal.name
                        ),
                    )
                    .with_module(module.name.clone()),
                );
            }

            let Some(expected_port) = expected.get(&port_signal.name) else {
                diagnostics.push(
                    Diagnostic::new(
                        "E_INTERFACE_SIGNAL_PORT",
                        format!(
                            "interface signal `{}` has unexpected port `{}`",
                            interface_signal.name, port_signal.name
                        ),
                    )
                    .with_module(module.name.clone()),
                );
                continue;
            };

            if port_signal.direction != expected_port.direction {
                diagnostics.push(
                    Diagnostic::new(
                        "E_INTERFACE_SIGNAL_DIRECTION",
                        format!(
                            "interface signal `{}` port `{}` has direction {:?}, expected {:?}",
                            interface_signal.name,
                            port_signal.name,
                            port_signal.direction,
                            expected_port.direction
                        ),
                    )
                    .with_module(module.name.clone()),
                );
            }

            validate_interface_signal_port(
                module,
                interface_signal,
                expected_port,
                port_signal,
                diagnostics,
            );
        }

        for port_name in expected.keys() {
            if !seen_ports.contains(port_name) {
                diagnostics.push(
                    Diagnostic::new(
                        "E_INTERFACE_SIGNAL_PORT_MISSING",
                        format!(
                            "interface signal `{}` is missing port `{}`",
                            interface_signal.name, port_name
                        ),
                    )
                    .with_module(module.name.clone()),
                );
            }
        }
    }
}

fn validate_interface_port(
    module: &Module,
    interface_type: &InterfaceType,
    port: &InterfacePort,
    port_names: &mut HashSet<String>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    if port.name.is_empty() {
        diagnostics.push(
            Diagnostic::new(
                "E_INTERFACE_PORT_NAME",
                format!(
                    "interface type `{}` has an empty port name",
                    interface_type.name
                ),
            )
            .with_module(module.name.clone()),
        );
    }
    if !port_names.insert(port.name.clone()) {
        diagnostics.push(
            Diagnostic::new(
                "E_INTERFACE_PORT_DUP",
                format!(
                    "interface type `{}` has duplicate port `{}`",
                    interface_type.name, port.name
                ),
            )
            .with_module(module.name.clone()),
        );
    }

    match &port.ty {
        InterfacePortType::Scalar(ty) if ty.width == 0 => diagnostics.push(
            Diagnostic::new(
                "E_INTERFACE_PORT_WIDTH",
                format!(
                    "interface type `{}` scalar port `{}` must have non-zero width",
                    interface_type.name, port.name
                ),
            )
            .with_module(module.name.clone()),
        ),
        InterfacePortType::Scalar(_) => {}
        InterfacePortType::Bundle(bundle) => validate_bundle_type_fields(
            module,
            bundle,
            &format!("{}.{}", interface_type.name, port.name),
            diagnostics,
        ),
    }
}

fn validate_interface_signal_port(
    module: &Module,
    interface_signal: &InterfaceSignal,
    expected_port: &InterfacePort,
    port_signal: &InterfaceSignalPort,
    diagnostics: &mut Vec<Diagnostic>,
) {
    match (&expected_port.ty, &port_signal.signals) {
        (InterfacePortType::Scalar(expected_ty), InterfaceSignalPortSignals::Scalar { signal }) => {
            let Some(signal) = module.signal(*signal) else {
                diagnostics.push(
                    Diagnostic::new(
                        "E_INTERFACE_SIGNAL_PORT",
                        format!(
                            "interface signal `{}` scalar port `{}` references an invalid signal",
                            interface_signal.name, port_signal.name
                        ),
                    )
                    .with_module(module.name.clone()),
                );
                return;
            };
            if signal.ty != *expected_ty {
                diagnostics.push(
                    Diagnostic::new(
                        "E_INTERFACE_SIGNAL_PORT_TYPE",
                        format!(
                            "interface signal `{}` scalar port `{}` has type {:?}, expected {:?}",
                            interface_signal.name, port_signal.name, signal.ty, expected_ty
                        ),
                    )
                    .with_module(module.name.clone())
                    .with_signal(signal.name.clone()),
                );
            }
        }
        (InterfacePortType::Bundle(bundle), InterfaceSignalPortSignals::Bundle { fields }) => {
            validate_interface_signal_bundle_port(
                module,
                interface_signal,
                &port_signal.name,
                bundle,
                fields,
                diagnostics,
            );
        }
        (InterfacePortType::Scalar(_), InterfaceSignalPortSignals::Bundle { .. }) => {
            diagnostics.push(
                Diagnostic::new(
                    "E_INTERFACE_SIGNAL_PORT_KIND",
                    format!(
                        "interface signal `{}` port `{}` is a bundle, expected scalar",
                        interface_signal.name, port_signal.name
                    ),
                )
                .with_module(module.name.clone()),
            );
        }
        (InterfacePortType::Bundle(_), InterfaceSignalPortSignals::Scalar { .. }) => {
            diagnostics.push(
                Diagnostic::new(
                    "E_INTERFACE_SIGNAL_PORT_KIND",
                    format!(
                        "interface signal `{}` port `{}` is scalar, expected bundle",
                        interface_signal.name, port_signal.name
                    ),
                )
                .with_module(module.name.clone()),
            );
        }
    }
}

fn validate_interface_signal_bundle_port(
    module: &Module,
    interface_signal: &InterfaceSignal,
    port_name: &str,
    bundle: &BundleType,
    fields: &[BundleSignalField],
    diagnostics: &mut Vec<Diagnostic>,
) {
    let expected = bundle
        .leaf_fields()
        .into_iter()
        .map(|leaf| (leaf.path, leaf.ty))
        .collect::<HashMap<Vec<String>, BitType>>();
    let mut seen_paths = HashSet::new();
    for field in fields {
        if !seen_paths.insert(field.path.clone()) {
            diagnostics.push(
                Diagnostic::new(
                    "E_INTERFACE_SIGNAL_FIELD_DUP",
                    format!(
                        "interface signal `{}` port `{}` has duplicate field `{}`",
                        interface_signal.name,
                        port_name,
                        field.path.join(".")
                    ),
                )
                .with_module(module.name.clone()),
            );
        }

        let Some(expected_ty) = expected.get(&field.path) else {
            diagnostics.push(
                Diagnostic::new(
                    "E_INTERFACE_SIGNAL_FIELD",
                    format!(
                        "interface signal `{}` port `{}` has unexpected field `{}`",
                        interface_signal.name,
                        port_name,
                        field.path.join(".")
                    ),
                )
                .with_module(module.name.clone()),
            );
            continue;
        };

        let Some(signal) = module.signal(field.signal) else {
            diagnostics.push(
                Diagnostic::new(
                    "E_INTERFACE_SIGNAL_FIELD",
                    format!(
                        "interface signal `{}` port `{}` field `{}` references an invalid signal",
                        interface_signal.name,
                        port_name,
                        field.path.join(".")
                    ),
                )
                .with_module(module.name.clone()),
            );
            continue;
        };

        if signal.ty != *expected_ty {
            diagnostics.push(
                Diagnostic::new(
                    "E_INTERFACE_SIGNAL_FIELD_TYPE",
                    format!(
                        "interface signal `{}` port `{}` field `{}` has type {:?}, expected {:?}",
                        interface_signal.name,
                        port_name,
                        field.path.join("."),
                        signal.ty,
                        expected_ty
                    ),
                )
                .with_module(module.name.clone())
                .with_signal(signal.name.clone()),
            );
        }
    }

    for path in expected.keys() {
        if !seen_paths.contains(path) {
            diagnostics.push(
                Diagnostic::new(
                    "E_INTERFACE_SIGNAL_FIELD_MISSING",
                    format!(
                        "interface signal `{}` port `{}` is missing field `{}`",
                        interface_signal.name,
                        port_name,
                        path.join(".")
                    ),
                )
                .with_module(module.name.clone()),
            );
        }
    }
}

fn validate_signal_names(module: &Module, diagnostics: &mut Vec<Diagnostic>) {
    let mut seen = HashSet::new();
    for signal in &module.signals {
        if signal.name.is_empty() {
            diagnostics.push(
                Diagnostic::new("E_SIGNAL_NAME_EMPTY", "signal name is empty")
                    .with_module(module.name.clone()),
            );
        }
        if !seen.insert(signal.name.clone()) {
            diagnostics.push(
                Diagnostic::new(
                    "E_SIGNAL_NAME_DUP",
                    format!("duplicate signal name `{}`", signal.name),
                )
                .with_module(module.name.clone())
                .with_signal(signal.name.clone()),
            );
        }
    }
}

fn validate_signal_widths(module: &Module, diagnostics: &mut Vec<Diagnostic>) {
    for signal in &module.signals {
        if signal.width == 0 {
            diagnostics.push(
                Diagnostic::new("E_WIDTH_ZERO", "signal width must be greater than zero")
                    .with_module(module.name.clone())
                    .with_signal(signal.name.clone()),
            );
        }
        if let SignalKind::Mem {
            addr_width,
            data_width,
            depth,
        } = signal.kind
        {
            if addr_width == 0 || data_width == 0 || depth == 0 {
                diagnostics.push(
                    Diagnostic::new(
                        "E_MEM_SHAPE",
                        "memory address width, data width, and depth must be non-zero",
                    )
                    .with_module(module.name.clone())
                    .with_signal(signal.name.clone()),
                );
            }
            if addr_width < usize::BITS && depth > (1usize << addr_width) {
                diagnostics.push(
                    Diagnostic::new(
                        "E_MEM_DEPTH_WIDTH",
                        format!(
                            "memory depth {depth} exceeds {}-bit address space",
                            addr_width
                        ),
                    )
                    .with_module(module.name.clone())
                    .with_signal(signal.name.clone()),
                );
            }
        }
    }
}

fn validate_assignments(module: &Module, diagnostics: &mut Vec<Diagnostic>) {
    let mut drivers: HashMap<Signal, usize> = HashMap::new();
    for assignment in &module.assignments {
        let Some(dst) = module.signal(assignment.dst) else {
            diagnostics.push(
                Diagnostic::new(
                    "E_ASSIGN_DST",
                    "assignment destination is not in this module",
                )
                .with_module(module.name.clone()),
            );
            continue;
        };

        match dst.kind {
            SignalKind::Input => diagnostics.push(
                Diagnostic::new("E_ASSIGN_INPUT", "cannot assign to an input")
                    .with_module(module.name.clone())
                    .with_signal(dst.name.clone()),
            ),
            SignalKind::Inout => diagnostics.push(
                Diagnostic::new("E_ASSIGN_INOUT", "cannot assign to an inout")
                    .with_module(module.name.clone())
                    .with_signal(dst.name.clone()),
            ),
            SignalKind::Reg { .. } => diagnostics.push(
                Diagnostic::new("E_ASSIGN_REG", "registers must be driven with `next`")
                    .with_module(module.name.clone())
                    .with_signal(dst.name.clone()),
            ),
            SignalKind::Mem { .. } => diagnostics.push(
                Diagnostic::new(
                    "E_ASSIGN_MEM",
                    "memories must be driven with `mem_write` ports",
                )
                .with_module(module.name.clone())
                .with_signal(dst.name.clone()),
            ),
            SignalKind::Output | SignalKind::Wire => {}
        }

        *drivers.entry(assignment.dst).or_insert(0) += 1;
        match expr_type(module, &assignment.expr, diagnostics) {
            Some(ty) if ty != dst.ty => diagnostics.push(
                Diagnostic::new(
                    "E_ASSIGN_TYPE",
                    format!(
                        "assignment type mismatch: destination is {:?}, expression is {:?}",
                        dst.ty, ty
                    ),
                )
                .with_module(module.name.clone())
                .with_signal(dst.name.clone()),
            ),
            _ => {}
        }
    }

    for (signal, count) in drivers {
        if count > 1 {
            let name = module
                .signal(signal)
                .map(|s| s.name.clone())
                .unwrap_or_else(|| format!("{:?}", signal.id));
            diagnostics.push(
                Diagnostic::new(
                    "E_MULTI_DRIVER",
                    format!("signal `{name}` has {count} continuous drivers"),
                )
                .with_module(module.name.clone())
                .with_signal(name),
            );
        }
    }
}

fn validate_memory_writes(module: &Module, diagnostics: &mut Vec<Diagnostic>) {
    for write in &module.memory_writes {
        let Some(mem) = module.signal(write.mem) else {
            diagnostics.push(
                Diagnostic::new(
                    "E_MEM_WRITE_TARGET",
                    "memory write target is not in this module",
                )
                .with_module(module.name.clone()),
            );
            continue;
        };

        let SignalKind::Mem {
            addr_width,
            data_width: _,
            ..
        } = mem.kind
        else {
            diagnostics.push(
                Diagnostic::new("E_MEM_WRITE_TARGET", "memory write target is not a memory")
                    .with_module(module.name.clone())
                    .with_signal(mem.name.clone()),
            );
            continue;
        };

        match signal_width(module, write.clock, diagnostics) {
            Some(1) => {}
            Some(width) => diagnostics.push(
                Diagnostic::new(
                    "E_MEM_CLOCK_WIDTH",
                    format!("memory write clock must be 1 bit, found {width} bits"),
                )
                .with_module(module.name.clone())
                .with_signal(mem.name.clone()),
            ),
            None => {}
        }

        match expr_width(module, &write.enable, diagnostics) {
            Some(1) => {}
            Some(width) => diagnostics.push(
                Diagnostic::new(
                    "E_MEM_ENABLE_WIDTH",
                    format!("memory write enable must be 1 bit, found {width} bits"),
                )
                .with_module(module.name.clone())
                .with_signal(mem.name.clone()),
            ),
            None => {}
        }

        match expr_type(module, &write.addr, diagnostics) {
            Some(ty) if ty == uint(addr_width) => {}
            Some(ty) => diagnostics.push(
                Diagnostic::new(
                    "E_MEM_ADDR_WIDTH",
                    format!(
                        "memory write address type mismatch: memory address is {:?}, expression is {:?}",
                        uint(addr_width), ty
                    ),
                )
                .with_module(module.name.clone())
                .with_signal(mem.name.clone()),
            ),
            None => {}
        }

        match expr_type(module, &write.data, diagnostics) {
            Some(ty) if ty == mem.ty => {}
            Some(ty) => diagnostics.push(
                Diagnostic::new(
                    "E_MEM_DATA_WIDTH",
                    format!(
                        "memory write data type mismatch: memory data is {:?}, expression is {:?}",
                        mem.ty, ty
                    ),
                )
                .with_module(module.name.clone())
                .with_signal(mem.name.clone()),
            ),
            None => {}
        }
    }
}

fn validate_initial_values(module: &Module, diagnostics: &mut Vec<Diagnostic>) {
    let mut initialized_registers = HashSet::new();
    for initial in &module.initial_register_values {
        let Some(signal) = module.signal(initial.signal) else {
            diagnostics.push(
                Diagnostic::new(
                    "E_INITIAL_REG_TARGET",
                    "initial register target is not in this module",
                )
                .with_module(module.name.clone()),
            );
            continue;
        };
        if !matches!(signal.kind, SignalKind::Reg { .. }) {
            diagnostics.push(
                Diagnostic::new("E_INITIAL_REG_TARGET", "initial target is not a register")
                    .with_module(module.name.clone())
                    .with_signal(signal.name.clone()),
            );
        }
        if !initialized_registers.insert(initial.signal) {
            diagnostics.push(
                Diagnostic::new(
                    "E_INITIAL_REG_DUP",
                    format!("register `{}` has multiple initial values", signal.name),
                )
                .with_module(module.name.clone())
                .with_signal(signal.name.clone()),
            );
        }
        if !fits_type(initial.value, signal.ty) {
            diagnostics.push(
                Diagnostic::new(
                    "E_INITIAL_REG_VALUE",
                    format!("initial value does not fit in {:?}", signal.ty),
                )
                .with_module(module.name.clone())
                .with_signal(signal.name.clone()),
            );
        }
    }

    let mut initialized_words = HashSet::new();
    for initial in &module.initial_memory_values {
        let Some(mem) = module.signal(initial.mem) else {
            diagnostics.push(
                Diagnostic::new(
                    "E_INITIAL_MEM_TARGET",
                    "initial memory target is not in this module",
                )
                .with_module(module.name.clone()),
            );
            continue;
        };
        let SignalKind::Mem { depth, .. } = mem.kind else {
            diagnostics.push(
                Diagnostic::new("E_INITIAL_MEM_TARGET", "initial target is not a memory")
                    .with_module(module.name.clone())
                    .with_signal(mem.name.clone()),
            );
            continue;
        };
        if initial.addr >= depth {
            diagnostics.push(
                Diagnostic::new(
                    "E_INITIAL_MEM_ADDR",
                    format!(
                        "initial memory address {} exceeds depth {depth}",
                        initial.addr
                    ),
                )
                .with_module(module.name.clone())
                .with_signal(mem.name.clone()),
            );
        }
        if !initialized_words.insert((initial.mem, initial.addr)) {
            diagnostics.push(
                Diagnostic::new(
                    "E_INITIAL_MEM_DUP",
                    format!(
                        "memory `{}` address {} has multiple initial values",
                        mem.name, initial.addr
                    ),
                )
                .with_module(module.name.clone())
                .with_signal(mem.name.clone()),
            );
        }
        if !fits_type(initial.value, mem.ty) {
            diagnostics.push(
                Diagnostic::new(
                    "E_INITIAL_MEM_VALUE",
                    format!("initial memory value does not fit in {:?}", mem.ty),
                )
                .with_module(module.name.clone())
                .with_signal(mem.name.clone()),
            );
        }
    }
}

fn validate_assertions(module: &Module, diagnostics: &mut Vec<Diagnostic>) {
    let mut names = HashSet::new();
    for assertion in &module.assertions {
        if assertion.name.is_empty() {
            diagnostics.push(
                Diagnostic::new("E_ASSERT_NAME_EMPTY", "assertion name is empty")
                    .with_module(module.name.clone()),
            );
        }
        if !names.insert(assertion.name.clone()) {
            diagnostics.push(
                Diagnostic::new(
                    "E_ASSERT_NAME_DUP",
                    format!("duplicate assertion name `{}`", assertion.name),
                )
                .with_module(module.name.clone()),
            );
        }

        match expr_type(module, &assertion.condition, diagnostics) {
            Some(ty) if ty == uint(1) => {}
            Some(ty) => diagnostics.push(
                Diagnostic::new(
                    "E_ASSERT_CONDITION_TYPE",
                    format!("assertion condition must be uint(1), found {:?}", ty),
                )
                .with_module(module.name.clone()),
            ),
            None => {}
        }

        if let Some(enable) = &assertion.enable {
            match expr_type(module, enable, diagnostics) {
                Some(ty) if ty == uint(1) => {}
                Some(ty) => diagnostics.push(
                    Diagnostic::new(
                        "E_ASSERT_ENABLE_TYPE",
                        format!("assertion enable must be uint(1), found {:?}", ty),
                    )
                    .with_module(module.name.clone()),
                ),
                None => {}
            }
        }

        if let Some(clock) = assertion.clock {
            match signal_width(module, clock, diagnostics) {
                Some(1) => {}
                Some(width) => diagnostics.push(
                    Diagnostic::new(
                        "E_ASSERT_CLOCK_WIDTH",
                        format!("assertion clock must be 1 bit, found {width} bits"),
                    )
                    .with_module(module.name.clone()),
                ),
                None => {}
            }
        }
    }
}

fn validate_cover_points(module: &Module, diagnostics: &mut Vec<Diagnostic>) {
    let mut names = HashSet::new();
    for cover in &module.cover_points {
        if cover.name.is_empty() {
            diagnostics.push(
                Diagnostic::new("E_COVER_NAME_EMPTY", "cover point name is empty")
                    .with_module(module.name.clone()),
            );
        }
        if !names.insert(cover.name.clone()) {
            diagnostics.push(
                Diagnostic::new(
                    "E_COVER_NAME_DUP",
                    format!("duplicate cover point name `{}`", cover.name),
                )
                .with_module(module.name.clone()),
            );
        }

        match expr_type(module, &cover.condition, diagnostics) {
            Some(ty) if ty == uint(1) => {}
            Some(ty) => diagnostics.push(
                Diagnostic::new(
                    "E_COVER_CONDITION_TYPE",
                    format!("cover condition must be uint(1), found {:?}", ty),
                )
                .with_module(module.name.clone()),
            ),
            None => {}
        }

        if let Some(enable) = &cover.enable {
            match expr_type(module, enable, diagnostics) {
                Some(ty) if ty == uint(1) => {}
                Some(ty) => diagnostics.push(
                    Diagnostic::new(
                        "E_COVER_ENABLE_TYPE",
                        format!("cover enable must be uint(1), found {:?}", ty),
                    )
                    .with_module(module.name.clone()),
                ),
                None => {}
            }
        }

        if let Some(clock) = cover.clock {
            match signal_width(module, clock, diagnostics) {
                Some(1) => {}
                Some(width) => diagnostics.push(
                    Diagnostic::new(
                        "E_COVER_CLOCK_WIDTH",
                        format!("cover clock must be 1 bit, found {width} bits"),
                    )
                    .with_module(module.name.clone()),
                ),
                None => {}
            }
        }
    }
}

fn validate_registers(module: &Module, diagnostics: &mut Vec<Diagnostic>) {
    for signal in &module.signals {
        let SignalKind::Reg { clock, reset, next } = &signal.kind else {
            continue;
        };

        match clock {
            Some(clock) => match signal_width(module, *clock, diagnostics) {
                Some(1) => {}
                Some(width) => diagnostics.push(
                    Diagnostic::new(
                        "E_CLOCK_WIDTH",
                        format!("register clock must be 1 bit, found {width} bits"),
                    )
                    .with_module(module.name.clone())
                    .with_signal(signal.name.clone()),
                ),
                None => {}
            },
            None => diagnostics.push(
                Diagnostic::new("E_REG_CLOCK_MISSING", "register has no clock")
                    .with_module(module.name.clone())
                    .with_signal(signal.name.clone()),
            ),
        }

        if let Some(reset) = reset {
            match signal_width(module, reset.signal, diagnostics) {
                Some(1) => {}
                Some(width) => diagnostics.push(
                    Diagnostic::new(
                        "E_RESET_WIDTH",
                        format!("register reset must be 1 bit, found {width} bits"),
                    )
                    .with_module(module.name.clone())
                    .with_signal(signal.name.clone()),
                ),
                None => {}
            }
            if !fits_type(reset.value, signal.ty) {
                diagnostics.push(
                    Diagnostic::new(
                        "E_RESET_VALUE_WIDTH",
                        format!("reset value does not fit in {:?}", signal.ty),
                    )
                    .with_module(module.name.clone())
                    .with_signal(signal.name.clone()),
                );
            }
        }

        match next {
            Some(expr) => match expr_type(module, expr, diagnostics) {
                Some(ty) if ty != signal.ty => diagnostics.push(
                    Diagnostic::new(
                        "E_REG_NEXT_TYPE",
                        format!(
                            "register next type mismatch: register is {:?}, expression is {:?}",
                            signal.ty, ty
                        ),
                    )
                    .with_module(module.name.clone())
                    .with_signal(signal.name.clone()),
                ),
                _ => {}
            },
            None => diagnostics.push(
                Diagnostic::new("E_REG_NEXT_MISSING", "register has no next expression")
                    .with_module(module.name.clone())
                    .with_signal(signal.name.clone()),
            ),
        }
    }
}

fn validate_instances(
    design: &rrtl_ir::Design,
    module: &Module,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let mut instance_names = HashSet::new();
    let mut output_drivers: HashMap<Signal, usize> = HashMap::new();
    for instance in &module.instances {
        if !instance_names.insert(instance.name.clone()) {
            diagnostics.push(
                Diagnostic::new(
                    "E_INSTANCE_NAME_DUP",
                    format!("duplicate instance name `{}`", instance.name),
                )
                .with_module(module.name.clone()),
            );
        }

        let Some(target) = design.find_module(&instance.module) else {
            diagnostics.push(
                Diagnostic::new(
                    "E_INSTANCE_TARGET",
                    format!(
                        "instance target module `{}` does not exist",
                        instance.module
                    ),
                )
                .with_module(module.name.clone()),
            );
            continue;
        };

        let mut connected_ports = HashSet::new();
        for (port_name, signal) in &instance.connections {
            let Some(port) = target.signals.iter().find(|s| {
                s.name == *port_name
                    && matches!(
                        s.kind,
                        SignalKind::Input | SignalKind::Output | SignalKind::Inout
                    )
            }) else {
                diagnostics.push(
                    Diagnostic::new(
                        "E_INSTANCE_PORT",
                        format!(
                            "target module `{}` has no port named `{}`",
                            target.name, port_name
                        ),
                    )
                    .with_module(module.name.clone()),
                );
                continue;
            };
            if !connected_ports.insert(port_name.clone()) {
                diagnostics.push(
                    Diagnostic::new(
                        "E_INSTANCE_PORT_DUP",
                        format!(
                            "instance `{}` connects port `{}` more than once",
                            instance.name, port_name
                        ),
                    )
                    .with_module(module.name.clone()),
                );
            }

            match signal_type(module, *signal, diagnostics) {
                Some(ty) if ty != port.ty => diagnostics.push(
                    Diagnostic::new(
                        "E_INSTANCE_TYPE",
                        format!(
                            "connection `{}` type mismatch: port is {:?}, signal is {:?}",
                            port_name, port.ty, ty
                        ),
                    )
                    .with_module(module.name.clone()),
                ),
                _ => {}
            }

            if matches!(port.kind, SignalKind::Inout)
                && !matches!(
                    module.signal(*signal).map(|s| &s.kind),
                    Some(SignalKind::Inout | SignalKind::Wire)
                )
            {
                let parent_name = module
                    .signal(*signal)
                    .map(|s| s.name.clone())
                    .unwrap_or_else(|| format!("{:?}", signal.id));
                diagnostics.push(
                    Diagnostic::new(
                        "E_INSTANCE_INOUT_TARGET",
                        format!(
                            "instance inout `{}` cannot connect to parent signal `{parent_name}`",
                            port_name
                        ),
                    )
                    .with_module(module.name.clone())
                    .with_signal(parent_name),
                );
            }

            if matches!(port.kind, SignalKind::Output) {
                let parent_name = module
                    .signal(*signal)
                    .map(|s| s.name.clone())
                    .unwrap_or_else(|| format!("{:?}", signal.id));
                if matches!(
                    module.signal(*signal).map(|s| &s.kind),
                    Some(
                        SignalKind::Input
                            | SignalKind::Inout
                            | SignalKind::Reg { .. }
                            | SignalKind::Mem { .. }
                    )
                ) {
                    diagnostics.push(
                        Diagnostic::new(
                            "E_INSTANCE_OUTPUT_TARGET",
                            format!(
                                "instance output `{}` cannot drive parent signal `{parent_name}`",
                                port_name
                            ),
                        )
                        .with_module(module.name.clone())
                        .with_signal(parent_name),
                    );
                }
                *output_drivers.entry(*signal).or_insert(0) += 1;
            }
        }

        for port in target.signals.iter().filter(|s| {
            matches!(
                s.kind,
                SignalKind::Input | SignalKind::Output | SignalKind::Inout
            )
        }) {
            if !connected_ports.contains(&port.name) {
                diagnostics.push(
                    Diagnostic::new(
                        "E_INSTANCE_UNCONNECTED",
                        format!(
                            "instance `{}` leaves port `{}` unconnected",
                            instance.name, port.name
                        ),
                    )
                    .with_module(module.name.clone()),
                );
            }
        }
    }

    for assignment in &module.assignments {
        if output_drivers.contains_key(&assignment.dst) {
            let name = module
                .signal(assignment.dst)
                .map(|s| s.name.clone())
                .unwrap_or_else(|| format!("{:?}", assignment.dst.id));
            diagnostics.push(
                Diagnostic::new(
                    "E_MULTI_DRIVER",
                    format!("signal `{name}` is driven by both assignment and instance output"),
                )
                .with_module(module.name.clone())
                .with_signal(name),
            );
        }
    }

    for (signal, count) in output_drivers {
        if count > 1 {
            let name = module
                .signal(signal)
                .map(|s| s.name.clone())
                .unwrap_or_else(|| format!("{:?}", signal.id));
            diagnostics.push(
                Diagnostic::new(
                    "E_MULTI_DRIVER",
                    format!("signal `{name}` has {count} instance output drivers"),
                )
                .with_module(module.name.clone())
                .with_signal(name),
            );
        }
    }
}

fn validate_comb_cycles(
    design: &rrtl_ir::Design,
    module: &Module,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let mut graph: HashMap<Signal, Vec<Signal>> = HashMap::new();
    for assignment in &module.assignments {
        graph
            .entry(assignment.dst)
            .or_default()
            .extend(assignment_signal_deps(assignment.dst, &assignment.expr));
    }
    for instance in &module.instances {
        add_instance_comb_deps(design, module, instance, &mut graph);
    }

    let mut visiting = HashSet::new();
    let mut visited = HashSet::new();
    for signal in graph.keys().copied().collect::<Vec<_>>() {
        if has_cycle(signal, &graph, &mut visiting, &mut visited) {
            let name = module
                .signal(signal)
                .map(|s| s.name.clone())
                .unwrap_or_else(|| format!("{:?}", signal.id));
            diagnostics.push(
                Diagnostic::new(
                    "E_COMB_CYCLE",
                    format!("combinational cycle reaches `{name}`"),
                )
                .with_module(module.name.clone())
                .with_signal(name),
            );
            break;
        }
    }
}

fn add_instance_comb_deps(
    design: &rrtl_ir::Design,
    module: &Module,
    instance: &Instance,
    graph: &mut HashMap<Signal, Vec<Signal>>,
) {
    let Some(target) = design.find_module(&instance.module) else {
        return;
    };
    let mut parent_inputs = Vec::new();
    let mut parent_outputs = Vec::new();
    for (port_name, parent_signal) in &instance.connections {
        let Some(port) = target
            .signals
            .iter()
            .find(|signal| signal.name == *port_name)
        else {
            continue;
        };
        match port.kind {
            SignalKind::Input if module.signal(*parent_signal).is_some() => {
                parent_inputs.push(*parent_signal)
            }
            SignalKind::Output if module.signal(*parent_signal).is_some() => {
                parent_outputs.push(*parent_signal)
            }
            _ => {}
        }
    }

    for output in parent_outputs {
        graph
            .entry(output)
            .or_default()
            .extend(parent_inputs.iter().copied());
    }
}

fn has_cycle(
    signal: Signal,
    graph: &HashMap<Signal, Vec<Signal>>,
    visiting: &mut HashSet<Signal>,
    visited: &mut HashSet<Signal>,
) -> bool {
    if visited.contains(&signal) {
        return false;
    }
    if !visiting.insert(signal) {
        return true;
    }
    for dep in graph.get(&signal).into_iter().flatten().copied() {
        if graph.contains_key(&dep) && has_cycle(dep, graph, visiting, visited) {
            return true;
        }
    }
    visiting.remove(&signal);
    visited.insert(signal);
    false
}

fn expr_width(module: &Module, expr: &Expr, diagnostics: &mut Vec<Diagnostic>) -> Option<Width> {
    expr_type(module, expr, diagnostics).map(|ty| ty.width)
}

fn expr_type(module: &Module, expr: &Expr, diagnostics: &mut Vec<Diagnostic>) -> Option<BitType> {
    match expr {
        Expr::Lit { value, ty } => {
            if ty.width == 0 {
                diagnostics.push(
                    Diagnostic::new(
                        "E_LIT_WIDTH_ZERO",
                        "literal width must be greater than zero",
                    )
                    .with_module(module.name.clone()),
                );
                return None;
            }
            if !fits_type(*value, *ty) {
                diagnostics.push(
                    Diagnostic::new(
                        "E_LIT_VALUE_WIDTH",
                        format!("literal value {value} does not fit in {:?}", ty),
                    )
                    .with_module(module.name.clone()),
                );
            }
            Some(*ty)
        }
        Expr::Signal(signal) => signal_type(module, *signal, diagnostics),
        Expr::Not(inner) => expr_type(module, inner, diagnostics),
        Expr::And(lhs, rhs) | Expr::Or(lhs, rhs) | Expr::Xor(lhs, rhs) => {
            same_type(module, lhs, rhs, diagnostics)
        }
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) | Expr::Mul(lhs, rhs) => {
            same_numeric_type(module, lhs, rhs, diagnostics)
        }
        Expr::Eq(lhs, rhs) | Expr::Ne(lhs, rhs) | Expr::Lt(lhs, rhs) => {
            same_numeric_type(module, lhs, rhs, diagnostics);
            Some(uint(1))
        }
        Expr::Mux {
            cond,
            then_expr,
            else_expr,
        } => {
            match expr_type(module, cond, diagnostics) {
                Some(ty) if ty == uint(1) => {}
                Some(ty) => diagnostics.push(
                    Diagnostic::new(
                        "E_MUX_COND_WIDTH",
                        format!("mux condition must be unsigned 1 bit, found {:?}", ty),
                    )
                    .with_module(module.name.clone()),
                ),
                None => {}
            }
            same_type(module, then_expr, else_expr, diagnostics)
        }
        Expr::Slice { expr, lsb, width } => {
            if *width == 0 {
                diagnostics.push(
                    Diagnostic::new(
                        "E_SLICE_WIDTH_ZERO",
                        "slice width must be greater than zero",
                    )
                    .with_module(module.name.clone()),
                );
                return None;
            }
            if let Some(inner_ty) = expr_type(module, expr, diagnostics) {
                if *lsb + *width > inner_ty.width {
                    diagnostics.push(
                        Diagnostic::new(
                            "E_SLICE_RANGE",
                            format!(
                                "slice [{}:{}] exceeds {}-bit expression",
                                *lsb + *width - 1,
                                lsb,
                                inner_ty.width
                            ),
                        )
                        .with_module(module.name.clone()),
                    );
                }
            }
            Some(uint(*width))
        }
        Expr::Zext { expr, width } => {
            validate_extend_width(
                module,
                expr,
                *width,
                "E_ZEXT_WIDTH_ZERO",
                "E_ZEXT_WIDTH",
                diagnostics,
            )?;
            Some(uint(*width))
        }
        Expr::Sext { expr, width } => {
            validate_extend_width(
                module,
                expr,
                *width,
                "E_SEXT_WIDTH_ZERO",
                "E_SEXT_WIDTH",
                diagnostics,
            )?;
            Some(sint(*width))
        }
        Expr::Trunc { expr, width } => {
            if *width == 0 {
                diagnostics.push(
                    Diagnostic::new(
                        "E_TRUNC_WIDTH_ZERO",
                        "trunc width must be greater than zero",
                    )
                    .with_module(module.name.clone()),
                );
                return None;
            }
            let inner_ty = expr_type(module, expr, diagnostics)?;
            if *width > inner_ty.width {
                diagnostics.push(
                    Diagnostic::new(
                        "E_TRUNC_WIDTH",
                        format!(
                            "trunc target width must be no greater than source width: target is {width} bits, source is {} bits",
                            inner_ty.width
                        ),
                    )
                    .with_module(module.name.clone()),
                );
            }
            Some(BitType::new(*width, inner_ty.signedness))
        }
        Expr::Cast { expr, signedness } => {
            let inner_ty = expr_type(module, expr, diagnostics)?;
            Some(BitType::new(inner_ty.width, *signedness))
        }
        Expr::Concat(parts) => {
            if parts.is_empty() {
                diagnostics.push(
                    Diagnostic::new("E_CONCAT_EMPTY", "concat needs at least one part")
                        .with_module(module.name.clone()),
                );
                return None;
            }
            let mut total = 0;
            for part in parts {
                if let Some(ty) = expr_type(module, part, diagnostics) {
                    total += ty.width;
                }
            }
            Some(uint(total))
        }
        Expr::MemRead { mem, addr } => {
            let Some(info) = module.signal(*mem) else {
                diagnostics.push(
                    Diagnostic::new(
                        "E_MEM_READ_TARGET",
                        "memory read target is not in this module",
                    )
                    .with_module(module.name.clone()),
                );
                return None;
            };
            let SignalKind::Mem { addr_width, .. } = info.kind else {
                diagnostics.push(
                    Diagnostic::new("E_MEM_READ_TARGET", "memory read target is not a memory")
                        .with_module(module.name.clone())
                        .with_signal(info.name.clone()),
                );
                return None;
            };
            match expr_type(module, addr, diagnostics) {
                Some(ty) if ty == uint(addr_width) => {}
                Some(ty) => diagnostics.push(
                    Diagnostic::new(
                        "E_MEM_ADDR_WIDTH",
                        format!(
                            "memory read address type mismatch: memory address is {:?}, expression is {:?}",
                            uint(addr_width), ty
                        ),
                    )
                    .with_module(module.name.clone())
                    .with_signal(info.name.clone()),
                ),
                None => {}
            }
            Some(info.ty)
        }
    }
}

fn validate_extend_width(
    module: &Module,
    expr: &Expr,
    width: Width,
    zero_code: &'static str,
    width_code: &'static str,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<()> {
    if width == 0 {
        diagnostics.push(
            Diagnostic::new(zero_code, "extension width must be greater than zero")
                .with_module(module.name.clone()),
        );
        return None;
    }
    if let Some(inner_ty) = expr_type(module, expr, diagnostics) {
        if width < inner_ty.width {
            diagnostics.push(
                Diagnostic::new(
                    width_code,
                    format!(
                        "extension target width must be at least source width: target is {width} bits, source is {} bits",
                        inner_ty.width
                    ),
                )
                .with_module(module.name.clone()),
            );
        }
    }
    Some(())
}

fn same_type(
    module: &Module,
    lhs: &Expr,
    rhs: &Expr,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<BitType> {
    let lhs_ty = expr_type(module, lhs, diagnostics);
    let rhs_ty = expr_type(module, rhs, diagnostics);
    match (lhs_ty, rhs_ty) {
        (Some(lhs_ty), Some(rhs_ty)) if lhs_ty == rhs_ty => Some(lhs_ty),
        (Some(lhs_ty), Some(rhs_ty)) => {
            diagnostics.push(
                Diagnostic::new(
                    "E_EXPR_TYPE",
                    format!("expression type mismatch: {:?} vs {:?}", lhs_ty, rhs_ty),
                )
                .with_module(module.name.clone()),
            );
            None
        }
        _ => None,
    }
}

fn same_numeric_type(
    module: &Module,
    lhs: &Expr,
    rhs: &Expr,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<BitType> {
    let lhs_ty = expr_type(module, lhs, diagnostics);
    let rhs_ty = expr_type(module, rhs, diagnostics);
    match (lhs_ty, rhs_ty) {
        (Some(lhs_ty), Some(rhs_ty)) if lhs_ty == rhs_ty => Some(lhs_ty),
        (Some(lhs_ty), Some(rhs_ty)) if lhs_ty.width == rhs_ty.width => {
            diagnostics.push(
                Diagnostic::new(
                    "E_SIGNED_MIX",
                    format!(
                        "mixed signedness requires an explicit cast: {:?} vs {:?}",
                        lhs_ty, rhs_ty
                    ),
                )
                .with_module(module.name.clone()),
            );
            None
        }
        (Some(lhs_ty), Some(rhs_ty)) => {
            diagnostics.push(
                Diagnostic::new(
                    "E_EXPR_TYPE",
                    format!("expression type mismatch: {:?} vs {:?}", lhs_ty, rhs_ty),
                )
                .with_module(module.name.clone()),
            );
            None
        }
        _ => None,
    }
}

fn signal_width(
    module: &Module,
    signal: Signal,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<Width> {
    let Some(info) = module.signal(signal) else {
        diagnostics.push(
            Diagnostic::new("E_SIGNAL_SCOPE", "signal does not belong to this module")
                .with_module(module.name.clone()),
        );
        return None;
    };
    Some(info.width)
}

fn signal_type(
    module: &Module,
    signal: Signal,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<BitType> {
    let Some(info) = module.signal(signal) else {
        diagnostics.push(
            Diagnostic::new("E_SIGNAL_SCOPE", "signal does not belong to this module")
                .with_module(module.name.clone()),
        );
        return None;
    };
    Some(info.ty)
}

fn expr_signal_deps(expr: &Expr) -> Vec<Signal> {
    let mut deps = Vec::new();
    collect_expr_signal_deps(expr, &mut deps);
    deps
}

fn assignment_signal_deps(dst: Signal, expr: &Expr) -> Vec<Signal> {
    let mut deps = Vec::new();
    match expr {
        Expr::Mux {
            cond,
            then_expr,
            else_expr,
        } if matches!(else_expr.as_ref(), Expr::Signal(signal) if *signal == dst) => {
            collect_expr_signal_deps(cond, &mut deps);
            collect_expr_signal_deps(then_expr, &mut deps);
        }
        _ => deps.extend(expr_signal_deps(expr)),
    }
    deps
}

fn collect_expr_signal_deps(expr: &Expr, deps: &mut Vec<Signal>) {
    match expr {
        Expr::Signal(signal) => deps.push(*signal),
        Expr::Lit { .. } => {}
        Expr::Not(inner)
        | Expr::Slice { expr: inner, .. }
        | Expr::Zext { expr: inner, .. }
        | Expr::Sext { expr: inner, .. }
        | Expr::Trunc { expr: inner, .. }
        | Expr::Cast { expr: inner, .. } => collect_expr_signal_deps(inner, deps),
        Expr::And(lhs, rhs)
        | Expr::Or(lhs, rhs)
        | Expr::Xor(lhs, rhs)
        | Expr::Add(lhs, rhs)
        | Expr::Sub(lhs, rhs)
        | Expr::Mul(lhs, rhs)
        | Expr::Eq(lhs, rhs)
        | Expr::Ne(lhs, rhs)
        | Expr::Lt(lhs, rhs) => {
            collect_expr_signal_deps(lhs, deps);
            collect_expr_signal_deps(rhs, deps);
        }
        Expr::Mux {
            cond,
            then_expr,
            else_expr,
        } => {
            collect_expr_signal_deps(cond, deps);
            collect_expr_signal_deps(then_expr, deps);
            collect_expr_signal_deps(else_expr, deps);
        }
        Expr::Concat(parts) => {
            for part in parts {
                collect_expr_signal_deps(part, deps);
            }
        }
        Expr::MemRead { addr, .. } => collect_expr_signal_deps(addr, deps),
    }
}

pub fn fits_width(value: u128, width: Width) -> bool {
    width >= 128 || value < (1u128 << width)
}

pub fn fits_type(value: i128, ty: BitType) -> bool {
    if ty.width == 0 {
        return false;
    }
    if ty.width >= 128 {
        return true;
    }
    match ty.signedness {
        Signedness::Unsigned => value >= 0 && (value as u128) < (1u128 << ty.width),
        Signedness::Signed => {
            let min = -(1i128 << (ty.width - 1));
            let max = (1i128 << (ty.width - 1)) - 1;
            value >= min && value <= max
        }
    }
}

fn mask(width: Width) -> u128 {
    if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    }
}

fn encode_value(value: i128, ty: BitType) -> u128 {
    (value as u128) & mask(ty.width)
}

fn signed_value(value: u128, width: Width) -> i128 {
    if width == 0 {
        return 0;
    }
    if width >= 128 {
        return value as i128;
    }
    let sign_bit = 1u128 << (width - 1);
    let value = value & mask(width);
    if value & sign_bit == 0 {
        value as i128
    } else {
        (value as i128) - (1i128 << width)
    }
}

#[derive(Clone, Debug)]
pub struct Simulator<'a> {
    design: &'a Design,
    module: &'a Module,
    path: String,
    values: HashMap<Signal, u128>,
    memories: HashMap<Signal, Vec<u128>>,
    cover_hits: HashMap<String, u64>,
    children: Vec<InstanceState<'a>>,
}

#[derive(Clone, Debug, Default)]
pub struct SimulationInit {
    pub registers: HashMap<Signal, u128>,
    pub memories: HashMap<Signal, Vec<(usize, u128)>>,
}

fn module_has_inout(module: &Module) -> bool {
    module
        .signals
        .iter()
        .any(|signal| matches!(signal.kind, SignalKind::Inout))
}

#[derive(Clone, Debug)]
struct InstanceState<'a> {
    instance: &'a Instance,
    sim: Box<Simulator<'a>>,
}

#[derive(Clone, Debug, Default)]
struct TickState {
    values: Vec<(Signal, u128)>,
    memories: Vec<(Signal, usize, u128)>,
    children: Vec<TickState>,
}

impl<'a> Simulator<'a> {
    pub fn new(design: &'a Design, module_name: &str) -> Result<Self, ErrorReport> {
        Self::new_with_init(design, module_name, &SimulationInit::default())
    }

    pub fn new_with_init(
        design: &'a Design,
        module_name: &str,
        init: &SimulationInit,
    ) -> Result<Self, ErrorReport> {
        design.validate()?;
        let Some(module) = design.ir.find_module(module_name) else {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_MODULE",
                format!("module `{module_name}` does not exist"),
            )]));
        };
        if module.is_external {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_EXTERN_MODULE",
                format!("external module `{module_name}` cannot be simulated"),
            )]));
        }
        if module_has_inout(module) {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_INOUT",
                format!("module `{module_name}` contains inout ports"),
            )]));
        }

        let mut sim = Self::from_module(design, module, module.name.clone(), init)?;
        sim.eval_combinational();
        Ok(sim)
    }

    fn from_module(
        design: &'a Design,
        module: &'a Module,
        path: String,
        init: &SimulationInit,
    ) -> Result<Self, ErrorReport> {
        if module.is_external {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_EXTERN_MODULE",
                format!("external module `{}` cannot be simulated", module.name),
            )]));
        }
        if module_has_inout(module) {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_SIM_INOUT",
                format!("module `{}` contains inout ports", module.name),
            )]));
        }
        let mut values = HashMap::new();
        let mut memories = HashMap::new();
        for signal in &module.signals {
            let initial_value = module
                .initial_register_values
                .iter()
                .find(|initial| initial.signal == signal.handle)
                .map(|initial| encode_value(initial.value, signal.ty))
                .map(|value| value & mask(signal.width))
                .unwrap_or(0);
            let initial_value = init
                .registers
                .get(&signal.handle)
                .copied()
                .map(|value| value & mask(signal.width))
                .unwrap_or(initial_value);
            values.insert(signal.handle, initial_value & mask(signal.width));
            if let SignalKind::Mem { depth, .. } = signal.kind {
                let mut memory = vec![0; depth];
                for initial in module
                    .initial_memory_values
                    .iter()
                    .filter(|initial| initial.mem == signal.handle)
                {
                    if initial.addr < memory.len() {
                        memory[initial.addr] = encode_value(initial.value, signal.ty);
                    }
                }
                if let Some(overrides) = init.memories.get(&signal.handle) {
                    for (addr, value) in overrides {
                        if *addr < memory.len() {
                            memory[*addr] = *value & mask(signal.width);
                        }
                    }
                }
                memories.insert(signal.handle, memory);
            }
        }

        let mut children = Vec::new();
        for instance in &module.instances {
            let Some(child_module) = design.ir.find_module(&instance.module) else {
                return Err(ErrorReport::new(vec![Diagnostic::new(
                    "E_SIM_INSTANCE",
                    format!("instance target `{}` does not exist", instance.module),
                )]));
            };
            if child_module.is_external {
                return Err(ErrorReport::new(vec![Diagnostic::new(
                    "E_SIM_EXTERN_MODULE",
                    format!(
                        "instance `{}` targets external module `{}`",
                        instance.name, child_module.name
                    ),
                )]));
            }
            children.push(InstanceState {
                instance,
                sim: Box::new(Self::from_module(
                    design,
                    child_module,
                    format!("{path}.{}", instance.name),
                    init,
                )?),
            });
        }

        Ok(Self {
            design,
            module,
            path,
            values,
            memories,
            cover_hits: HashMap::new(),
            children,
        })
    }

    pub fn set(&mut self, signal: Signal, value: u128) {
        let width = self.module.signal(signal).map(|s| s.width).unwrap_or(128);
        self.values.insert(signal, value & mask(width));
        self.eval_combinational();
    }

    pub fn get(&self, signal: Signal) -> u128 {
        *self.values.get(&signal).unwrap_or(&0)
    }

    pub fn peek_mem(&self, mem: Signal, addr: usize) -> Option<u128> {
        self.memories
            .get(&mem)
            .and_then(|memory| memory.get(addr).copied())
    }

    pub fn tick(&mut self) {
        self.tick_checked().expect("assertion failed during tick");
    }

    pub fn tick_checked(&mut self) -> Result<(), ErrorReport> {
        self.eval_combinational();
        self.sample_cover_points(true);
        self.check_clocked_assertions()?;
        let tick = self.capture_tick();
        self.apply_tick(tick);
        self.eval_combinational();
        Ok(())
    }

    pub fn check_assertions(&mut self) -> Result<(), ErrorReport> {
        self.eval_combinational();
        self.sample_cover_points(false);
        let mut diagnostics = Vec::new();
        self.collect_assertion_failures(false, &mut diagnostics);
        if diagnostics.is_empty() {
            Ok(())
        } else {
            Err(ErrorReport::new(diagnostics))
        }
    }

    pub fn cover_hits(&self, name: &str) -> u64 {
        if let Some(hits) = self.cover_hits.get(name) {
            return *hits;
        }
        self.children
            .iter()
            .map(|child| child.sim.cover_hits(name))
            .find(|hits| *hits > 0)
            .unwrap_or(0)
    }

    pub fn cover_report(&self) -> Vec<CoverHit> {
        let mut report = Vec::new();
        self.collect_cover_report(&mut report);
        report.sort_by(|left, right| left.name.cmp(&right.name));
        report
    }

    fn check_clocked_assertions(&mut self) -> Result<(), ErrorReport> {
        let mut diagnostics = Vec::new();
        self.collect_assertion_failures(true, &mut diagnostics);
        if diagnostics.is_empty() {
            Ok(())
        } else {
            Err(ErrorReport::new(diagnostics))
        }
    }

    fn sample_cover_points(&mut self, clocked: bool) {
        self.drive_child_inputs();
        for child in &mut self.children {
            child.sim.eval_combinational();
        }

        let mut hits = Vec::new();
        for cover in &self.module.cover_points {
            if cover.clock.is_some() != clocked {
                continue;
            }
            if let Some(enable) = &cover.enable {
                if self.eval_expr(enable) == 0 {
                    continue;
                }
            }
            if self.eval_expr(&cover.condition) != 0 {
                hits.push(format!("{}.{}", self.path, cover.name));
            }
        }
        for name in hits {
            *self.cover_hits.entry(name).or_insert(0) += 1;
        }

        for child in &mut self.children {
            child.sim.sample_cover_points(clocked);
        }
    }

    fn collect_cover_report(&self, report: &mut Vec<CoverHit>) {
        for cover in &self.module.cover_points {
            let name = format!("{}.{}", self.path, cover.name);
            let hits = self.cover_hits.get(&name).copied().unwrap_or(0);
            report.push(CoverHit { name, hits });
        }
        for child in &self.children {
            child.sim.collect_cover_report(report);
        }
    }

    fn capture_tick(&mut self) -> TickState {
        self.drive_child_inputs();
        for child in &mut self.children {
            child.sim.eval_combinational();
        }

        let mut tick = TickState::default();
        for signal in &self.module.signals {
            let SignalKind::Reg { reset, next, .. } = &signal.kind else {
                continue;
            };
            let value = if let Some(reset) = reset {
                if self.reset_asserted(reset) {
                    encode_value(reset.value, signal.ty)
                } else {
                    self.eval_expr(next.as_ref().expect("validated next"))
                }
            } else {
                self.eval_expr(next.as_ref().expect("validated next"))
            };
            tick.values
                .push((signal.handle, value & mask(signal.width)));
        }

        for write in &self.module.memory_writes {
            if self.eval_expr(&write.enable) != 0 {
                let Some(mem) = self.module.signal(write.mem) else {
                    continue;
                };
                let addr = self.eval_expr(&write.addr) as usize;
                let data = self.eval_expr(&write.data) & mask(mem.width);
                tick.memories.push((write.mem, addr, data));
            }
        }

        for child in &mut self.children {
            tick.children.push(child.sim.capture_tick());
        }

        tick
    }

    fn collect_assertion_failures(&mut self, clocked: bool, diagnostics: &mut Vec<Diagnostic>) {
        self.drive_child_inputs();
        for child in &mut self.children {
            child.sim.eval_combinational();
        }

        for assertion in &self.module.assertions {
            if assertion.clock.is_some() != clocked {
                continue;
            }
            if let Some(enable) = &assertion.enable {
                if self.eval_expr(enable) == 0 {
                    continue;
                }
            }
            if self.eval_expr(&assertion.condition) == 0 {
                diagnostics.push(assertion_failure_diagnostic(&self.module.name, assertion));
            }
        }

        for child in &mut self.children {
            child.sim.collect_assertion_failures(clocked, diagnostics);
        }
    }

    fn apply_tick(&mut self, tick: TickState) {
        for (signal, value) in tick.values {
            self.values.insert(signal, value);
        }
        for (mem, addr, data) in tick.memories {
            if let Some(memory) = self.memories.get_mut(&mem) {
                if addr < memory.len() {
                    memory[addr] = data;
                }
            }
        }
        for (child, child_tick) in self.children.iter_mut().zip(tick.children) {
            child.sim.apply_tick(child_tick);
        }
    }

    fn eval_combinational(&mut self) {
        let limit = self
            .module
            .assignments
            .len()
            .saturating_add(self.module.instances.len())
            .max(1);
        for _ in 0..limit {
            let mut changed = self.apply_async_resets();
            for assignment in &self.module.assignments {
                let Some(dst) = self.module.signal(assignment.dst) else {
                    continue;
                };
                let value = self.eval_expr(&assignment.expr) & mask(dst.width);
                if self.values.insert(assignment.dst, value) != Some(value) {
                    changed = true;
                }
            }
            if self.eval_instances() {
                changed = true;
            }
            if self.apply_async_resets() {
                changed = true;
            }
            if !changed {
                break;
            }
        }
    }

    fn apply_async_resets(&mut self) -> bool {
        let mut writes = Vec::new();
        for signal in &self.module.signals {
            let SignalKind::Reg {
                reset: Some(reset), ..
            } = &signal.kind
            else {
                continue;
            };
            if reset.kind == ResetKind::Async && self.reset_asserted(reset) {
                writes.push((signal.handle, encode_value(reset.value, signal.ty)));
            }
        }

        let mut changed = false;
        for (signal, value) in writes {
            if self.values.insert(signal, value) != Some(value) {
                changed = true;
            }
        }
        changed
    }

    fn reset_asserted(&self, reset: &Reset) -> bool {
        match reset.polarity {
            ResetPolarity::ActiveHigh => self.get(reset.signal) != 0,
            ResetPolarity::ActiveLow => self.get(reset.signal) == 0,
        }
    }

    fn eval_instances(&mut self) -> bool {
        self.drive_child_inputs();

        let mut writes = Vec::new();
        for child in &mut self.children {
            child.sim.eval_combinational();
            let Some(child_module) = self.design.ir.find_module(&child.instance.module) else {
                continue;
            };
            for (port_name, parent_signal) in &child.instance.connections {
                let Some(port) = child_module
                    .signals
                    .iter()
                    .find(|s| s.name == *port_name && matches!(s.kind, SignalKind::Output))
                else {
                    continue;
                };
                let value = child.sim.get(port.handle) & mask(port.width);
                writes.push((*parent_signal, value));
            }
        }

        let mut changed = false;
        for (signal, value) in writes {
            if self.values.insert(signal, value) != Some(value) {
                changed = true;
            }
        }
        changed
    }

    fn drive_child_inputs(&mut self) {
        let parent_values = self.values.clone();
        for child in &mut self.children {
            let Some(child_module) = self.design.ir.find_module(&child.instance.module) else {
                continue;
            };
            let mut input_writes = Vec::new();
            for (port_name, parent_signal) in &child.instance.connections {
                let Some(port) = child_module
                    .signals
                    .iter()
                    .find(|s| s.name == *port_name && matches!(s.kind, SignalKind::Input))
                else {
                    continue;
                };
                let value = parent_values.get(parent_signal).copied().unwrap_or(0);
                input_writes.push((port.handle, value & mask(port.width)));
            }
            for (signal, value) in input_writes {
                child.sim.values.insert(signal, value);
            }
        }
    }

    fn eval_expr(&self, expr: &Expr) -> u128 {
        match expr {
            Expr::Lit { value, ty } => encode_value(*value, *ty),
            Expr::Signal(signal) => self.get(*signal),
            Expr::Not(inner) => !self.eval_expr(inner),
            Expr::And(lhs, rhs) => self.eval_expr(lhs) & self.eval_expr(rhs),
            Expr::Or(lhs, rhs) => self.eval_expr(lhs) | self.eval_expr(rhs),
            Expr::Xor(lhs, rhs) => self.eval_expr(lhs) ^ self.eval_expr(rhs),
            Expr::Add(lhs, rhs) => self.eval_expr(lhs).wrapping_add(self.eval_expr(rhs)),
            Expr::Sub(lhs, rhs) => self.eval_expr(lhs).wrapping_sub(self.eval_expr(rhs)),
            Expr::Mul(lhs, rhs) => self.eval_expr(lhs).wrapping_mul(self.eval_expr(rhs)),
            Expr::Eq(lhs, rhs) => u128::from(self.eval_expr(lhs) == self.eval_expr(rhs)),
            Expr::Ne(lhs, rhs) => u128::from(self.eval_expr(lhs) != self.eval_expr(rhs)),
            Expr::Lt(lhs, rhs) => {
                let lhs_ty = expr_type_no_diag(self.module, lhs).unwrap_or(uint(128));
                if lhs_ty.is_signed() {
                    u128::from(
                        signed_value(self.eval_expr(lhs), lhs_ty.width)
                            < signed_value(self.eval_expr(rhs), lhs_ty.width),
                    )
                } else {
                    u128::from(self.eval_expr(lhs) < self.eval_expr(rhs))
                }
            }
            Expr::Mux {
                cond,
                then_expr,
                else_expr,
            } => {
                if self.eval_expr(cond) != 0 {
                    self.eval_expr(then_expr)
                } else {
                    self.eval_expr(else_expr)
                }
            }
            Expr::Slice { expr, lsb, width } => (self.eval_expr(expr) >> *lsb) & mask(*width),
            Expr::Zext { expr, width } | Expr::Trunc { expr, width } => {
                self.eval_expr(expr) & mask(*width)
            }
            Expr::Sext { expr, width } => {
                let inner_ty = expr_type_no_diag(self.module, expr).unwrap_or(uint(*width));
                encode_value(
                    signed_value(self.eval_expr(expr), inner_ty.width),
                    sint(*width),
                )
            }
            Expr::Cast {
                expr,
                signedness: _,
            } => self.eval_expr(expr),
            Expr::Concat(parts) => {
                let mut value = 0;
                let mut shift = 0;
                for part in parts.iter().rev() {
                    let width = expr_width_no_diag(self.module, part).unwrap_or(0);
                    value |= self.eval_expr(part) << shift;
                    shift += width;
                }
                value
            }
            Expr::MemRead { mem, addr } => {
                let addr = self.eval_expr(addr) as usize;
                self.memories
                    .get(mem)
                    .and_then(|memory| memory.get(addr).copied())
                    .unwrap_or(0)
            }
        }
    }
}

fn assertion_failure_diagnostic(module_name: &str, assertion: &Assertion) -> Diagnostic {
    let message = assertion.message.as_deref().unwrap_or("assertion failed");
    Diagnostic::new(
        "E_SIM_ASSERT",
        format!("assertion `{}` failed: {message}", assertion.name),
    )
    .with_module(module_name.to_string())
}

#[derive(Clone, Debug)]
pub struct VcdTrace<'a> {
    design: &'a Design,
    module: &'a Module,
    root: VcdTraceScope,
    last_time: Option<u64>,
    output: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VcdTraceOptions {
    pub memories: VcdMemoryTrace,
}

impl Default for VcdTraceOptions {
    fn default() -> Self {
        Self {
            memories: VcdMemoryTrace::Off,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VcdMemoryTrace {
    Off,
    All,
}

#[derive(Clone, Debug)]
struct VcdTraceScope {
    name: String,
    module: ModuleId,
    signals: Vec<VcdTraceSignal>,
    memories: Vec<VcdTraceMemory>,
    children: Vec<VcdTraceScope>,
}

#[derive(Clone, Debug)]
struct VcdTraceSignal {
    signal: Signal,
    name: String,
    width: Width,
    id: String,
}

#[derive(Clone, Debug)]
struct VcdTraceMemory {
    signal: Signal,
    name: String,
    width: Width,
    words: Vec<VcdTraceMemoryWord>,
}

#[derive(Clone, Debug)]
struct VcdTraceMemoryWord {
    addr: usize,
    id: String,
}

impl<'a> VcdTrace<'a> {
    pub fn new(design: &'a Design, module_name: &str) -> Result<Self, ErrorReport> {
        Self::with_options(design, module_name, VcdTraceOptions::default())
    }

    pub fn with_options(
        design: &'a Design,
        module_name: &str,
        options: VcdTraceOptions,
    ) -> Result<Self, ErrorReport> {
        design.validate()?;
        let Some(module) = design.ir.find_module(module_name) else {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_VCD_MODULE",
                format!("module `{module_name}` does not exist"),
            )]));
        };
        if module.is_external {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_VCD_EXTERN_MODULE",
                format!("external module `{module_name}` cannot be traced"),
            )]));
        }
        if module_has_inout(module) {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_VCD_INOUT",
                format!("module `{module_name}` contains inout ports"),
            )]));
        }

        let mut next_id = 0;
        let root = build_vcd_scope(design, module, module.name.clone(), options, &mut next_id)?;

        let mut output = String::new();
        output.push_str("$timescale 1ns $end\n");
        emit_vcd_scope_header(&mut output, &root);
        output.push_str("$enddefinitions $end\n");

        Ok(Self {
            design,
            module,
            root,
            last_time: None,
            output,
        })
    }

    pub fn sample(&mut self, sim: &Simulator<'a>, time: u64) -> Result<(), ErrorReport> {
        if !std::ptr::eq(self.design, sim.design) || self.module.id != sim.module.id {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_VCD_SIM_MODULE",
                format!(
                    "trace for module `{}` cannot sample simulator for module `{}`",
                    self.module.name, sim.module.name
                ),
            )]));
        }
        if let Some(last_time) = self.last_time {
            if time < last_time {
                return Err(ErrorReport::new(vec![Diagnostic::new(
                    "E_VCD_TIME",
                    format!("sample time {time} is earlier than previous sample time {last_time}"),
                )]));
            }
        }

        self.output.push_str(&format!("#{time}\n"));
        sample_vcd_scope(&self.root, sim, &mut self.output)?;
        self.last_time = Some(time);
        Ok(())
    }

    pub fn finish(self) -> String {
        self.output
    }
}

fn build_vcd_scope(
    design: &Design,
    module: &Module,
    name: String,
    options: VcdTraceOptions,
    next_id: &mut usize,
) -> Result<VcdTraceScope, ErrorReport> {
    if module.is_external {
        return Err(ErrorReport::new(vec![Diagnostic::new(
            "E_VCD_EXTERN_MODULE",
            format!("external module `{}` cannot be traced", module.name),
        )]));
    }
    if module_has_inout(module) {
        return Err(ErrorReport::new(vec![Diagnostic::new(
            "E_VCD_INOUT",
            format!("module `{}` contains inout ports", module.name),
        )]));
    }

    let mut signals = Vec::new();
    for signal in module
        .signals
        .iter()
        .filter(|signal| !matches!(signal.kind, SignalKind::Mem { .. }))
    {
        signals.push(VcdTraceSignal {
            signal: signal.handle,
            name: signal.name.clone(),
            width: signal.width,
            id: vcd_id(*next_id),
        });
        *next_id += 1;
    }

    let mut memories = Vec::new();
    if options.memories == VcdMemoryTrace::All {
        for signal in &module.signals {
            let SignalKind::Mem {
                data_width, depth, ..
            } = signal.kind
            else {
                continue;
            };
            let mut words = Vec::with_capacity(depth);
            for addr in 0..depth {
                words.push(VcdTraceMemoryWord {
                    addr,
                    id: vcd_id(*next_id),
                });
                *next_id += 1;
            }
            memories.push(VcdTraceMemory {
                signal: signal.handle,
                name: signal.name.clone(),
                width: data_width,
                words,
            });
        }
    }

    let mut children = Vec::new();
    for instance in &module.instances {
        let Some(child_module) = design.ir.find_module(&instance.module) else {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_VCD_INSTANCE",
                format!("instance target `{}` does not exist", instance.module),
            )]));
        };
        children.push(build_vcd_scope(
            design,
            child_module,
            instance.name.clone(),
            options,
            next_id,
        )?);
    }

    Ok(VcdTraceScope {
        name,
        module: module.id,
        signals,
        memories,
        children,
    })
}

fn emit_vcd_scope_header(output: &mut String, scope: &VcdTraceScope) {
    output.push_str(&format!("$scope module {} $end\n", vcd_ref(&scope.name)));
    for signal in &scope.signals {
        output.push_str(&format!(
            "$var wire {} {} {} $end\n",
            signal.width,
            signal.id,
            vcd_ref(&signal.name)
        ));
    }
    for memory in &scope.memories {
        output.push_str(&format!("$scope module {} $end\n", vcd_ref(&memory.name)));
        for word in &memory.words {
            output.push_str(&format!(
                "$var wire {} {} {} $end\n",
                memory.width, word.id, word.addr
            ));
        }
        output.push_str("$upscope $end\n");
    }
    for child in &scope.children {
        emit_vcd_scope_header(output, child);
    }
    output.push_str("$upscope $end\n");
}

fn sample_vcd_scope(
    scope: &VcdTraceScope,
    sim: &Simulator<'_>,
    output: &mut String,
) -> Result<(), ErrorReport> {
    if scope.module != sim.module.id || scope.children.len() != sim.children.len() {
        return Err(ErrorReport::new(vec![Diagnostic::new(
            "E_VCD_SIM_MODULE",
            format!(
                "trace scope `{}` does not match simulator module `{}`",
                scope.name, sim.module.name
            ),
        )]));
    }

    for signal in &scope.signals {
        let value = sim.get(signal.signal) & mask(signal.width);
        output.push_str(&vcd_value(value, signal.width, &signal.id));
    }
    for memory in &scope.memories {
        let Some(values) = sim.memories.get(&memory.signal) else {
            return Err(ErrorReport::new(vec![Diagnostic::new(
                "E_VCD_SIM_MODULE",
                format!(
                    "trace memory `{}` is missing from simulator module `{}`",
                    memory.name, sim.module.name
                ),
            )]));
        };
        for word in &memory.words {
            let value = values.get(word.addr).copied().unwrap_or(0) & mask(memory.width);
            output.push_str(&vcd_value(value, memory.width, &word.id));
        }
    }
    for (child_scope, child_sim) in scope.children.iter().zip(&sim.children) {
        sample_vcd_scope(child_scope, child_sim.sim.as_ref(), output)?;
    }
    Ok(())
}

fn vcd_id(mut index: usize) -> String {
    const FIRST: u8 = b'!';
    const COUNT: usize = 94;

    let mut id = String::new();
    loop {
        id.push((FIRST + (index % COUNT) as u8) as char);
        index /= COUNT;
        if index == 0 {
            break;
        }
    }
    id
}

fn vcd_ref(name: &str) -> String {
    if name
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.'))
    {
        name.to_string()
    } else {
        format!("\\{} ", name.replace('\\', "\\\\"))
    }
}

fn vcd_value(value: u128, width: Width, id: &str) -> String {
    if width == 1 {
        format!("{}{}\n", value & 1, id)
    } else {
        format!("b{:0width$b} {id}\n", value, width = width as usize)
    }
}

fn expr_width_no_diag(module: &Module, expr: &Expr) -> Option<Width> {
    let mut diagnostics = Vec::new();
    expr_width(module, expr, &mut diagnostics)
}

fn expr_type_no_diag(module: &Module, expr: &Expr) -> Option<BitType> {
    let mut diagnostics = Vec::new();
    expr_type(module, expr, &mut diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counter_design() -> (Design, Signal, Signal, Signal, Signal) {
        let mut design = Design::new();
        let (rst, en, out, count);
        {
            let mut m = design.module("Counter");
            let clk = m.input("clk", 1);
            rst = m.input("rst", 1);
            en = m.input("en", 1);
            out = m.output("out", 8);
            count = m.reg("count", 8);
            m.clock(count, clk);
            m.reset(count, rst, 0);
            m.next(count, mux(en, count + lit(1, 8), count));
            m.assign(out, count);
        }
        (design, rst, en, out, count)
    }

    fn req_bundle_type() -> BundleType {
        bundle_type(
            "Req",
            [
                field("valid", uint(1)),
                nested("meta", bundle_type("ReqMeta", [field("tag", uint(4))])),
            ],
        )
    }

    fn resp_bundle_type() -> BundleType {
        bundle_type("Resp", [field("valid", uint(1)), field("data", uint(8))])
    }

    fn bus_interface_type() -> InterfaceType {
        interface_type(
            "Bus",
            [
                iface_input("req", req_bundle_type()),
                iface_output("resp", resp_bundle_type()),
                iface_input("ready", uint(1)),
            ],
        )
    }

    fn rv_payload_bundle_type() -> BundleType {
        bundle_type("Payload", [field("data", uint(8)), field("last", uint(1))])
    }

    #[test]
    fn validates_counter() {
        let (design, ..) = counter_design();
        design.validate().unwrap();
    }

    #[test]
    fn compiles_counter_into_normalized_registers() {
        let (design, ..) = counter_design();
        let compiled = compile(&design).unwrap();
        let counter = compiled.find_module("Counter").unwrap();

        assert_eq!(counter.assignments.len(), 1);
        assert_eq!(counter.registers.len(), 1);
        assert_eq!(counter.registers[0].next.width, 8);
        assert_eq!(counter.assignments[0].expr.width, 8);
    }

    #[test]
    fn rejects_width_mismatch() {
        let mut design = Design::new();
        {
            let mut m = design.module("Bad");
            let a = m.input("a", 1);
            let y = m.output("y", 2);
            m.assign(y, a);
        }

        let err = design.validate().unwrap_err();
        assert!(err.diagnostics.iter().any(|d| d.code == "E_ASSIGN_TYPE"));
    }

    #[test]
    fn simulates_counter() {
        let (design, rst, en, out, _) = counter_design();
        let mut sim = Simulator::new(&design, "Counter").unwrap();

        sim.set(rst, 1);
        sim.tick();
        assert_eq!(sim.get(out), 0);

        sim.set(rst, 0);
        sim.set(en, 1);
        sim.tick();
        sim.tick();
        assert_eq!(sim.get(out), 2);
    }

    #[test]
    fn validates_assertion_types() {
        let mut design = Design::new();
        {
            let mut m = design.module("BadAssertions");
            let clk = m.input("clk", uint(2));
            let enable = m.input("enable", uint(2));
            let condition = m.input("condition", uint(8));
            m.assert("bad_condition", condition);
            m.assert_when("bad_enable", enable, condition.value().eq_expr(lit_u(0, 8)));
            m.assert_clocked("bad_clock", clk, condition.value().eq_expr(lit_u(0, 8)));
        }

        let err = design.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_ASSERT_CONDITION_TYPE"));
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_ASSERT_ENABLE_TYPE"));
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_ASSERT_CLOCK_WIDTH"));
    }

    #[test]
    fn simulates_combinational_assertions() {
        let mut design = Design::new();
        let value;
        {
            let mut m = design.module("AssertComb");
            value = m.input("value", uint(4));
            m.assert_msg(
                "value_lt_10",
                value.value().lt_expr(lit_u(10, 4)),
                "value exceeded limit",
            );
        }

        let mut sim = Simulator::new(&design, "AssertComb").unwrap();
        sim.set(value, 9);
        sim.check_assertions().unwrap();
        sim.set(value, 10);
        let err = sim.check_assertions().unwrap_err();
        assert!(err.diagnostics.iter().any(|d| {
            d.code == "E_SIM_ASSERT"
                && d.message.contains("value_lt_10")
                && d.message.contains("value exceeded limit")
        }));
    }

    #[test]
    fn simulates_clocked_assertions_on_checked_tick() {
        let mut design = Design::new();
        let (clk, count);
        {
            let mut m = design.module("AssertClocked");
            clk = m.input("clk", uint(1));
            count = m.reg("count", uint(4));
            m.clock(count, clk);
            m.next(count, count + lit_u(1, 4));
            m.assert_clocked("count_lt_2", clk, count.value().lt_expr(lit_u(2, 4)));
        }

        let mut sim = Simulator::new(&design, "AssertClocked").unwrap();
        sim.tick_checked().unwrap();
        sim.tick_checked().unwrap();
        let err = sim.tick_checked().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| { d.code == "E_SIM_ASSERT" && d.message.contains("count_lt_2") }));
    }

    #[test]
    fn validates_cover_types() {
        let mut design = Design::new();
        {
            let mut m = design.module("BadCovers");
            let clk = m.input("clk", uint(2));
            let enable = m.input("enable", uint(2));
            let condition = m.input("condition", uint(8));
            m.cover("bad_condition", condition);
            m.cover_when("bad_enable", enable, condition.value().eq_expr(lit_u(0, 8)));
            m.cover_clocked("bad_clock", clk, condition.value().eq_expr(lit_u(0, 8)));
        }

        let err = design.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_COVER_CONDITION_TYPE"));
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_COVER_ENABLE_TYPE"));
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_COVER_CLOCK_WIDTH"));
    }

    #[test]
    fn simulates_combinational_cover_points() {
        let mut design = Design::new();
        let value;
        {
            let mut m = design.module("CoverComb");
            value = m.input("value", uint(4));
            m.cover("value_is_3", value.value().eq_expr(lit_u(3, 4)));
        }

        let mut sim = Simulator::new(&design, "CoverComb").unwrap();
        sim.set(value, 2);
        sim.check_assertions().unwrap();
        assert_eq!(sim.cover_hits("CoverComb.value_is_3"), 0);
        assert!(sim.cover_report().contains(&CoverHit {
            name: "CoverComb.value_is_3".to_string(),
            hits: 0,
        }));

        sim.set(value, 3);
        sim.check_assertions().unwrap();
        sim.check_assertions().unwrap();
        assert_eq!(sim.cover_hits("CoverComb.value_is_3"), 2);
    }

    #[test]
    fn simulates_clocked_cover_points_on_checked_tick() {
        let mut design = Design::new();
        let (clk, count);
        {
            let mut m = design.module("CoverClocked");
            clk = m.input("clk", uint(1));
            count = m.reg("count", uint(4));
            m.clock(count, clk);
            m.next(count, count + lit_u(1, 4));
            m.cover_clocked("count_is_1", clk, count.value().eq_expr(lit_u(1, 4)));
        }

        let mut sim = Simulator::new(&design, "CoverClocked").unwrap();
        sim.tick_checked().unwrap();
        assert_eq!(sim.cover_hits("CoverClocked.count_is_1"), 0);
        sim.tick_checked().unwrap();
        assert_eq!(sim.cover_hits("CoverClocked.count_is_1"), 1);
    }

    #[test]
    fn reports_hierarchical_cover_hits() {
        let mut design = Design::new();
        {
            let mut m = design.module("Child");
            let fire = m.input("fire", uint(1));
            m.cover("seen_fire", fire);
        }
        let fire;
        {
            let mut m = design.module("Top");
            fire = m.input("fire", uint(1));
            m.instance("u_child", "Child", [("fire", fire)]);
        }

        let mut sim = Simulator::new(&design, "Top").unwrap();
        sim.set(fire, 1);
        sim.check_assertions().unwrap();

        assert_eq!(sim.cover_hits("Top.u_child.seen_fire"), 1);
        assert!(sim.cover_report().contains(&CoverHit {
            name: "Top.u_child.seen_fire".to_string(),
            hits: 1,
        }));
    }

    #[test]
    fn traces_counter_to_vcd() {
        let (design, rst, en, _, count) = counter_design();
        let mut sim = Simulator::new(&design, "Counter").unwrap();
        let mut trace = VcdTrace::new(&design, "Counter").unwrap();

        trace.sample(&sim, 0).unwrap();
        sim.set(rst, 0);
        sim.set(en, 1);
        sim.tick();
        trace.sample(&sim, 1).unwrap();

        let vcd = trace.finish();
        assert!(vcd.contains("$timescale 1ns $end"));
        assert!(vcd.contains("$scope module Counter $end"));
        assert!(vcd.contains("$var wire 8 % count $end"));
        assert!(vcd.contains("#0\n"));
        assert!(vcd.contains("#1\n"));
        assert!(vcd.contains("b00000000 %\n"));
        assert!(vcd.contains("b00000001 %\n"));
        assert_eq!(sim.get(count), 1);
    }

    #[test]
    fn traces_instance_hierarchy_to_vcd() {
        let mut design = Design::new();
        {
            let mut m = design.module("AddOne");
            let a = m.input("a", 8);
            let y = m.output("y", 8);
            m.assign(y, a + lit(1, 8));
        }
        let (a, y);
        {
            let mut m = design.module("Top");
            a = m.input("a", 8);
            y = m.output("y", 8);
            m.instance("u_add", "AddOne", [("a", a), ("y", y)]);
        }

        let mut sim = Simulator::new(&design, "Top").unwrap();
        let mut trace = VcdTrace::new(&design, "Top").unwrap();
        sim.set(a, 0x2a);
        trace.sample(&sim, 0).unwrap();

        let vcd = trace.finish();
        assert!(vcd.contains("$scope module Top $end"));
        assert!(vcd.contains("$scope module u_add $end"));
        assert!(vcd.contains("$var wire 8 ! a $end"));
        assert!(vcd.contains("$var wire 8 \" y $end"));
        assert!(vcd.contains("$var wire 8 # a $end"));
        assert!(vcd.contains("$var wire 8 $ y $end"));
        assert!(vcd.contains("b00101010 !\n"));
        assert!(vcd.contains("b00101011 \"\n"));
        assert!(vcd.contains("b00101010 #\n"));
        assert!(vcd.contains("b00101011 $\n"));
        assert_eq!(sim.get(y), 0x2b);
    }

    #[test]
    fn traces_nested_instance_hierarchy_to_vcd() {
        let mut design = Design::new();
        {
            let mut m = design.module("Leaf");
            let a = m.input("a", 4);
            let y = m.output("y", 4);
            m.assign(y, a + lit(1, 4));
        }
        {
            let mut m = design.module("Middle");
            let a = m.input("a", 4);
            let y = m.output("y", 4);
            m.instance("u_leaf", "Leaf", [("a", a), ("y", y)]);
        }
        let a;
        {
            let mut m = design.module("Top");
            a = m.input("a", 4);
            let y = m.output("y", 4);
            m.instance("u_mid", "Middle", [("a", a), ("y", y)]);
        }

        let mut sim = Simulator::new(&design, "Top").unwrap();
        let mut trace = VcdTrace::new(&design, "Top").unwrap();
        sim.set(a, 3);
        trace.sample(&sim, 0).unwrap();

        let vcd = trace.finish();
        assert!(vcd.contains("$scope module Top $end"));
        assert!(vcd.contains("$scope module u_mid $end"));
        assert!(vcd.contains("$scope module u_leaf $end"));
        assert!(vcd.contains("b0011 !\n"));
        assert!(vcd.contains("b0100 &\n"));
    }

    #[test]
    fn vcd_trace_memory_words_are_opt_in() {
        let mut design = Design::new();
        let (clk, we, waddr, wdata);
        {
            let mut m = design.module("RegFile");
            clk = m.input("clk", 1);
            we = m.input("we", 1);
            waddr = m.input("waddr", 1);
            wdata = m.input("wdata", 8);
            let raddr = m.input("raddr", 1);
            let rdata = m.output("rdata", 8);
            let regs = m.mem("regs", 1, uint(8), 2);
            m.mem_write(regs, clk, we, waddr, wdata);
            let read = m.mem_read(regs, raddr);
            m.assign(rdata, read);
        }

        let mut sim = Simulator::new(&design, "RegFile").unwrap();
        sim.set(we, 1);
        sim.set(waddr, 1);
        sim.set(wdata, 0x5a);
        sim.tick();

        let mut default_trace = VcdTrace::new(&design, "RegFile").unwrap();
        default_trace.sample(&sim, 0).unwrap();
        assert!(!default_trace.finish().contains("$scope module regs $end"));

        let mut memory_trace = VcdTrace::with_options(
            &design,
            "RegFile",
            VcdTraceOptions {
                memories: VcdMemoryTrace::All,
            },
        )
        .unwrap();
        memory_trace.sample(&sim, 0).unwrap();
        let vcd = memory_trace.finish();

        assert!(vcd.contains("$scope module regs $end"));
        assert!(vcd.contains("$var wire 8 ' 0 $end"));
        assert!(vcd.contains("$var wire 8 ( 1 $end"));
        assert!(vcd.contains("b00000000 '\n"));
        assert!(vcd.contains("b01011010 (\n"));
    }

    #[test]
    fn vcd_trace_memory_words_in_child_scopes() {
        let mut design = Design::new();
        {
            let mut m = design.module("MemChild");
            let clk = m.input("clk", 1);
            let we = m.input("we", 1);
            let waddr = m.input("waddr", 1);
            let wdata = m.input("wdata", 8);
            let rdata = m.output("rdata", 8);
            let regs = m.mem("regs", 1, uint(8), 2);
            m.mem_write(regs, clk, we, waddr, wdata);
            let read = m.mem_read(regs, waddr);
            m.assign(rdata, read);
        }
        let (clk, we, waddr, wdata);
        {
            let mut m = design.module("Top");
            clk = m.input("clk", 1);
            we = m.input("we", 1);
            waddr = m.input("waddr", 1);
            wdata = m.input("wdata", 8);
            let rdata = m.output("rdata", 8);
            m.instance(
                "u_mem",
                "MemChild",
                [
                    ("clk", clk),
                    ("we", we),
                    ("waddr", waddr),
                    ("wdata", wdata),
                    ("rdata", rdata),
                ],
            );
        }

        let mut sim = Simulator::new(&design, "Top").unwrap();
        sim.set(we, 1);
        sim.set(waddr, 0);
        sim.set(wdata, 0xa5);
        sim.tick();

        let mut trace = VcdTrace::with_options(
            &design,
            "Top",
            VcdTraceOptions {
                memories: VcdMemoryTrace::All,
            },
        )
        .unwrap();
        trace.sample(&sim, 0).unwrap();
        let vcd = trace.finish();

        assert!(vcd.contains("$scope module Top $end"));
        assert!(vcd.contains("$scope module u_mem $end"));
        assert!(vcd.contains("$scope module regs $end"));
        assert!(vcd.contains("b10100101 +\n"));
    }

    #[test]
    fn rejects_vcd_trace_for_unsupported_modules() {
        let mut missing = Design::new();
        {
            missing.module("Top");
        }
        let err = VcdTrace::new(&missing, "Nope").unwrap_err();
        assert!(err.diagnostics.iter().any(|d| d.code == "E_VCD_MODULE"));

        let mut external = Design::new();
        {
            external.extern_module("Vendor").input("a", 1);
        }
        let err = VcdTrace::new(&external, "Vendor").unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_VCD_EXTERN_MODULE"));

        let mut inout = Design::new();
        {
            let mut m = inout.module("Pad");
            m.inout("pad", 1);
        }
        let err = VcdTrace::new(&inout, "Pad").unwrap_err();
        assert!(err.diagnostics.iter().any(|d| d.code == "E_VCD_INOUT"));

        let mut child_external = Design::new();
        {
            let mut ext = child_external.extern_module("Vendor");
            ext.input("a", uint(1));
            ext.output("y", uint(1));
        }
        {
            let mut m = child_external.module("Top");
            let a = m.input("a", uint(1));
            let y = m.output("y", uint(1));
            m.instance("u_vendor", "Vendor", [("a", a), ("y", y)]);
        }
        let err = VcdTrace::new(&child_external, "Top").unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_VCD_EXTERN_MODULE"));
    }

    #[test]
    fn rejects_decreasing_vcd_sample_time() {
        let (design, ..) = counter_design();
        let sim = Simulator::new(&design, "Counter").unwrap();
        let mut trace = VcdTrace::new(&design, "Counter").unwrap();

        trace.sample(&sim, 10).unwrap();
        let err = trace.sample(&sim, 9).unwrap_err();
        assert!(err.diagnostics.iter().any(|d| d.code == "E_VCD_TIME"));
    }

    #[test]
    fn rejects_sampling_different_simulator_module() {
        let mut design = Design::new();
        {
            let mut a = design.module("A");
            a.output("out", 1);
        }
        {
            let mut b = design.module("B");
            b.output("out", 1);
        }
        let sim = Simulator::new(&design, "B").unwrap();
        let mut trace = VcdTrace::new(&design, "A").unwrap();

        let err = trace.sample(&sim, 0).unwrap_err();
        assert!(err.diagnostics.iter().any(|d| d.code == "E_VCD_SIM_MODULE"));
    }

    #[test]
    fn simulates_active_high_async_reset_immediately() {
        let mut design = Design::new();
        let (rst, en, out);
        {
            let mut m = design.module("AsyncCounter");
            let clk = m.input("clk", uint(1));
            rst = m.input("rst", uint(1));
            en = m.input("en", uint(1));
            out = m.output("out", uint(8));
            let count = m.reg("count", uint(8));
            m.clock(count, clk);
            m.async_reset(count, rst, 0);
            m.next(count, mux(en, count + lit(1, 8), count));
            m.assign(out, count);
        }

        let mut sim = Simulator::new(&design, "AsyncCounter").unwrap();
        sim.set(en, 1);
        sim.tick();
        sim.tick();
        assert_eq!(sim.get(out), 2);

        sim.set(rst, 1);
        assert_eq!(sim.get(out), 0);
        sim.tick();
        assert_eq!(sim.get(out), 0);

        sim.set(rst, 0);
        sim.tick();
        assert_eq!(sim.get(out), 1);
    }

    #[test]
    fn validates_and_simulates_memory_write_read() {
        let mut design = Design::new();
        let (we, waddr, wdata, raddr, rdata, mem);
        {
            let mut m = design.module("RegFile");
            let clk = m.input("clk", 1);
            we = m.input("we", 1);
            waddr = m.input("waddr", 2);
            wdata = m.input("wdata", 8);
            raddr = m.input("raddr", 2);
            rdata = m.output("rdata", 8);
            mem = m.mem("regs", 2, 8, 4);

            m.mem_write(mem, clk, we, waddr, wdata);
            let read = m.mem_read(mem, raddr);
            m.assign(rdata, read);
        }

        design.validate().unwrap();
        let mut sim = Simulator::new(&design, "RegFile").unwrap();
        sim.set(we, 1);
        sim.set(waddr, 2);
        sim.set(wdata, 0xab);
        sim.tick();
        assert_eq!(sim.peek_mem(mem, 2), Some(0xab));

        sim.set(raddr, 2);
        assert_eq!(sim.get(rdata), 0xab);
    }

    #[test]
    fn rejects_memory_address_width_mismatch() {
        let mut design = Design::new();
        {
            let mut m = design.module("BadMem");
            let clk = m.input("clk", 1);
            let we = m.input("we", 1);
            let addr = m.input("addr", 3);
            let data = m.input("data", 8);
            let mem = m.mem("mem", 2, 8, 4);
            m.mem_write(mem, clk, we, addr, data);
        }

        let err = design.validate().unwrap_err();
        assert!(err.diagnostics.iter().any(|d| d.code == "E_MEM_ADDR_WIDTH"));
    }

    #[test]
    fn simulates_simple_instance() {
        let mut design = Design::new();
        {
            let mut m = design.module("Child");
            let a = m.input("a", 4);
            let y = m.output("y", 4);
            m.assign(y, a + lit(1, 4));
        }
        let (a, y);
        {
            let mut m = design.module("Top");
            a = m.input("a", 4);
            y = m.output("y", 4);
            m.instance("u_child", "Child", [("a", a), ("y", y)]);
        }

        design.validate().unwrap();
        let mut sim = Simulator::new(&design, "Top").unwrap();
        sim.set(a, 6);
        assert_eq!(sim.get(y), 7);
    }

    #[test]
    fn compiles_external_module_instance() {
        let mut design = Design::new();
        {
            let mut ext = design.extern_module("VendorAddOne");
            ext.input("a", uint(8));
            ext.output("y", uint(8));
        }
        {
            let mut m = design.module("Top");
            let a = m.input("a", 8);
            let y = m.output("y", 8);
            m.instance("u_vendor", "VendorAddOne", [("a", a), ("y", y)]);
        }

        design.validate().unwrap();
        let compiled = design.compile().unwrap();
        let ext = compiled.find_module("VendorAddOne").unwrap();
        assert!(ext.is_external);
        let top = compiled.find_module("Top").unwrap();
        assert_eq!(top.instances[0].module, "VendorAddOne");
        assert_eq!(top.instances[0].connections.len(), 2);
    }

    #[test]
    fn rejects_external_module_instance_mismatches() {
        let mut missing_extra_dup = Design::new();
        {
            let mut ext = missing_extra_dup.extern_module("Vendor");
            ext.input("a", uint(8));
            ext.output("y", uint(8));
        }
        {
            let mut m = missing_extra_dup.module("Top");
            let a = m.input("a", 8);
            let y = m.output("y", 8);
            m.instance("u_vendor", "Vendor", [("a", a), ("a", a), ("z", y)]);
        }
        let err = missing_extra_dup.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_INSTANCE_PORT_DUP"));
        assert!(err.diagnostics.iter().any(|d| d.code == "E_INSTANCE_PORT"));
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_INSTANCE_UNCONNECTED"));

        let mut type_mismatch = Design::new();
        {
            let mut ext = type_mismatch.extern_module("Vendor");
            ext.input("a", sint(8));
            ext.output("y", uint(8));
        }
        {
            let mut m = type_mismatch.module("Top");
            let a = m.input("a", uint(8));
            let y = m.output("y", uint(8));
            m.instance("u_vendor", "Vendor", [("a", a), ("y", y)]);
        }
        let err = type_mismatch.validate().unwrap_err();
        assert!(err.diagnostics.iter().any(|d| d.code == "E_INSTANCE_TYPE"));

        let mut bad_output_target = Design::new();
        {
            let mut ext = bad_output_target.extern_module("Vendor");
            ext.input("a", uint(8));
            ext.output("y", uint(8));
        }
        {
            let mut m = bad_output_target.module("Top");
            let a = m.input("a", uint(8));
            let y = m.input("y", uint(8));
            m.instance("u_vendor", "Vendor", [("a", a), ("y", y)]);
        }
        let err = bad_output_target.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_INSTANCE_OUTPUT_TARGET"));
    }

    #[test]
    fn compiled_json_includes_external_module_marker() {
        let mut design = Design::new();
        {
            let mut ext = design.extern_module("Vendor");
            ext.input("a", uint(1));
            ext.output("y", uint(1));
        }

        let json = design.compile().unwrap().to_json_pretty().unwrap();
        assert!(json.contains("\"is_external\""));
        assert!(json.contains("true"));
    }

    #[test]
    fn rejects_simulating_external_modules() {
        let mut top_external = Design::new();
        {
            let mut ext = top_external.extern_module("Vendor");
            ext.input("a", uint(1));
            ext.output("y", uint(1));
        }
        let err = Simulator::new(&top_external, "Vendor").unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_SIM_EXTERN_MODULE"));

        let mut child_external = Design::new();
        {
            let mut ext = child_external.extern_module("Vendor");
            ext.input("a", uint(1));
            ext.output("y", uint(1));
        }
        {
            let mut m = child_external.module("Top");
            let a = m.input("a", uint(1));
            let y = m.output("y", uint(1));
            m.instance("u_vendor", "Vendor", [("a", a), ("y", y)]);
        }
        let err = Simulator::new(&child_external, "Top").unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_SIM_EXTERN_MODULE"));
    }

    #[test]
    fn compiles_inout_external_module_instance() {
        let mut design = Design::new();
        {
            let mut ext = design.extern_module("IOBUF");
            ext.inout("PAD", uint(1));
            ext.input("I", uint(1));
            ext.input("T", uint(1));
            ext.output("O", uint(1));
        }
        {
            let mut m = design.module("Top");
            let pad = m.inout("pad", uint(1));
            let i = m.input("i", uint(1));
            let t = m.input("t", uint(1));
            let o = m.output("o", uint(1));
            m.instance(
                "u_iobuf",
                "IOBUF",
                [("PAD", pad), ("I", i), ("T", t), ("O", o)],
            );
        }

        design.validate().unwrap();
        let compiled = design.compile().unwrap();
        let top = compiled.find_module("Top").unwrap();
        let pad = top.instances[0]
            .connections
            .iter()
            .find(|connection| connection.port == "PAD")
            .unwrap();
        assert_eq!(pad.direction, PortDirection::Inout);

        let json = compiled.to_json_pretty().unwrap();
        assert!(json.contains("\"Inout\""));
    }

    #[test]
    fn rejects_invalid_inout_usage() {
        let mut type_mismatch = Design::new();
        {
            let mut ext = type_mismatch.extern_module("IOBUF");
            ext.inout("PAD", uint(8));
        }
        {
            let mut m = type_mismatch.module("Top");
            let pad = m.inout("pad", uint(1));
            m.instance("u_iobuf", "IOBUF", [("PAD", pad)]);
        }
        let err = type_mismatch.validate().unwrap_err();
        assert!(err.diagnostics.iter().any(|d| d.code == "E_INSTANCE_TYPE"));

        let mut bad_target = Design::new();
        {
            let mut ext = bad_target.extern_module("IOBUF");
            ext.inout("PAD", uint(1));
        }
        {
            let mut m = bad_target.module("Top");
            let pad = m.input("pad", uint(1));
            m.instance("u_iobuf", "IOBUF", [("PAD", pad)]);
        }
        let err = bad_target.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_INSTANCE_INOUT_TARGET"));

        let mut assign_inout = Design::new();
        {
            let mut m = assign_inout.module("Top");
            let pad = m.inout("pad", uint(1));
            let x = m.input("x", uint(1));
            m.assign(pad, x);
        }
        let err = assign_inout.validate().unwrap_err();
        assert!(err.diagnostics.iter().any(|d| d.code == "E_ASSIGN_INOUT"));
    }

    #[test]
    fn rejects_simulating_inout_modules() {
        let mut design = Design::new();
        {
            let mut m = design.module("Top");
            m.inout("pad", uint(1));
        }

        let err = Simulator::new(&design, "Top").unwrap_err();
        assert!(err.diagnostics.iter().any(|d| d.code == "E_SIM_INOUT"));
    }

    #[test]
    fn rejects_duplicate_instance_port_connections() {
        let mut design = Design::new();
        {
            let mut m = design.module("Child");
            let a = m.input("a", 1);
            let y = m.output("y", 1);
            m.assign(y, a);
        }
        {
            let mut m = design.module("Top");
            let a = m.input("a", 1);
            let y = m.output("y", 1);
            m.instance("u_child", "Child", [("a", a), ("a", a), ("y", y)]);
        }

        let err = design.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_INSTANCE_PORT_DUP"));
    }

    #[test]
    fn rejects_hierarchy_cycles() {
        let mut design = Design::new();
        {
            let mut m = design.module("Loop");
            m.instance("self_ref", "Loop", std::iter::empty::<(&str, Signal)>());
        }

        let err = design.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_HIERARCHY_CYCLE"));
    }

    #[test]
    fn rejects_instance_mediated_combinational_cycles() {
        let mut design = Design::new();
        {
            let mut m = design.module("Child");
            let a = m.input("a", 1);
            let y = m.output("y", 1);
            m.assign(y, a);
        }
        {
            let mut m = design.module("Top");
            let y = m.output("y", 1);
            m.instance("u_child", "Child", [("a", y), ("y", y)]);
        }

        let err = design.validate().unwrap_err();
        assert!(err.diagnostics.iter().any(|d| d.code == "E_COMB_CYCLE"));
    }

    #[test]
    fn validates_explicit_width_helpers() {
        let mut design = Design::new();
        let (a, wide, narrow);
        {
            let mut m = design.module("Widths");
            a = m.input("a", 4);
            wide = m.output("wide", 8);
            narrow = m.output("narrow", 2);
            m.assign(wide, zext(a, 8));
            m.assign(narrow, trunc(a, 2));
        }

        design.validate().unwrap();
        let mut sim = Simulator::new(&design, "Widths").unwrap();
        sim.set(a, 0b1011);
        assert_eq!(sim.get(wide), 0b1011);
        assert_eq!(sim.get(narrow), 0b11);
    }

    #[test]
    fn simulates_signed_arithmetic_and_comparison() {
        let mut design = Design::new();
        let (a, b, sum, lt, wide);
        {
            let mut m = design.module("SignedAlu");
            a = m.input("a", sint(8));
            b = m.input("b", sint(8));
            sum = m.output("sum", sint(8));
            lt = m.output("lt", uint(1));
            wide = m.output("wide", sint(16));
            m.assign(sum, a + b);
            m.assign(lt, a.value().lt_expr(b));
            m.assign(wide, sext(a, 16));
        }

        design.validate().unwrap();
        let mut sim = Simulator::new(&design, "SignedAlu").unwrap();
        sim.set(a, 0xff);
        sim.set(b, 0x01);
        assert_eq!(sim.get(sum), 0x00);
        assert_eq!(sim.get(lt), 1);
        assert_eq!(sim.get(wide), 0xffff);
    }

    #[test]
    fn rejects_mixed_signed_unsigned_arithmetic() {
        let mut design = Design::new();
        {
            let mut m = design.module("Mixed");
            let a = m.input("a", sint(8));
            let b = m.input("b", uint(8));
            let y = m.output("y", sint(8));
            m.assign(y, a + b);
        }

        let err = design.validate().unwrap_err();
        assert!(err.diagnostics.iter().any(|d| d.code == "E_SIGNED_MIX"));
    }

    #[test]
    fn serializes_compiled_design_to_json_with_types() {
        let (design, ..) = counter_design();
        let json = design.compile().unwrap().to_json_pretty().unwrap();
        assert!(json.contains("\"ty\""));
        assert!(json.contains("\"signedness\""));
        assert!(json.contains("\"Unsigned\""));
    }

    #[test]
    fn constructs_and_looks_up_nested_bundle_fields() {
        let mut design = Design::new();
        let (valid, tag);
        {
            let mut m = design.module("BundleUser");
            let req = m.input_bundle("req", req_bundle_type());
            valid = req.field("valid").unwrap();
            tag = req.path(["meta", "tag"]).unwrap();
            assert!(req.field("missing").is_none());
            assert!(req.path(["meta", "missing"]).is_none());
        }

        let module = design.ir().find_module("BundleUser").unwrap();
        assert_eq!(module.signal(valid).unwrap().name, "req_valid");
        assert_eq!(module.signal(tag).unwrap().name, "req_meta_tag");
        assert_eq!(module.bundle_signals[0].fields.len(), 2);
        design.validate().unwrap();
    }

    #[test]
    fn expands_exact_bundle_instance_connections() {
        let mut design = Design::new();
        {
            let mut m = design.module("Child");
            let req = m.input_bundle("req", req_bundle_type());
            let resp = m.output_bundle("resp", resp_bundle_type());
            m.assign(resp.field("valid").unwrap(), req.field("valid").unwrap());
            m.assign(
                resp.field("data").unwrap(),
                zext(req.path(["meta", "tag"]).unwrap(), 8),
            );
        }

        let (req_valid, req_tag, resp_valid, resp_data);
        {
            let mut m = design.module("Top");
            let req = m.input_bundle("req", req_bundle_type());
            let resp = m.output_bundle("resp", resp_bundle_type());
            req_valid = req.field("valid").unwrap();
            req_tag = req.path(["meta", "tag"]).unwrap();
            resp_valid = resp.field("valid").unwrap();
            resp_data = resp.field("data").unwrap();
            m.instance_bundles("u_child", "Child", [("req", req), ("resp", resp)]);
        }

        let compiled = design.compile().unwrap();
        let top = compiled.find_module("Top").unwrap();
        let ports = top.instances[0]
            .connections
            .iter()
            .map(|connection| connection.port.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            ports,
            ["req_valid", "req_meta_tag", "resp_valid", "resp_data"]
        );

        let mut sim = Simulator::new(&design, "Top").unwrap();
        sim.set(req_valid, 1);
        sim.set(req_tag, 0xa);
        assert_eq!(sim.get(resp_valid), 1);
        assert_eq!(sim.get(resp_data), 0xa);
    }

    #[test]
    fn assigns_bundle_leaves() {
        let mut design = Design::new();
        let (req_valid, req_tag, resp_valid, resp_tag);
        {
            let mut m = design.module("BundleAssign");
            let req = m.input_bundle("req", req_bundle_type());
            let resp = m.output_bundle("resp", req_bundle_type());
            req_valid = req.field("valid").unwrap();
            req_tag = req.path(["meta", "tag"]).unwrap();
            resp_valid = resp.field("valid").unwrap();
            resp_tag = resp.path(["meta", "tag"]).unwrap();
            m.assign_bundle(&resp, &req);
        }

        let mut sim = Simulator::new(&design, "BundleAssign").unwrap();
        sim.set(req_valid, 1);
        sim.set(req_tag, 0xb);
        assert_eq!(sim.get(resp_valid), 1);
        assert_eq!(sim.get(resp_tag), 0xb);
    }

    #[test]
    fn conditionally_assigns_bundle_leaves() {
        let mut design = Design::new();
        let (enable, src_valid, src_tag, dst_valid, dst_tag);
        {
            let mut m = design.module("BundleAssignWhen");
            enable = m.input("enable", uint(1));
            let src = m.input_bundle("src", req_bundle_type());
            let dst = m.output_bundle("dst", req_bundle_type());
            src_valid = src.field("valid").unwrap();
            src_tag = src.path(["meta", "tag"]).unwrap();
            dst_valid = dst.field("valid").unwrap();
            dst_tag = dst.path(["meta", "tag"]).unwrap();
            m.assign_bundle_when(&dst, enable, &src);
        }

        let mut sim = Simulator::new(&design, "BundleAssignWhen").unwrap();
        sim.set(enable, 1);
        sim.set(src_valid, 1);
        sim.set(src_tag, 0x6);
        assert_eq!(sim.get(dst_valid), 1);
        assert_eq!(sim.get(dst_tag), 0x6);

        sim.set(enable, 0);
        sim.set(src_valid, 0);
        sim.set(src_tag, 0x2);
        assert_eq!(sim.get(dst_valid), 1);
        assert_eq!(sim.get(dst_tag), 0x6);
    }

    #[test]
    fn rejects_invalid_bundle_type_shapes() {
        let mut design = Design::new();
        {
            let mut m = design.module("BadBundles");
            let duplicate = bundle_type("Dup", [field("x", uint(1)), field("x", uint(2))]);
            let empty_names = bundle_type("", [field("", uint(0))]);
            m.input_bundle("dup", duplicate);
            m.input_bundle("empty", empty_names);
        }

        let err = design.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_BUNDLE_FIELD_DUP"));
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_BUNDLE_TYPE_NAME"));
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_BUNDLE_FIELD_NAME"));
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_BUNDLE_FIELD_WIDTH"));
    }

    #[test]
    fn rejects_bundle_instance_mismatch() {
        let small_req = bundle_type("SmallReq", [field("valid", uint(1))]);
        let wide_req = bundle_type(
            "Req",
            [
                field("valid", uint(1)),
                nested("meta", bundle_type("ReqMeta", [field("tag", uint(5))])),
            ],
        );

        let mut missing = Design::new();
        {
            let mut m = missing.module("Child");
            m.input_bundle("req", req_bundle_type());
        }
        {
            let mut m = missing.module("Top");
            let req = m.input_bundle("req", small_req.clone());
            m.instance_bundles("u_child", "Child", [("req", req)]);
        }
        let err = missing.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_BUNDLE_CONNECT_MISSING"));

        let mut extra = Design::new();
        {
            let mut m = extra.module("Child");
            m.input_bundle("req", small_req);
        }
        {
            let mut m = extra.module("Top");
            let req = m.input_bundle("req", req_bundle_type());
            m.instance_bundles("u_child", "Child", [("req", req)]);
        }
        let err = extra.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_BUNDLE_CONNECT_EXTRA"));

        let mut type_mismatch = Design::new();
        {
            let mut m = type_mismatch.module("Child");
            m.input_bundle("req", req_bundle_type());
        }
        {
            let mut m = type_mismatch.module("Top");
            let req = m.input_bundle("req", wide_req);
            m.instance_bundles("u_child", "Child", [("req", req)]);
        }
        let err = type_mismatch.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_BUNDLE_CONNECT_TYPE"));
    }

    #[test]
    fn rejects_bundle_assignment_mismatch() {
        let small_req = bundle_type("SmallReq", [field("valid", uint(1))]);
        let wide_req = bundle_type(
            "Req",
            [
                field("valid", uint(1)),
                nested("meta", bundle_type("ReqMeta", [field("tag", uint(5))])),
            ],
        );

        let mut missing = Design::new();
        {
            let mut m = missing.module("MissingBundleAssign");
            let dst = m.output_bundle("dst", req_bundle_type());
            let src = m.input_bundle("src", small_req.clone());
            m.assign_bundle(&dst, &src);
        }
        let err = missing.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_BUNDLE_ASSIGN_MISSING"));

        let mut extra = Design::new();
        {
            let mut m = extra.module("ExtraBundleAssign");
            let dst = m.output_bundle("dst", small_req);
            let src = m.input_bundle("src", req_bundle_type());
            m.assign_bundle(&dst, &src);
        }
        let err = extra.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_BUNDLE_ASSIGN_EXTRA"));

        let mut type_mismatch = Design::new();
        {
            let mut m = type_mismatch.module("TypeBundleAssign");
            let dst = m.output_bundle("dst", req_bundle_type());
            let src = m.input_bundle("src", wide_req);
            m.assign_bundle(&dst, &src);
        }
        let err = type_mismatch.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_BUNDLE_ASSIGN_TYPE"));
    }

    #[test]
    fn compiled_json_includes_bundle_metadata() {
        let mut design = Design::new();
        {
            let mut m = design.module("BusUser");
            m.input_bundle("req", req_bundle_type());
        }

        let json = design.compile().unwrap().to_json_pretty().unwrap();
        assert!(json.contains("\"bundle_types\""));
        assert!(json.contains("\"bundle_signals\""));
        assert!(json.contains("Req"));
        assert!(json.contains("req_meta_tag"));
    }

    #[test]
    fn constructs_and_looks_up_mixed_interface_ports() {
        let mut design = Design::new();
        let (req_valid, req_tag, resp_data, ready);
        {
            let mut m = design.module("InterfaceUser");
            let bus = m.interface("bus", bus_interface_type());
            let req = bus.port("req").unwrap();
            req_valid = bus.field("req", "valid").unwrap();
            req_tag = bus.path("req", ["meta", "tag"]).unwrap();
            resp_data = bus.field("resp", "data").unwrap();
            ready = bus.signal("ready").unwrap();
            assert!(bus.port("missing").is_none());
            assert!(bus.signal("missing").is_none());
            assert!(bus.signal("req").is_none());
            assert!(bus.bundle("ready").is_none());
            assert!(bus.field("ready", "valid").is_none());
            assert!(bus.path("req", ["meta", "missing"]).is_none());
            assert!(req.signal().is_none());
            assert_eq!(req.field("valid"), Some(req_valid));
            assert_eq!(req.path(["meta", "tag"]), Some(req_tag));
        }

        let module = design.ir().find_module("InterfaceUser").unwrap();
        assert_eq!(module.signal(req_valid).unwrap().name, "bus_req_valid");
        assert_eq!(module.signal(req_tag).unwrap().name, "bus_req_meta_tag");
        assert_eq!(module.signal(resp_data).unwrap().name, "bus_resp_data");
        assert_eq!(module.signal(ready).unwrap().name, "bus_ready");
        assert_eq!(module.interface_signals[0].ports.len(), 3);
        design.validate().unwrap();
    }

    #[test]
    fn expands_exact_interface_instance_connections() {
        let mut design = Design::new();
        {
            let mut m = design.module("Responder");
            let bus = m.interface("bus", bus_interface_type());
            let req = bus.port("req").unwrap().bundle().unwrap().clone();
            let resp = bus.port("resp").unwrap().bundle().unwrap().clone();
            m.assign(resp.field("valid").unwrap(), req.field("valid").unwrap());
            m.assign(
                resp.field("data").unwrap(),
                zext(req.path(["meta", "tag"]).unwrap(), 8),
            );
        }

        let (req_valid, req_tag, resp_valid, resp_data);
        {
            let mut m = design.module("Top");
            let bus = m.interface("bus", bus_interface_type());
            let req = bus.port("req").unwrap().bundle().unwrap().clone();
            let resp = bus.port("resp").unwrap().bundle().unwrap().clone();
            req_valid = req.field("valid").unwrap();
            req_tag = req.path(["meta", "tag"]).unwrap();
            resp_valid = resp.field("valid").unwrap();
            resp_data = resp.field("data").unwrap();
            m.instance_interfaces("u_responder", "Responder", [("bus", bus)]);
        }

        let compiled = design.compile().unwrap();
        let top = compiled.find_module("Top").unwrap();
        let ports = top.instances[0]
            .connections
            .iter()
            .map(|connection| connection.port.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            ports,
            [
                "bus_req_valid",
                "bus_req_meta_tag",
                "bus_resp_valid",
                "bus_resp_data",
                "bus_ready"
            ]
        );

        let mut sim = Simulator::new(&design, "Top").unwrap();
        sim.set(req_valid, 1);
        sim.set(req_tag, 0xc);
        assert_eq!(sim.get(resp_valid), 1);
        assert_eq!(sim.get(resp_data), 0xc);
    }

    #[test]
    fn rejects_invalid_interface_type_shapes() {
        let mut design = Design::new();
        {
            let mut m = design.module("BadInterfaces");
            let duplicate = interface_type(
                "Bad",
                [iface_input("x", uint(1)), iface_output("x", uint(1))],
            );
            let empty = interface_type("", [iface_input("", uint(0))]);
            m.interface("dup", duplicate);
            m.interface("empty", empty);
        }

        let err = design.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_INTERFACE_PORT_DUP"));
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_INTERFACE_TYPE_NAME"));
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_INTERFACE_PORT_NAME"));
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_INTERFACE_PORT_WIDTH"));
    }

    #[test]
    fn rejects_interface_instance_mismatches() {
        let small = interface_type("Small", [iface_input("req", req_bundle_type())]);
        let scalar_req = interface_type("ScalarReq", [iface_input("req", uint(1))]);
        let flipped = interface_type(
            "Flipped",
            [
                iface_output("req", req_bundle_type()),
                iface_output("resp", resp_bundle_type()),
                iface_input("ready", uint(1)),
            ],
        );
        let wide_req = interface_type(
            "WideReq",
            [
                iface_input(
                    "req",
                    bundle_type(
                        "Req",
                        [
                            field("valid", uint(1)),
                            nested("meta", bundle_type("ReqMeta", [field("tag", uint(5))])),
                        ],
                    ),
                ),
                iface_output("resp", resp_bundle_type()),
                iface_input("ready", uint(1)),
            ],
        );

        let mut missing = Design::new();
        {
            let mut m = missing.module("Child");
            m.interface("bus", bus_interface_type());
        }
        {
            let mut m = missing.module("Top");
            let bus = m.interface("bus", small.clone());
            m.instance_interfaces("u_child", "Child", [("bus", bus)]);
        }
        let err = missing.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_INTERFACE_CONNECT_MISSING"));

        let mut extra = Design::new();
        {
            let mut m = extra.module("Child");
            m.interface("bus", small);
        }
        {
            let mut m = extra.module("Top");
            let bus = m.interface("bus", bus_interface_type());
            m.instance_interfaces("u_child", "Child", [("bus", bus)]);
        }
        let err = extra.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_INTERFACE_CONNECT_EXTRA"));

        let mut direction = Design::new();
        {
            let mut m = direction.module("Child");
            m.interface("bus", bus_interface_type());
        }
        {
            let mut m = direction.module("Top");
            let bus = m.interface("bus", flipped);
            m.instance_interfaces("u_child", "Child", [("bus", bus)]);
        }
        let err = direction.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_INTERFACE_CONNECT_DIRECTION"));

        let mut kind = Design::new();
        {
            let mut m = kind.module("Child");
            m.interface("bus", bus_interface_type());
        }
        {
            let mut m = kind.module("Top");
            let bus = m.interface("bus", scalar_req);
            m.instance_interfaces("u_child", "Child", [("bus", bus)]);
        }
        let err = kind.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_INTERFACE_CONNECT_KIND"));

        let mut type_mismatch = Design::new();
        {
            let mut m = type_mismatch.module("Child");
            m.interface("bus", bus_interface_type());
        }
        {
            let mut m = type_mismatch.module("Top");
            let bus = m.interface("bus", wide_req);
            m.instance_interfaces("u_child", "Child", [("bus", bus)]);
        }
        let err = type_mismatch.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_INTERFACE_CONNECT_TYPE"));
    }

    #[test]
    fn compiled_json_includes_interface_metadata() {
        let mut design = Design::new();
        {
            let mut m = design.module("BusUser");
            m.interface("bus", bus_interface_type());
        }

        let json = design.compile().unwrap().to_json_pretty().unwrap();
        assert!(json.contains("\"interface_types\""));
        assert!(json.contains("\"interface_signals\""));
        assert!(json.contains("Bus"));
        assert!(json.contains("bus_req_meta_tag"));
    }

    #[test]
    fn constructs_scalar_ready_valid_source_and_fire() {
        let mut design = Design::new();
        let (valid, ready, bits, fire);
        {
            let mut m = design.module("StreamSource");
            let stream = m.rv_source("stream", rv_scalar(uint(8)));
            valid = stream.valid();
            ready = stream.ready();
            bits = stream.bits_signal().unwrap();
            fire = m.output("fire", uint(1));
            assert_eq!(stream.role(), ReadyValidRole::Source);
            assert_eq!(stream.payload(), &rv_scalar(uint(8)));
            assert!(stream.bits_bundle().is_none());
            m.assign(fire, stream.fire());
        }

        let module = design.ir().find_module("StreamSource").unwrap();
        assert_eq!(module.signal(valid).unwrap().name, "stream_valid");
        assert_eq!(module.signal(ready).unwrap().name, "stream_ready");
        assert_eq!(module.signal(bits).unwrap().name, "stream_bits");
        design.validate().unwrap();

        let mut sim = Simulator::new(&design, "StreamSource").unwrap();
        sim.set(valid, 1);
        sim.set(ready, 0);
        assert_eq!(sim.get(fire), 0);
        sim.set(ready, 1);
        assert_eq!(sim.get(fire), 1);
    }

    #[test]
    fn constructs_bundle_ready_valid_sink() {
        let mut design = Design::new();
        let (valid, ready, data, last);
        {
            let mut m = design.module("StreamSink");
            let stream = m.rv_sink("stream", rv_bundle(rv_payload_bundle_type()));
            let bits = stream.bits_bundle().unwrap();
            valid = stream.valid();
            ready = stream.ready();
            data = bits.field("data").unwrap();
            last = bits.field("last").unwrap();
            assert_eq!(stream.role(), ReadyValidRole::Sink);
            assert!(stream.bits_signal().is_none());
        }

        let module = design.ir().find_module("StreamSink").unwrap();
        assert_eq!(module.signal(valid).unwrap().name, "stream_valid");
        assert_eq!(module.signal(ready).unwrap().name, "stream_ready");
        assert_eq!(module.signal(data).unwrap().name, "stream_bits_data");
        assert_eq!(module.signal(last).unwrap().name, "stream_bits_last");
        design.validate().unwrap();
    }

    #[test]
    fn expands_ready_valid_interfaces_through_instances() {
        let mut design = Design::new();
        {
            let mut m = design.module("Consumer");
            let stream = m.rv_sink("stream", rv_scalar(uint(8)));
            m.assign(stream.ready(), stream.valid());
        }

        let (valid, ready, bits);
        {
            let mut m = design.module("Top");
            let stream = m.rv_sink("stream", rv_scalar(uint(8)));
            valid = stream.valid();
            ready = stream.ready();
            bits = stream.bits_signal().unwrap();
            m.instance_interfaces(
                "u_consumer",
                "Consumer",
                [("stream", stream.into_interface())],
            );
        }

        let compiled = design.compile().unwrap();
        let top = compiled.find_module("Top").unwrap();
        let ports = top.instances[0]
            .connections
            .iter()
            .map(|connection| connection.port.as_str())
            .collect::<Vec<_>>();
        assert_eq!(ports, ["stream_valid", "stream_ready", "stream_bits"]);

        let mut sim = Simulator::new(&design, "Top").unwrap();
        sim.set(valid, 1);
        sim.set(bits, 0xab);
        assert_eq!(sim.get(ready), 1);
    }

    #[test]
    fn simulates_scalar_ready_valid_connect() {
        let mut design = Design::new();
        let (in_valid, in_ready, in_bits, out_valid, out_ready, out_bits);
        {
            let mut m = design.module("ScalarConnect");
            let input = m.rv_sink("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_scalar(uint(8)));
            in_valid = input.valid();
            in_ready = input.ready();
            in_bits = input.bits_signal().unwrap();
            out_valid = output.valid();
            out_ready = output.ready();
            out_bits = output.bits_signal().unwrap();
            m.rv_connect(&input, &output);
        }

        design.validate().unwrap();
        let mut sim = Simulator::new(&design, "ScalarConnect").unwrap();
        sim.set(in_valid, 1);
        sim.set(in_bits, 0x5a);
        sim.set(out_ready, 0);
        assert_eq!(sim.get(out_valid), 1);
        assert_eq!(sim.get(out_bits), 0x5a);
        assert_eq!(sim.get(in_ready), 0);

        sim.set(out_ready, 1);
        assert_eq!(sim.get(in_ready), 1);
    }

    #[test]
    fn simulates_bundle_ready_valid_connect() {
        let mut design = Design::new();
        let (in_valid, in_ready, out_valid, out_ready, in_data, in_last, out_data, out_last);
        {
            let mut m = design.module("BundleConnect");
            let input = m.rv_sink("in", rv_bundle(rv_payload_bundle_type()));
            let output = m.rv_source("out", rv_bundle(rv_payload_bundle_type()));
            let input_bits = input.bits_bundle().unwrap();
            let output_bits = output.bits_bundle().unwrap();
            in_valid = input.valid();
            in_ready = input.ready();
            out_valid = output.valid();
            out_ready = output.ready();
            in_data = input_bits.field("data").unwrap();
            in_last = input_bits.field("last").unwrap();
            out_data = output_bits.field("data").unwrap();
            out_last = output_bits.field("last").unwrap();
            m.rv_connect(&input, &output);
        }

        design.validate().unwrap();
        let mut sim = Simulator::new(&design, "BundleConnect").unwrap();
        sim.set(in_valid, 1);
        sim.set(in_data, 0xa5);
        sim.set(in_last, 1);
        sim.set(out_ready, 1);
        assert_eq!(sim.get(out_valid), 1);
        assert_eq!(sim.get(out_data), 0xa5);
        assert_eq!(sim.get(out_last), 1);
        assert_eq!(sim.get(in_ready), 1);
    }

    #[test]
    fn rejects_ready_valid_connect_mismatches() {
        let mut wrong_role = Design::new();
        {
            let mut m = wrong_role.module("WrongRoleConnect");
            let input = m.rv_source("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_scalar(uint(8)));
            m.rv_connect(&input, &output);
        }
        let err = wrong_role.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_RV_CONNECT_ROLE"));

        let mut kind_mismatch = Design::new();
        {
            let mut m = kind_mismatch.module("KindConnect");
            let input = m.rv_sink("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_bundle(rv_payload_bundle_type()));
            m.rv_connect(&input, &output);
        }
        let err = kind_mismatch.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_RV_CONNECT_PAYLOAD_KIND"));

        let mut type_mismatch = Design::new();
        {
            let mut m = type_mismatch.module("TypeConnect");
            let input = m.rv_sink("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_scalar(uint(4)));
            m.rv_connect(&input, &output);
        }
        let err = type_mismatch.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_RV_CONNECT_PAYLOAD_TYPE"));
    }

    #[test]
    fn rejects_malformed_ready_valid_contracts() {
        let missing = interface_type(
            "MissingReady",
            [
                iface_output("valid", uint(1)),
                iface_output("bits", uint(8)),
            ],
        );
        let err = validate_ready_valid_type(&missing, ReadyValidRole::Source).unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_RV_PORT_MISSING"));

        let wrong_control = interface_type(
            "WrongValid",
            [
                iface_output("valid", uint(2)),
                iface_input("ready", uint(1)),
                iface_output("bits", uint(8)),
            ],
        );
        let err = validate_ready_valid_type(&wrong_control, ReadyValidRole::Source).unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_RV_CONTROL_TYPE"));

        let wrong_direction = rv_sink("WrongDirection", uint(8));
        let err = validate_ready_valid_type(&wrong_direction, ReadyValidRole::Source).unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_RV_PORT_DIRECTION"));

        let extra = interface_type(
            "Extra",
            [
                iface_output("valid", uint(1)),
                iface_input("ready", uint(1)),
                iface_output("bits", uint(8)),
                iface_output("sideband", uint(1)),
            ],
        );
        let err = validate_ready_valid_type(&extra, ReadyValidRole::Source).unwrap_err();
        assert!(err.diagnostics.iter().any(|d| d.code == "E_RV_PORT_EXTRA"));
    }

    #[test]
    fn simulates_scalar_ready_valid_register_slice() {
        let mut design = Design::new();
        let (rst, in_valid, in_ready, in_bits, out_valid, out_ready, out_bits);
        {
            let mut m = design.module("ScalarSlice");
            let clk = m.input("clk", uint(1));
            rst = m.input("rst", uint(1));
            let input = m.rv_sink("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_scalar(uint(8)));
            in_valid = input.valid();
            in_ready = input.ready();
            in_bits = input.bits_signal().unwrap();
            out_valid = output.valid();
            out_ready = output.ready();
            out_bits = output.bits_signal().unwrap();
            m.rv_register_slice("slice", &input, &output, clk, rst);
        }

        design.validate().unwrap();
        let mut sim = Simulator::new(&design, "ScalarSlice").unwrap();
        sim.set(out_ready, 1);
        assert_eq!(sim.get(in_ready), 1);

        sim.set(in_valid, 1);
        sim.set(in_bits, 0xaa);
        assert_eq!(sim.get(out_valid), 0);
        assert_eq!(sim.get(out_bits), 0);
        sim.tick();
        assert_eq!(sim.get(out_valid), 1);
        assert_eq!(sim.get(out_bits), 0xaa);

        sim.set(rst, 1);
        sim.tick();
        assert_eq!(sim.get(out_valid), 0);
        assert_eq!(sim.get(out_bits), 0);
    }

    #[test]
    fn simulates_bundle_ready_valid_register_slice() {
        let mut design = Design::new();
        let (in_valid, out_ready, out_valid, in_data, in_last, out_data, out_last);
        {
            let mut m = design.module("BundleSlice");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input = m.rv_sink("in", rv_bundle(rv_payload_bundle_type()));
            let output = m.rv_source("out", rv_bundle(rv_payload_bundle_type()));
            let input_bits = input.bits_bundle().unwrap();
            let output_bits = output.bits_bundle().unwrap();
            in_valid = input.valid();
            out_ready = output.ready();
            out_valid = output.valid();
            in_data = input_bits.field("data").unwrap();
            in_last = input_bits.field("last").unwrap();
            out_data = output_bits.field("data").unwrap();
            out_last = output_bits.field("last").unwrap();
            m.rv_register_slice("slice", &input, &output, clk, rst);
        }

        design.validate().unwrap();
        let mut sim = Simulator::new(&design, "BundleSlice").unwrap();
        sim.set(out_ready, 1);
        sim.set(in_valid, 1);
        sim.set(in_data, 0x5a);
        sim.set(in_last, 1);
        sim.tick();
        assert_eq!(sim.get(out_valid), 1);
        assert_eq!(sim.get(out_data), 0x5a);
        assert_eq!(sim.get(out_last), 1);
    }

    #[test]
    fn rejects_ready_valid_register_slice_mismatches() {
        let mut kind_mismatch = Design::new();
        {
            let mut m = kind_mismatch.module("KindMismatch");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input = m.rv_sink("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_bundle(rv_payload_bundle_type()));
            m.rv_register_slice("slice", &input, &output, clk, rst);
        }
        let err = kind_mismatch.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_RV_SLICE_PAYLOAD_KIND"));

        let mut type_mismatch = Design::new();
        {
            let mut m = type_mismatch.module("TypeMismatch");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input = m.rv_sink("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_scalar(uint(9)));
            m.rv_register_slice("slice", &input, &output, clk, rst);
        }
        let err = type_mismatch.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_RV_SLICE_PAYLOAD_TYPE"));

        let mut role_mismatch = Design::new();
        {
            let mut m = role_mismatch.module("RoleMismatch");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input = m.rv_source("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_scalar(uint(8)));
            m.rv_register_slice("slice", &input, &output, clk, rst);
        }
        let err = role_mismatch.validate().unwrap_err();
        assert!(err.diagnostics.iter().any(|d| d.code == "E_RV_SLICE_ROLE"));
    }

    #[test]
    fn simulates_scalar_ready_valid_skid_buffer() {
        let mut design = Design::new();
        let (rst, in_valid, in_ready, in_bits, out_valid, out_ready, out_bits);
        {
            let mut m = design.module("ScalarSkid");
            let clk = m.input("clk", uint(1));
            rst = m.input("rst", uint(1));
            let input = m.rv_sink("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_scalar(uint(8)));
            in_valid = input.valid();
            in_ready = input.ready();
            in_bits = input.bits_signal().unwrap();
            out_valid = output.valid();
            out_ready = output.ready();
            out_bits = output.bits_signal().unwrap();
            m.rv_skid_buffer("skid", &input, &output, clk, rst);
        }

        design.validate().unwrap();
        let full_q = design.find_signal("ScalarSkid", "skid_full_q").unwrap();
        let bits_q = design.find_signal("ScalarSkid", "skid_bits_q").unwrap();
        let mut sim = Simulator::new(&design, "ScalarSkid").unwrap();

        sim.set(out_ready, 1);
        sim.set(in_valid, 1);
        sim.set(in_bits, 0x11);
        assert_eq!(sim.get(in_ready), 1);
        assert_eq!(sim.get(out_valid), 1);
        assert_eq!(sim.get(out_bits), 0x11);

        sim.set(out_ready, 0);
        sim.set(in_bits, 0x22);
        assert_eq!(sim.get(in_ready), 1);
        assert_eq!(sim.get(out_valid), 1);
        assert_eq!(sim.get(out_bits), 0x22);
        sim.tick();
        assert_eq!(sim.get(full_q), 1);
        assert_eq!(sim.get(bits_q), 0x22);
        assert_eq!(sim.get(in_ready), 0);
        assert_eq!(sim.get(out_bits), 0x22);

        sim.set(in_bits, 0x33);
        sim.tick();
        assert_eq!(sim.get(bits_q), 0x22);
        assert_eq!(sim.get(out_bits), 0x22);

        sim.set(out_ready, 1);
        assert_eq!(sim.get(in_ready), 1);
        assert_eq!(sim.get(out_bits), 0x22);
        sim.tick();
        assert_eq!(sim.get(full_q), 1);
        assert_eq!(sim.get(bits_q), 0x33);
        assert_eq!(sim.get(out_bits), 0x33);

        sim.set(in_valid, 0);
        sim.tick();
        assert_eq!(sim.get(full_q), 0);
        assert_eq!(sim.get(out_valid), 0);

        sim.set(in_bits, 0);
        sim.set(out_ready, 0);
        sim.set(in_valid, 1);
        sim.set(in_bits, 0x44);
        sim.tick();
        assert_eq!(sim.get(full_q), 1);
        sim.set(in_valid, 0);
        sim.set(in_bits, 0);
        sim.set(rst, 1);
        sim.tick();
        assert_eq!(sim.get(full_q), 0);
        assert_eq!(sim.get(bits_q), 0);
        assert_eq!(sim.get(out_valid), 0);
        assert_eq!(sim.get(out_bits), 0);
    }

    #[test]
    fn simulates_bundle_ready_valid_skid_buffer() {
        let mut design = Design::new();
        let (in_valid, in_ready, out_ready, out_valid, in_data, in_last, out_data, out_last);
        {
            let mut m = design.module("BundleSkid");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input = m.rv_sink("in", rv_bundle(rv_payload_bundle_type()));
            let output = m.rv_source("out", rv_bundle(rv_payload_bundle_type()));
            let input_bits = input.bits_bundle().unwrap();
            let output_bits = output.bits_bundle().unwrap();
            in_valid = input.valid();
            in_ready = input.ready();
            out_ready = output.ready();
            out_valid = output.valid();
            in_data = input_bits.field("data").unwrap();
            in_last = input_bits.field("last").unwrap();
            out_data = output_bits.field("data").unwrap();
            out_last = output_bits.field("last").unwrap();
            m.rv_skid_buffer("skid", &input, &output, clk, rst);
        }

        design.validate().unwrap();
        let data_q = design
            .find_signal("BundleSkid", "skid_bits_data_q")
            .unwrap();
        let last_q = design
            .find_signal("BundleSkid", "skid_bits_last_q")
            .unwrap();
        let mut sim = Simulator::new(&design, "BundleSkid").unwrap();
        sim.set(out_ready, 0);
        sim.set(in_valid, 1);
        sim.set(in_data, 0x5a);
        sim.set(in_last, 1);
        assert_eq!(sim.get(in_ready), 1);
        assert_eq!(sim.get(out_valid), 1);
        assert_eq!(sim.get(out_data), 0x5a);
        assert_eq!(sim.get(out_last), 1);
        sim.tick();
        assert_eq!(sim.get(data_q), 0x5a);
        assert_eq!(sim.get(last_q), 1);

        sim.set(in_data, 0xa5);
        sim.set(in_last, 0);
        assert_eq!(sim.get(in_ready), 0);
        assert_eq!(sim.get(out_data), 0x5a);
        assert_eq!(sim.get(out_last), 1);
        sim.set(out_ready, 1);
        sim.tick();
        assert_eq!(sim.get(data_q), 0xa5);
        assert_eq!(sim.get(last_q), 0);
        assert_eq!(sim.get(out_data), 0xa5);
        assert_eq!(sim.get(out_last), 0);
    }

    #[test]
    fn rejects_ready_valid_skid_buffer_mismatches() {
        let mut kind_mismatch = Design::new();
        {
            let mut m = kind_mismatch.module("KindMismatch");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input = m.rv_sink("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_bundle(rv_payload_bundle_type()));
            m.rv_skid_buffer("skid", &input, &output, clk, rst);
        }
        let err = kind_mismatch.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_RV_SKID_PAYLOAD_KIND"));

        let mut type_mismatch = Design::new();
        {
            let mut m = type_mismatch.module("TypeMismatch");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input = m.rv_sink("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_scalar(sint(8)));
            m.rv_skid_buffer("skid", &input, &output, clk, rst);
        }
        let err = type_mismatch.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_RV_SKID_PAYLOAD_TYPE"));

        let mut role_mismatch = Design::new();
        {
            let mut m = role_mismatch.module("RoleMismatch");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input = m.rv_source("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_scalar(uint(8)));
            m.rv_skid_buffer("skid", &input, &output, clk, rst);
        }
        let err = role_mismatch.validate().unwrap_err();
        assert!(err.diagnostics.iter().any(|d| d.code == "E_RV_SKID_ROLE"));

        let mut field_mismatch = Design::new();
        {
            let mut m = field_mismatch.module("FieldMismatch");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input_ty = bundle_type(
                "InputPayload",
                [field("data", uint(8)), field("last", uint(1))],
            );
            let output_ty = bundle_type(
                "OutputPayload",
                [field("data", uint(9)), field("keep", uint(1))],
            );
            let input = m.rv_sink("in", rv_bundle(input_ty));
            let output = m.rv_source("out", rv_bundle(output_ty));
            m.rv_skid_buffer("skid", &input, &output, clk, rst);
        }
        let err = field_mismatch.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_RV_SKID_PAYLOAD_TYPE"));
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_RV_SKID_FIELD_MISSING"));
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_RV_SKID_FIELD_EXTRA"));
    }

    #[test]
    fn simulates_scalar_ready_valid_fifo() {
        let mut design = Design::new();
        let (rst, in_valid, in_ready, in_bits, out_valid, out_ready, out_bits);
        {
            let mut m = design.module("ScalarFifo");
            let clk = m.input("clk", uint(1));
            rst = m.input("rst", uint(1));
            let input = m.rv_sink("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_scalar(uint(8)));
            in_valid = input.valid();
            in_ready = input.ready();
            in_bits = input.bits_signal().unwrap();
            out_valid = output.valid();
            out_ready = output.ready();
            out_bits = output.bits_signal().unwrap();
            m.rv_fifo("fifo", &input, &output, clk, rst, 3);
        }

        design.validate().unwrap();
        let count_q = design.find_signal("ScalarFifo", "fifo_count_q").unwrap();
        let mut sim = Simulator::new(&design, "ScalarFifo").unwrap();
        assert_eq!(sim.get(in_ready), 1);
        assert_eq!(sim.get(out_valid), 0);

        sim.set(out_ready, 0);
        sim.set(in_valid, 1);
        sim.set(in_bits, 0x10);
        sim.tick();
        assert_eq!(sim.get(count_q), 1);
        assert_eq!(sim.get(out_valid), 1);
        assert_eq!(sim.get(out_bits), 0x10);

        sim.set(in_bits, 0x20);
        sim.tick();
        sim.set(in_bits, 0x30);
        sim.tick();
        assert_eq!(sim.get(count_q), 3);
        assert_eq!(sim.get(in_ready), 0);
        assert_eq!(sim.get(out_bits), 0x10);

        sim.set(in_bits, 0x40);
        sim.tick();
        assert_eq!(sim.get(count_q), 3);
        assert_eq!(sim.get(out_bits), 0x10);

        sim.set(out_ready, 1);
        assert_eq!(sim.get(in_ready), 0);
        sim.tick();
        assert_eq!(sim.get(count_q), 2);
        assert_eq!(sim.get(in_ready), 1);
        assert_eq!(sim.get(out_bits), 0x20);

        sim.set(in_bits, 0x40);
        sim.tick();
        assert_eq!(sim.get(count_q), 2);
        assert_eq!(sim.get(out_bits), 0x30);

        sim.set(in_valid, 0);
        sim.tick();
        assert_eq!(sim.get(out_bits), 0x40);
        sim.tick();
        assert_eq!(sim.get(count_q), 0);
        assert_eq!(sim.get(out_valid), 0);

        sim.set(out_ready, 0);
        sim.set(in_valid, 1);
        sim.set(in_bits, 0x55);
        sim.tick();
        assert_eq!(sim.get(count_q), 1);
        sim.set(rst, 1);
        sim.tick();
        assert_eq!(sim.get(count_q), 0);
        assert_eq!(sim.get(out_valid), 0);
        assert_eq!(sim.get(in_ready), 1);
    }

    #[test]
    fn simulates_bundle_ready_valid_fifo() {
        let mut design = Design::new();
        let (in_valid, in_ready, out_ready, out_valid, in_data, in_last, out_data, out_last);
        {
            let mut m = design.module("BundleFifo");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input = m.rv_sink("in", rv_bundle(rv_payload_bundle_type()));
            let output = m.rv_source("out", rv_bundle(rv_payload_bundle_type()));
            let input_bits = input.bits_bundle().unwrap();
            let output_bits = output.bits_bundle().unwrap();
            in_valid = input.valid();
            in_ready = input.ready();
            out_ready = output.ready();
            out_valid = output.valid();
            in_data = input_bits.field("data").unwrap();
            in_last = input_bits.field("last").unwrap();
            out_data = output_bits.field("data").unwrap();
            out_last = output_bits.field("last").unwrap();
            m.rv_fifo("fifo", &input, &output, clk, rst, 2);
        }

        design.validate().unwrap();
        let mut sim = Simulator::new(&design, "BundleFifo").unwrap();
        sim.set(out_ready, 0);
        sim.set(in_valid, 1);
        sim.set(in_data, 0xaa);
        sim.set(in_last, 0);
        sim.tick();
        assert_eq!(sim.get(out_valid), 1);
        assert_eq!(sim.get(out_data), 0xaa);
        assert_eq!(sim.get(out_last), 0);

        sim.set(in_data, 0xbb);
        sim.set(in_last, 1);
        sim.tick();
        assert_eq!(sim.get(in_ready), 0);
        assert_eq!(sim.get(out_data), 0xaa);
        assert_eq!(sim.get(out_last), 0);

        sim.set(out_ready, 1);
        sim.tick();
        assert_eq!(sim.get(in_ready), 1);
        assert_eq!(sim.get(out_data), 0xbb);
        assert_eq!(sim.get(out_last), 1);
    }

    #[test]
    fn rejects_ready_valid_fifo_mismatches() {
        let mut bad_depth = Design::new();
        {
            let mut m = bad_depth.module("BadDepth");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input = m.rv_sink("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_scalar(uint(8)));
            m.rv_fifo("fifo", &input, &output, clk, rst, 1);
        }
        let err = bad_depth.validate().unwrap_err();
        assert!(err.diagnostics.iter().any(|d| d.code == "E_RV_FIFO_DEPTH"));

        let mut kind_mismatch = Design::new();
        {
            let mut m = kind_mismatch.module("KindMismatch");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input = m.rv_sink("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_bundle(rv_payload_bundle_type()));
            m.rv_fifo("fifo", &input, &output, clk, rst, 2);
        }
        let err = kind_mismatch.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_RV_FIFO_PAYLOAD_KIND"));

        let mut type_mismatch = Design::new();
        {
            let mut m = type_mismatch.module("TypeMismatch");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input = m.rv_sink("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_scalar(sint(8)));
            m.rv_fifo("fifo", &input, &output, clk, rst, 2);
        }
        let err = type_mismatch.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_RV_FIFO_PAYLOAD_TYPE"));

        let mut role_mismatch = Design::new();
        {
            let mut m = role_mismatch.module("RoleMismatch");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input = m.rv_source("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_scalar(uint(8)));
            m.rv_fifo("fifo", &input, &output, clk, rst, 2);
        }
        let err = role_mismatch.validate().unwrap_err();
        assert!(err.diagnostics.iter().any(|d| d.code == "E_RV_FIFO_ROLE"));

        let mut field_mismatch = Design::new();
        {
            let mut m = field_mismatch.module("FieldMismatch");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input_ty = bundle_type(
                "InputPayload",
                [field("data", uint(8)), field("last", uint(1))],
            );
            let output_ty = bundle_type(
                "OutputPayload",
                [field("data", uint(9)), field("keep", uint(1))],
            );
            let input = m.rv_sink("in", rv_bundle(input_ty));
            let output = m.rv_source("out", rv_bundle(output_ty));
            m.rv_fifo("fifo", &input, &output, clk, rst, 2);
        }
        let err = field_mismatch.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_RV_FIFO_PAYLOAD_TYPE"));
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_RV_FIFO_FIELD_MISSING"));
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_RV_FIFO_FIELD_EXTRA"));
    }

    #[test]
    fn simulates_scalar_ready_valid_mem_fifo() {
        let mut design = Design::new();
        let (rst, in_valid, in_ready, in_bits, out_valid, out_ready, out_bits);
        {
            let mut m = design.module("ScalarMemFifo");
            let clk = m.input("clk", uint(1));
            rst = m.input("rst", uint(1));
            let input = m.rv_sink("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_scalar(uint(8)));
            in_valid = input.valid();
            in_ready = input.ready();
            in_bits = input.bits_signal().unwrap();
            out_valid = output.valid();
            out_ready = output.ready();
            out_bits = output.bits_signal().unwrap();
            m.rv_mem_fifo("fifo", &input, &output, clk, rst, 3);
        }

        design.validate().unwrap();
        let count_q = design.find_signal("ScalarMemFifo", "fifo_count_q").unwrap();
        let mem = design.find_signal("ScalarMemFifo", "fifo_mem").unwrap();
        let mut sim = Simulator::new(&design, "ScalarMemFifo").unwrap();
        assert_eq!(sim.get(in_ready), 1);
        assert_eq!(sim.get(out_valid), 0);

        sim.set(out_ready, 0);
        sim.set(in_valid, 1);
        sim.set(in_bits, 0x10);
        sim.tick();
        assert_eq!(sim.peek_mem(mem, 0), Some(0x10));
        assert_eq!(sim.get(count_q), 1);
        assert_eq!(sim.get(out_valid), 1);
        assert_eq!(sim.get(out_bits), 0x10);

        sim.set(in_bits, 0x20);
        sim.tick();
        sim.set(in_bits, 0x30);
        sim.tick();
        assert_eq!(sim.get(count_q), 3);
        assert_eq!(sim.get(in_ready), 0);
        assert_eq!(sim.get(out_bits), 0x10);

        sim.set(in_bits, 0x40);
        sim.tick();
        assert_eq!(sim.get(count_q), 3);
        assert_eq!(sim.get(out_bits), 0x10);

        sim.set(out_ready, 1);
        assert_eq!(sim.get(in_ready), 0);
        sim.tick();
        assert_eq!(sim.get(count_q), 2);
        assert_eq!(sim.get(in_ready), 1);
        assert_eq!(sim.get(out_bits), 0x20);

        sim.set(in_bits, 0x40);
        sim.tick();
        assert_eq!(sim.get(count_q), 2);
        assert_eq!(sim.get(out_bits), 0x30);

        sim.set(in_valid, 0);
        sim.tick();
        assert_eq!(sim.get(out_bits), 0x40);
        sim.tick();
        assert_eq!(sim.get(count_q), 0);
        assert_eq!(sim.get(out_valid), 0);

        sim.set(out_ready, 0);
        sim.set(in_valid, 1);
        sim.set(in_bits, 0x55);
        sim.tick();
        assert_eq!(sim.get(count_q), 1);
        sim.set(rst, 1);
        sim.tick();
        assert_eq!(sim.get(count_q), 0);
        assert_eq!(sim.get(out_valid), 0);
        assert_eq!(sim.get(in_ready), 1);
    }

    #[test]
    fn simulates_bundle_ready_valid_mem_fifo() {
        let mut design = Design::new();
        let (in_valid, in_ready, out_ready, out_valid, in_data, in_last, out_data, out_last);
        {
            let mut m = design.module("BundleMemFifo");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input = m.rv_sink("in", rv_bundle(rv_payload_bundle_type()));
            let output = m.rv_source("out", rv_bundle(rv_payload_bundle_type()));
            let input_bits = input.bits_bundle().unwrap();
            let output_bits = output.bits_bundle().unwrap();
            in_valid = input.valid();
            in_ready = input.ready();
            out_ready = output.ready();
            out_valid = output.valid();
            in_data = input_bits.field("data").unwrap();
            in_last = input_bits.field("last").unwrap();
            out_data = output_bits.field("data").unwrap();
            out_last = output_bits.field("last").unwrap();
            m.rv_mem_fifo("fifo", &input, &output, clk, rst, 2);
        }

        design.validate().unwrap();
        let mem = design.find_signal("BundleMemFifo", "fifo_mem").unwrap();
        let mut sim = Simulator::new(&design, "BundleMemFifo").unwrap();
        sim.set(out_ready, 0);
        sim.set(in_valid, 1);
        sim.set(in_data, 0xaa);
        sim.set(in_last, 0);
        sim.tick();
        assert_eq!(sim.peek_mem(mem, 0), Some(0x154));
        assert_eq!(sim.get(out_valid), 1);
        assert_eq!(sim.get(out_data), 0xaa);
        assert_eq!(sim.get(out_last), 0);

        sim.set(in_data, 0xbb);
        sim.set(in_last, 1);
        sim.tick();
        assert_eq!(sim.peek_mem(mem, 1), Some(0x177));
        assert_eq!(sim.get(in_ready), 0);
        assert_eq!(sim.get(out_data), 0xaa);
        assert_eq!(sim.get(out_last), 0);

        sim.set(out_ready, 1);
        sim.tick();
        assert_eq!(sim.get(in_ready), 1);
        assert_eq!(sim.get(out_data), 0xbb);
        assert_eq!(sim.get(out_last), 1);
    }

    #[test]
    fn rejects_ready_valid_mem_fifo_mismatches() {
        let mut bad_depth = Design::new();
        {
            let mut m = bad_depth.module("BadDepth");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input = m.rv_sink("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_scalar(uint(8)));
            m.rv_mem_fifo("fifo", &input, &output, clk, rst, 1);
        }
        let err = bad_depth.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_RV_MEM_FIFO_DEPTH"));

        let mut kind_mismatch = Design::new();
        {
            let mut m = kind_mismatch.module("KindMismatch");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input = m.rv_sink("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_bundle(rv_payload_bundle_type()));
            m.rv_mem_fifo("fifo", &input, &output, clk, rst, 2);
        }
        let err = kind_mismatch.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_RV_MEM_FIFO_PAYLOAD_KIND"));

        let mut type_mismatch = Design::new();
        {
            let mut m = type_mismatch.module("TypeMismatch");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input = m.rv_sink("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_scalar(sint(8)));
            m.rv_mem_fifo("fifo", &input, &output, clk, rst, 2);
        }
        let err = type_mismatch.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_RV_MEM_FIFO_PAYLOAD_TYPE"));

        let mut role_mismatch = Design::new();
        {
            let mut m = role_mismatch.module("RoleMismatch");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input = m.rv_source("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_scalar(uint(8)));
            m.rv_mem_fifo("fifo", &input, &output, clk, rst, 2);
        }
        let err = role_mismatch.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_RV_MEM_FIFO_ROLE"));

        let mut field_mismatch = Design::new();
        {
            let mut m = field_mismatch.module("FieldMismatch");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input_ty = bundle_type(
                "InputPayload",
                [field("data", uint(8)), field("last", uint(1))],
            );
            let output_ty = bundle_type(
                "OutputPayload",
                [field("data", uint(9)), field("keep", uint(1))],
            );
            let input = m.rv_sink("in", rv_bundle(input_ty));
            let output = m.rv_source("out", rv_bundle(output_ty));
            m.rv_mem_fifo("fifo", &input, &output, clk, rst, 2);
        }
        let err = field_mismatch.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_RV_MEM_FIFO_PAYLOAD_TYPE"));
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_RV_MEM_FIFO_FIELD_MISSING"));
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_RV_MEM_FIFO_FIELD_EXTRA"));
    }

    #[test]
    fn validates_state_type_duplicates_and_ranges() {
        let mut design = Design::new();
        {
            let mut m = design.module("BadStates");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let states = state_type(
                "BadState",
                uint(1),
                [("Idle", 0), ("Idle", 1), ("TooBig", 2)],
            );
            let state = m.state_reg("state", states, clk, rst, "Idle");
            m.state_next_hold(&state, std::iter::empty::<(Expr, String)>());
        }

        let err = design.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_STATE_VARIANT_DUP"));
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_STATE_ENCODING_RANGE"));
    }

    #[test]
    fn simulates_async_reset_state_helper() {
        let mut design = Design::new();
        let (rst, start, busy);
        {
            let mut m = design.module("AsyncController");
            let clk = m.input("clk", uint(1));
            rst = m.input("rst", uint(1));
            start = m.input("start", uint(1));
            busy = m.output("busy", uint(1));
            let states = state_type("ControllerState", uint(1), [("Idle", 0), ("Run", 1)]);
            let state = m.state_reg_async_reset("state", states, clk, rst, "Idle");
            m.state_next_hold(&state, [(start.value(), "Run")]);
            m.assign(busy, state.is("Run"));
        }

        let mut sim = Simulator::new(&design, "AsyncController").unwrap();
        sim.set(start, 1);
        sim.tick();
        assert_eq!(sim.get(busy), 1);

        sim.set(rst, 1);
        assert_eq!(sim.get(busy), 0);
    }

    #[test]
    fn simulates_active_low_sync_reset() {
        let mut design = Design::new();
        let (rst_n, en, out);
        {
            let mut m = design.module("ActiveLowSyncCounter");
            let clk = m.input("clk", uint(1));
            rst_n = m.input("rst_n", uint(1));
            en = m.input("en", uint(1));
            out = m.output("out", uint(4));
            let count = m.reg("count", uint(4));
            m.clock(count, clk);
            m.reset_low(count, rst_n, 0);
            m.next(count, mux(en, count + lit_u(1, 4), count));
            m.assign(out, count);
        }

        let mut sim = Simulator::new(&design, "ActiveLowSyncCounter").unwrap();
        sim.set(rst_n, 1);
        sim.set(en, 1);
        sim.tick();
        assert_eq!(sim.get(out), 1);

        sim.set(rst_n, 0);
        assert_eq!(sim.get(out), 1);
        sim.tick();
        assert_eq!(sim.get(out), 0);
    }

    #[test]
    fn simulates_active_low_async_reset_immediately() {
        let mut design = Design::new();
        let (rst_n, en, out);
        {
            let mut m = design.module("ActiveLowAsyncCounter");
            let clk = m.input("clk", uint(1));
            rst_n = m.input("rst_n", uint(1));
            en = m.input("en", uint(1));
            out = m.output("out", uint(4));
            let count = m.reg("count", uint(4));
            m.clock(count, clk);
            m.async_reset_low(count, rst_n, 0);
            m.next(count, mux(en, count + lit_u(1, 4), count));
            m.assign(out, count);
        }

        let mut sim = Simulator::new(&design, "ActiveLowAsyncCounter").unwrap();
        sim.set(rst_n, 1);
        sim.set(en, 1);
        sim.tick();
        assert_eq!(sim.get(out), 1);

        sim.set(rst_n, 0);
        assert_eq!(sim.get(out), 0);
    }

    #[test]
    fn simulates_active_low_state_helpers() {
        let mut design = Design::new();
        let (sync_rst_n, sync_start, sync_busy, async_rst_n, async_start, async_busy);
        {
            let mut m = design.module("ActiveLowStateHelpers");
            let clk = m.input("clk", uint(1));
            sync_rst_n = m.input("sync_rst_n", uint(1));
            sync_start = m.input("sync_start", uint(1));
            sync_busy = m.output("sync_busy", uint(1));
            async_rst_n = m.input("async_rst_n", uint(1));
            async_start = m.input("async_start", uint(1));
            async_busy = m.output("async_busy", uint(1));
            let states = state_type("ControllerState", uint(1), [("Idle", 0), ("Run", 1)]);
            let sync_state =
                m.state_reg_reset_low("sync_state", states.clone(), clk, sync_rst_n, "Idle");
            let async_state =
                m.state_reg_async_reset_low("async_state", states, clk, async_rst_n, "Idle");
            m.state_next_hold(&sync_state, [(sync_start.value(), "Run")]);
            m.state_next_hold(&async_state, [(async_start.value(), "Run")]);
            m.assign(sync_busy, sync_state.is("Run"));
            m.assign(async_busy, async_state.is("Run"));
        }

        let mut sim = Simulator::new(&design, "ActiveLowStateHelpers").unwrap();
        sim.set(sync_rst_n, 1);
        sim.set(async_rst_n, 1);
        sim.set(sync_start, 1);
        sim.set(async_start, 1);
        sim.tick();
        assert_eq!(sim.get(sync_busy), 1);
        assert_eq!(sim.get(async_busy), 1);

        sim.set(sync_rst_n, 0);
        assert_eq!(sim.get(sync_busy), 1);
        sim.tick();
        assert_eq!(sim.get(sync_busy), 0);

        sim.set(async_rst_n, 0);
        assert_eq!(sim.get(async_busy), 0);
    }

    #[test]
    fn validates_async_reset_width_and_value() {
        let mut design = Design::new();
        {
            let mut m = design.module("BadAsyncReset");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(2));
            let reg = m.reg("reg", uint(1));
            m.clock(reg, clk);
            m.async_reset(reg, rst, 2);
            m.next(reg, lit(0, 1));
        }

        let err = design.validate().unwrap_err();
        assert!(err.diagnostics.iter().any(|d| d.code == "E_RESET_WIDTH"));
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_RESET_VALUE_WIDTH"));
    }

    #[test]
    fn compiled_json_includes_async_reset_kind() {
        let mut design = Design::new();
        {
            let mut m = design.module("AsyncJson");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let reg = m.reg("reg", uint(1));
            m.clock(reg, clk);
            m.async_reset(reg, rst, 0);
            m.next(reg, lit(0, 1));
        }

        let json = design.compile().unwrap().to_json_pretty().unwrap();
        assert!(json.contains("\"kind\""));
        assert!(json.contains("\"Async\""));
    }

    #[test]
    fn compiled_json_includes_reset_polarity() {
        let mut design = Design::new();
        {
            let mut m = design.module("ResetPolarityJson");
            let clk = m.input("clk", uint(1));
            let rst_n = m.input("rst_n", uint(1));
            let reg = m.reg("reg", uint(1));
            m.clock(reg, clk);
            m.async_reset_low(reg, rst_n, 0);
            m.next(reg, lit(0, 1));
        }

        let json = design.compile().unwrap().to_json_pretty().unwrap();
        assert!(json.contains("\"polarity\""));
        assert!(json.contains("\"ActiveLow\""));
    }

    #[test]
    fn reports_reset_kind_changes() {
        let mut design = Design::new();
        {
            let mut m = design.module("BadResetKind");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let reg = m.reg("reg", uint(1));
            m.clock(reg, clk);
            m.reset(reg, rst, 0);
            m.async_reset(reg, rst, 0);
            m.next(reg, lit(0, 1));
        }

        let err = design.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_BUILDER_RESET_KIND"));
    }

    #[test]
    fn reports_reset_polarity_changes() {
        let mut design = Design::new();
        {
            let mut m = design.module("BadResetPolarity");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let reg = m.reg("reg", uint(1));
            m.clock(reg, clk);
            m.reset(reg, rst, 0);
            m.reset_low(reg, rst, 0);
            m.next(reg, lit(0, 1));
        }

        let err = design.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_BUILDER_RESET_KIND"));
    }

    #[test]
    fn rejects_state_systemverilog_name_collisions() {
        let mut type_collision = Design::new();
        {
            let mut m = type_collision.module("BadStateTypes");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let first = state_type("Flow-State", uint(1), [("Idle", 0), ("Run", 1)]);
            let second = state_type("Flow_State", uint(1), [("Idle", 0), ("Run", 1)]);
            let s1 = m.state_reg("s1", first, clk, rst, "Idle");
            let s2 = m.state_reg("s2", second, clk, rst, "Idle");
            m.state_next_hold(&s1, std::iter::empty::<(Expr, String)>());
            m.state_next_hold(&s2, std::iter::empty::<(Expr, String)>());
        }
        let err = type_collision.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_STATE_TYPE_SV_NAME_DUP"));

        let mut variant_collision = Design::new();
        {
            let mut m = variant_collision.module("BadStateVariants");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let states = state_type("FlowState", uint(1), [("Run-1", 0), ("Run_1", 1)]);
            let state = m.state_reg("state", states, clk, rst, "Run-1");
            m.state_next_hold(&state, std::iter::empty::<(Expr, String)>());
        }
        let err = variant_collision.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_STATE_VARIANT_SV_NAME_DUP"));
    }

    #[test]
    fn simulates_state_helper_transitions() {
        let mut design = Design::new();
        let (start, done, busy);
        {
            let mut m = design.module("Controller");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            start = m.input("start", uint(1));
            done = m.input("done", uint(1));
            busy = m.output("busy", uint(1));
            let states = state_type(
                "ControllerState",
                uint(2),
                [("Idle", 0), ("Run", 1), ("Done", 2)],
            );
            let state = m.state_reg("state", states, clk, rst, "Idle");
            m.state_next_hold(&state, [(start.value(), "Run"), (done.value(), "Done")]);
            m.assign(busy, state.is("Run"));
        }

        design.validate().unwrap();
        let mut sim = Simulator::new(&design, "Controller").unwrap();
        sim.set(start, 1);
        sim.tick();
        assert_eq!(sim.get(busy), 1);
        sim.set(start, 0);
        sim.set(done, 1);
        sim.tick();
        assert_eq!(sim.get(busy), 0);
    }

    #[test]
    fn compiled_json_includes_state_metadata() {
        let mut design = Design::new();
        {
            let mut m = design.module("Controller");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let states = state_type("ControllerState", uint(2), [("Idle", 0), ("Run", 1)]);
            let state = m.state_reg("state", states, clk, rst, "Idle");
            m.state_next_hold(&state, std::iter::empty::<(Expr, String)>());
        }

        let json = design.compile().unwrap().to_json_pretty().unwrap();
        assert!(json.contains("\"state_types\""));
        assert!(json.contains("ControllerState"));
        assert!(json.contains("\"state_signals\""));
    }

    #[test]
    fn reports_invalid_register_builder_calls() {
        let mut design = Design::new();
        {
            let mut m = design.module("BadBuilder");
            let clk = m.input("clk", 1);
            let y = m.output("y", 1);
            m.clock(y, clk);
            m.assign(y, lit(0, 1));
        }

        let err = design.validate().unwrap_err();
        assert!(err
            .diagnostics
            .iter()
            .any(|d| d.code == "E_BUILDER_REG_KIND"));
    }
}
