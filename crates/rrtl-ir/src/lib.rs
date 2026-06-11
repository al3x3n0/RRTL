use std::fmt;
use std::ops::{Add, BitAnd, BitOr, BitXor, Mul, Not, Sub};

use serde::{Deserialize, Serialize};

pub type Width = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Signedness {
    Unsigned,
    Signed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BitType {
    pub width: Width,
    pub signedness: Signedness,
}

impl BitType {
    pub fn new(width: Width, signedness: Signedness) -> Self {
        Self { width, signedness }
    }

    pub fn is_signed(self) -> bool {
        self.signedness == Signedness::Signed
    }
}

impl From<Width> for BitType {
    fn from(width: Width) -> Self {
        uint(width)
    }
}

pub fn uint(width: Width) -> BitType {
    BitType::new(width, Signedness::Unsigned)
}

pub fn sint(width: Width) -> BitType {
    BitType::new(width, Signedness::Signed)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateType {
    pub name: String,
    pub ty: BitType,
    pub variants: Vec<StateVariant>,
}

impl StateType {
    pub fn lit(&self, name: &str) -> Option<Expr> {
        self.value_of(name)
            .map(|value| Expr::Lit { value, ty: self.ty })
    }

    pub fn value_of(&self, name: &str) -> Option<i128> {
        self.variants
            .iter()
            .find(|variant| variant.name == name)
            .map(|variant| variant.value)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateVariant {
    pub name: String,
    pub value: i128,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleType {
    pub name: String,
    pub fields: Vec<BundleField>,
}

impl BundleType {
    pub fn leaf_fields(&self) -> Vec<BundleLeaf> {
        let mut leaves = Vec::new();
        collect_bundle_leaves(self, &mut Vec::new(), &mut leaves);
        leaves
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BundleField {
    Leaf { name: String, ty: BitType },
    Nested { name: String, bundle: BundleType },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleLeaf {
    pub path: Vec<String>,
    pub ty: BitType,
}

fn collect_bundle_leaves(
    bundle: &BundleType,
    prefix: &mut Vec<String>,
    leaves: &mut Vec<BundleLeaf>,
) {
    for field in &bundle.fields {
        match field {
            BundleField::Leaf { name, ty } => {
                let mut path = prefix.clone();
                path.push(name.clone());
                leaves.push(BundleLeaf { path, ty: *ty });
            }
            BundleField::Nested { name, bundle } => {
                prefix.push(name.clone());
                collect_bundle_leaves(bundle, prefix, leaves);
                prefix.pop();
            }
        }
    }
}

pub fn bundle_type(
    name: impl Into<String>,
    fields: impl IntoIterator<Item = BundleField>,
) -> BundleType {
    BundleType {
        name: name.into(),
        fields: fields.into_iter().collect(),
    }
}

pub fn field(name: impl Into<String>, ty: BitType) -> BundleField {
    BundleField::Leaf {
        name: name.into(),
        ty,
    }
}

pub fn nested(name: impl Into<String>, bundle: BundleType) -> BundleField {
    BundleField::Nested {
        name: name.into(),
        bundle,
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterfaceType {
    pub name: String,
    pub ports: Vec<InterfacePort>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterfacePort {
    pub name: String,
    pub direction: InterfaceDirection,
    pub ty: InterfacePortType,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterfaceDirection {
    Input,
    Output,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterfacePortType {
    Scalar(BitType),
    Bundle(BundleType),
}

impl From<BitType> for InterfacePortType {
    fn from(value: BitType) -> Self {
        Self::Scalar(value)
    }
}

impl From<Width> for InterfacePortType {
    fn from(value: Width) -> Self {
        Self::Scalar(uint(value))
    }
}

impl From<BundleType> for InterfacePortType {
    fn from(value: BundleType) -> Self {
        Self::Bundle(value)
    }
}

pub fn interface_type(
    name: impl Into<String>,
    ports: impl IntoIterator<Item = InterfacePort>,
) -> InterfaceType {
    InterfaceType {
        name: name.into(),
        ports: ports.into_iter().collect(),
    }
}

pub fn iface_input(name: impl Into<String>, ty: impl Into<InterfacePortType>) -> InterfacePort {
    InterfacePort {
        name: name.into(),
        direction: InterfaceDirection::Input,
        ty: ty.into(),
    }
}

pub fn iface_output(name: impl Into<String>, ty: impl Into<InterfacePortType>) -> InterfacePort {
    InterfacePort {
        name: name.into(),
        direction: InterfaceDirection::Output,
        ty: ty.into(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModuleId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SignalId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Signal {
    pub module: ModuleId,
    pub id: SignalId,
}

impl Signal {
    pub fn value(self) -> Expr {
        Expr::Signal(self)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Expr {
    Lit {
        value: i128,
        ty: BitType,
    },
    Signal(Signal),
    Not(Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Xor(Box<Expr>, Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Eq(Box<Expr>, Box<Expr>),
    Ne(Box<Expr>, Box<Expr>),
    Lt(Box<Expr>, Box<Expr>),
    Mux {
        cond: Box<Expr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
    },
    Slice {
        expr: Box<Expr>,
        lsb: Width,
        width: Width,
    },
    Zext {
        expr: Box<Expr>,
        width: Width,
    },
    Sext {
        expr: Box<Expr>,
        width: Width,
    },
    Trunc {
        expr: Box<Expr>,
        width: Width,
    },
    Cast {
        expr: Box<Expr>,
        signedness: Signedness,
    },
    Concat(Vec<Expr>),
    MemRead {
        mem: Signal,
        addr: Box<Expr>,
    },
}

impl Expr {
    pub fn eq_expr(self, rhs: impl Into<Expr>) -> Expr {
        Expr::Eq(Box::new(self), Box::new(rhs.into()))
    }

    pub fn ne_expr(self, rhs: impl Into<Expr>) -> Expr {
        Expr::Ne(Box::new(self), Box::new(rhs.into()))
    }

    pub fn lt_expr(self, rhs: impl Into<Expr>) -> Expr {
        Expr::Lt(Box::new(self), Box::new(rhs.into()))
    }

    pub fn slice(self, lsb: Width, width: Width) -> Expr {
        Expr::Slice {
            expr: Box::new(self),
            lsb,
            width,
        }
    }

    pub fn zext(self, width: Width) -> Expr {
        Expr::Zext {
            expr: Box::new(self),
            width,
        }
    }

    pub fn sext(self, width: Width) -> Expr {
        Expr::Sext {
            expr: Box::new(self),
            width,
        }
    }

    pub fn trunc(self, width: Width) -> Expr {
        Expr::Trunc {
            expr: Box::new(self),
            width,
        }
    }

    pub fn as_uint(self) -> Expr {
        Expr::Cast {
            expr: Box::new(self),
            signedness: Signedness::Unsigned,
        }
    }

    pub fn as_sint(self) -> Expr {
        Expr::Cast {
            expr: Box::new(self),
            signedness: Signedness::Signed,
        }
    }
}

pub fn lit(value: u128, width: Width) -> Expr {
    lit_u(value, width)
}

pub fn lit_u(value: u128, width: Width) -> Expr {
    Expr::Lit {
        value: value as i128,
        ty: uint(width),
    }
}

pub fn lit_s(value: i128, width: Width) -> Expr {
    Expr::Lit {
        value,
        ty: sint(width),
    }
}

pub fn mux(cond: impl Into<Expr>, then_expr: impl Into<Expr>, else_expr: impl Into<Expr>) -> Expr {
    Expr::Mux {
        cond: Box::new(cond.into()),
        then_expr: Box::new(then_expr.into()),
        else_expr: Box::new(else_expr.into()),
    }
}

pub fn concat(parts: impl IntoIterator<Item = Expr>) -> Expr {
    Expr::Concat(parts.into_iter().collect())
}

pub fn zext(expr: impl Into<Expr>, width: Width) -> Expr {
    expr.into().zext(width)
}

pub fn sext(expr: impl Into<Expr>, width: Width) -> Expr {
    expr.into().sext(width)
}

pub fn trunc(expr: impl Into<Expr>, width: Width) -> Expr {
    expr.into().trunc(width)
}

pub fn as_uint(expr: impl Into<Expr>) -> Expr {
    expr.into().as_uint()
}

pub fn as_sint(expr: impl Into<Expr>) -> Expr {
    expr.into().as_sint()
}

pub fn mem_read(mem: Signal, addr: impl Into<Expr>) -> Expr {
    Expr::MemRead {
        mem,
        addr: Box::new(addr.into()),
    }
}

impl From<Signal> for Expr {
    fn from(value: Signal) -> Self {
        Expr::Signal(value)
    }
}

impl Not for Expr {
    type Output = Expr;

    fn not(self) -> Self::Output {
        Expr::Not(Box::new(self))
    }
}

macro_rules! impl_expr_binop {
    ($trait:ident, $method:ident, $variant:ident) => {
        impl<T: Into<Expr>> $trait<T> for Expr {
            type Output = Expr;

            fn $method(self, rhs: T) -> Self::Output {
                Expr::$variant(Box::new(self), Box::new(rhs.into()))
            }
        }

        impl<T: Into<Expr>> $trait<T> for Signal {
            type Output = Expr;

            fn $method(self, rhs: T) -> Self::Output {
                Expr::$variant(Box::new(self.into()), Box::new(rhs.into()))
            }
        }
    };
}

impl_expr_binop!(Add, add, Add);
impl_expr_binop!(Sub, sub, Sub);
impl_expr_binop!(Mul, mul, Mul);
impl_expr_binop!(BitAnd, bitand, And);
impl_expr_binop!(BitOr, bitor, Or);
impl_expr_binop!(BitXor, bitxor, Xor);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalKind {
    Input,
    Output,
    Inout,
    Wire,
    Reg {
        clock: Option<Signal>,
        reset: Option<Reset>,
        next: Option<Expr>,
    },
    Mem {
        addr_width: Width,
        data_width: Width,
        depth: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reset {
    pub signal: Signal,
    pub value: i128,
    pub kind: ResetKind,
    pub polarity: ResetPolarity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResetKind {
    Sync,
    Async,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResetPolarity {
    ActiveHigh,
    ActiveLow,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignalInfo {
    pub handle: Signal,
    pub name: String,
    pub width: Width,
    pub ty: BitType,
    pub kind: SignalKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Assignment {
    pub dst: Signal,
    pub expr: Expr,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryWrite {
    pub mem: Signal,
    pub clock: Signal,
    pub enable: Expr,
    pub addr: Expr,
    pub data: Expr,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InitialRegisterValue {
    pub signal: Signal,
    pub value: i128,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InitialMemoryValue {
    pub mem: Signal,
    pub addr: usize,
    pub value: i128,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Assertion {
    pub name: String,
    pub clock: Option<Signal>,
    pub enable: Option<Expr>,
    pub condition: Expr,
    pub message: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoverPoint {
    pub name: String,
    pub clock: Option<Signal>,
    pub enable: Option<Expr>,
    pub condition: Expr,
    pub message: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Instance {
    pub name: String,
    pub module: String,
    pub connections: Vec<(String, Signal)>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Module {
    pub id: ModuleId,
    pub name: String,
    pub is_external: bool,
    pub signals: Vec<SignalInfo>,
    pub assignments: Vec<Assignment>,
    pub memory_writes: Vec<MemoryWrite>,
    pub initial_register_values: Vec<InitialRegisterValue>,
    pub initial_memory_values: Vec<InitialMemoryValue>,
    pub assertions: Vec<Assertion>,
    pub cover_points: Vec<CoverPoint>,
    pub instances: Vec<Instance>,
    pub state_types: Vec<StateType>,
    pub state_signals: Vec<StateSignal>,
    pub bundle_types: Vec<BundleType>,
    pub bundle_signals: Vec<BundleSignal>,
    pub interface_types: Vec<InterfaceType>,
    pub interface_signals: Vec<InterfaceSignal>,
    pub builder_diagnostics: Vec<Diagnostic>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateSignal {
    pub signal: Signal,
    pub state_type: StateType,
    pub reset_variant: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleSignal {
    pub name: String,
    pub bundle_type: BundleType,
    pub fields: Vec<BundleSignalField>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleSignalField {
    pub path: Vec<String>,
    pub signal: Signal,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterfaceSignal {
    pub name: String,
    pub interface_type: InterfaceType,
    pub ports: Vec<InterfaceSignalPort>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterfaceSignalPort {
    pub name: String,
    pub direction: InterfaceDirection,
    pub signals: InterfaceSignalPortSignals,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterfaceSignalPortSignals {
    Scalar { signal: Signal },
    Bundle { fields: Vec<BundleSignalField> },
}

impl Module {
    pub fn signal(&self, signal: Signal) -> Option<&SignalInfo> {
        if signal.module != self.id {
            return None;
        }
        self.signals.get(signal.id.0)
    }

    pub fn signal_mut(&mut self, signal: Signal) -> Option<&mut SignalInfo> {
        if signal.module != self.id {
            return None;
        }
        self.signals.get_mut(signal.id.0)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Design {
    pub modules: Vec<Module>,
}

impl Design {
    pub fn module(&self, id: ModuleId) -> Option<&Module> {
        self.modules.get(id.0)
    }

    pub fn module_mut(&mut self, id: ModuleId) -> Option<&mut Module> {
        self.modules.get_mut(id.0)
    }

    pub fn find_module(&self, name: &str) -> Option<&Module> {
        self.modules.iter().find(|m| m.name == name)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Diagnostic {
    pub code: String,
    pub message: String,
    pub module: Option<String>,
    pub signal: Option<String>,
}

impl Diagnostic {
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code: code.to_string(),
            message: message.into(),
            module: None,
            signal: None,
        }
    }

    pub fn with_module(mut self, module: impl Into<String>) -> Self {
        self.module = Some(module.into());
        self
    }

    pub fn with_signal(mut self, signal: impl Into<String>) -> Self {
        self.signal = Some(signal.into());
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorReport {
    pub diagnostics: Vec<Diagnostic>,
}

impl ErrorReport {
    pub fn new(diagnostics: Vec<Diagnostic>) -> Self {
        Self { diagnostics }
    }

    pub fn is_empty(&self) -> bool {
        self.diagnostics.is_empty()
    }
}

impl fmt::Display for ErrorReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for diagnostic in &self.diagnostics {
            write!(f, "{}: {}", diagnostic.code, diagnostic.message)?;
            if let Some(module) = &diagnostic.module {
                write!(f, " [module: {module}]")?;
            }
            if let Some(signal) = &diagnostic.signal {
                write!(f, " [signal: {signal}]")?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl std::error::Error for ErrorReport {}
