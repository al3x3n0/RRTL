use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{braced, parenthesized, parse_macro_input, Expr, Ident, LitInt, LitStr, Result, Token};

#[proc_macro]
pub fn bundle(input: TokenStream) -> TokenStream {
    let bundle = parse_macro_input!(input as BundleSpec);
    bundle.to_tokens().into()
}

#[proc_macro]
pub fn interface(input: TokenStream) -> TokenStream {
    let interface = parse_macro_input!(input as InterfaceSpec);
    interface.to_tokens().into()
}

#[proc_macro]
pub fn extern_module(input: TokenStream) -> TokenStream {
    let extern_module = parse_macro_input!(input as ExternModuleSpec);
    extern_module.to_tokens().into()
}

#[proc_macro]
pub fn instances(input: TokenStream) -> TokenStream {
    let instances = parse_macro_input!(input as InstancesSpec);
    instances.to_tokens().into()
}

#[proc_macro]
pub fn signals(input: TokenStream) -> TokenStream {
    let signals = parse_macro_input!(input as SignalsSpec);
    signals.to_tokens().into()
}

#[proc_macro]
pub fn logic(input: TokenStream) -> TokenStream {
    let logic = parse_macro_input!(input as LogicSpec);
    logic.to_tokens().into()
}

#[proc_macro]
pub fn mem_read(input: TokenStream) -> TokenStream {
    let mem_read = parse_macro_input!(input as MemReadSpec);
    mem_read.to_tokens().into()
}

#[proc_macro]
pub fn ready_valid(input: TokenStream) -> TokenStream {
    let ready_valid = parse_macro_input!(input as ReadyValidSpec);
    ready_valid.to_tokens().into()
}

#[proc_macro]
pub fn state(input: TokenStream) -> TokenStream {
    let state = parse_macro_input!(input as StateSpec);
    state.to_tokens().into()
}

struct BundleSpec {
    name: Ident,
    fields: Punctuated<FieldSpec, Token![,]>,
}

struct FieldSpec {
    name: Ident,
    ty: TypeSpec,
}

enum TypeSpec {
    Scalar { ctor: Ident, width: LitInt },
    Bundle(BundleSpec),
}

struct InterfaceSpec {
    name: Ident,
    ports: Punctuated<InterfacePortSpec, Token![,]>,
}

struct InterfacePortSpec {
    direction: InterfaceDirection,
    name: Ident,
    ty: TypeSpec,
}

enum InterfaceDirection {
    Input,
    Output,
}

struct ExternModuleSpec {
    design: Ident,
    name: Ident,
    ports: Punctuated<ExternPortSpec, Token![,]>,
}

struct ExternPortSpec {
    direction: ExternPortDirection,
    name: Ident,
    ty: ExternPortTypeSpec,
}

enum ExternPortDirection {
    Input,
    Output,
    Inout,
}

enum ExternPortTypeSpec {
    Scalar { ctor: Ident, width: LitInt },
}

struct InstancesSpec {
    builder: Ident,
    stmts: Punctuated<InstanceStmtSpec, Token![,]>,
}

struct InstanceStmtSpec {
    kind: InstanceStmtKind,
    name: Ident,
    module: Ident,
    connections: Punctuated<InstanceConnectionSpec, Token![,]>,
}

enum InstanceStmtKind {
    Scalar,
    Bundles,
    Interfaces,
}

struct InstanceConnectionSpec {
    port: Ident,
    signal: Expr,
}

struct SignalsSpec {
    builder: Ident,
    decls: Punctuated<SignalDeclSpec, Token![,]>,
}

struct SignalDeclSpec {
    kind: SignalDeclKind,
    name: Ident,
    ty: SignalDeclType,
}

enum SignalDeclKind {
    Input,
    Output,
    Wire,
    Reg,
    InputBundle,
    OutputBundle,
    WireBundle,
    Interface,
    Mem,
    RvSink,
    RvSource,
}

enum SignalDeclType {
    Scalar {
        ctor: Ident,
        width: LitInt,
    },
    Bundle(BundleSpec),
    Interface(InterfaceSpec),
    Mem {
        addr_width: LitInt,
        data_ctor: Ident,
        data_width: LitInt,
        depth: LitInt,
    },
    ReadyValid(ReadyValidPayloadSpec),
}

enum ReadyValidPayloadSpec {
    Scalar { ctor: Ident, width: LitInt },
    Bundle(BundleSpec),
}

struct LogicSpec {
    builder: Ident,
    stmts: Vec<LogicStmt>,
}

enum LogicStmt {
    Assign {
        dst: Ident,
        expr: Expr,
    },
    AssignBundle {
        dst: Ident,
        src: Ident,
    },
    AssignBundleWhen {
        dst: Ident,
        enable: Expr,
        src: Ident,
    },
    Clock {
        reg: Ident,
        clock: Ident,
    },
    Reset {
        kind: ResetStmtKind,
        reg: Ident,
        reset: Ident,
        value: Expr,
    },
    Next {
        reg: Ident,
        expr: Expr,
    },
    StateNextHold {
        state: Ident,
        transitions: Punctuated<StateTransitionSpec, Token![,]>,
    },
    StateNext {
        state: Ident,
        default_variant: Ident,
        transitions: Punctuated<StateTransitionSpec, Token![,]>,
    },
    MemWrite {
        mem: Ident,
        clock: Ident,
        enable: Expr,
        addr: Expr,
        data: Expr,
    },
    Assert {
        name: Ident,
        condition: Expr,
    },
    AssertWhen {
        name: Ident,
        enable: Expr,
        condition: Expr,
    },
    AssertClocked {
        name: Ident,
        clock: Ident,
        condition: Expr,
    },
    AssertMsg {
        name: Ident,
        condition: Expr,
        message: LitStr,
    },
    Cover {
        name: Ident,
        condition: Expr,
    },
    CoverWhen {
        name: Ident,
        enable: Expr,
        condition: Expr,
    },
    CoverClocked {
        name: Ident,
        clock: Ident,
        condition: Expr,
    },
    CoverMsg {
        name: Ident,
        condition: Expr,
        message: LitStr,
    },
}

enum ResetStmtKind {
    SyncHigh,
    SyncLow,
    AsyncHigh,
    AsyncLow,
}

struct StateSpec {
    name: Ident,
    ty: StateTypeSpec,
    variants: Punctuated<StateVariantSpec, Token![,]>,
}

enum StateTypeSpec {
    Scalar { ctor: Ident, width: LitInt },
}

struct StateVariantSpec {
    name: Ident,
    value: Expr,
}

struct StateTransitionSpec {
    cond: Expr,
    variant: Ident,
}

struct MemReadSpec {
    builder: Ident,
    mem: Ident,
    addr: Expr,
}

struct ReadyValidSpec {
    builder: Ident,
    stmts: Vec<ReadyValidStmtSpec>,
}

enum ReadyValidStmtSpec {
    Connect {
        input: Ident,
        output: Ident,
    },
    RegisterSlice {
        name: Ident,
        input: Ident,
        output: Ident,
        clock: Ident,
        reset: Ident,
    },
    SkidBuffer {
        name: Ident,
        input: Ident,
        output: Ident,
        clock: Ident,
        reset: Ident,
    },
    Fifo {
        name: Ident,
        input: Ident,
        output: Ident,
        clock: Ident,
        reset: Ident,
        depth: LitInt,
    },
    MemFifo {
        name: Ident,
        input: Ident,
        output: Ident,
        clock: Ident,
        reset: Ident,
        depth: LitInt,
    },
}

impl BundleSpec {
    fn parse_with_name(name: Ident, input: ParseStream<'_>) -> Result<Self> {
        let content;
        braced!(content in input);
        let fields = content.parse_terminated(FieldSpec::parse, Token![,])?;
        if fields.is_empty() {
            return Err(syn::Error::new(
                name.span(),
                "bundle must contain at least one field",
            ));
        }

        Ok(Self { name, fields })
    }

    fn to_tokens(&self) -> TokenStream2 {
        let name = self.name.to_string();
        let fields = self.fields.iter().map(FieldSpec::to_tokens);

        quote! {
            ::rrtl::bundle_type(#name, [#(#fields),*])
        }
    }
}

impl Parse for BundleSpec {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let name = input.parse()?;
        Self::parse_with_name(name, input)
    }
}

impl FieldSpec {
    fn to_tokens(&self) -> TokenStream2 {
        let name = self.name.to_string();
        match &self.ty {
            TypeSpec::Scalar { ctor, width } => {
                let ty = scalar_type_tokens(ctor, width);
                quote! {
                    ::rrtl::field(#name, #ty)
                }
            }
            TypeSpec::Bundle(bundle) => {
                let bundle = bundle.to_tokens();
                quote! {
                    ::rrtl::nested(#name, #bundle)
                }
            }
        }
    }
}

impl Parse for FieldSpec {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let name = input.parse()?;
        input.parse::<Token![:]>()?;
        let ty = input.parse()?;

        Ok(Self { name, ty })
    }
}

impl Parse for TypeSpec {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let ctor_or_nested_name: Ident = input.parse()?;

        if input.peek(syn::token::Brace) {
            return Ok(TypeSpec::Bundle(BundleSpec::parse_with_name(
                ctor_or_nested_name,
                input,
            )?));
        }

        let ctor_name = ctor_or_nested_name.to_string();
        if ctor_name != "uint" && ctor_name != "sint" {
            return Err(syn::Error::new(
                ctor_or_nested_name.span(),
                "expected uint(width), sint(width), or NestedName { ... }",
            ));
        }

        let content;
        parenthesized!(content in input);
        let width = content.parse()?;
        if !content.is_empty() {
            return Err(syn::Error::new(
                content.span(),
                "expected a single integer width literal",
            ));
        }

        Ok(TypeSpec::Scalar {
            ctor: ctor_or_nested_name,
            width,
        })
    }
}

impl TypeSpec {
    fn to_interface_port_type_tokens(&self) -> TokenStream2 {
        match self {
            TypeSpec::Scalar { ctor, width } => scalar_type_tokens(ctor, width),
            TypeSpec::Bundle(bundle) => bundle.to_tokens(),
        }
    }
}

impl InterfaceSpec {
    fn to_tokens(&self) -> TokenStream2 {
        let name = self.name.to_string();
        let ports = self.ports.iter().map(InterfacePortSpec::to_tokens);

        quote! {
            ::rrtl::interface_type(#name, [#(#ports),*])
        }
    }
}

impl Parse for InterfaceSpec {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let name: Ident = input.parse()?;
        let content;
        braced!(content in input);
        let ports = content.parse_terminated(InterfacePortSpec::parse, Token![,])?;
        if ports.is_empty() {
            return Err(syn::Error::new(
                name.span(),
                "interface must contain at least one port",
            ));
        }

        Ok(Self { name, ports })
    }
}

impl InterfacePortSpec {
    fn to_tokens(&self) -> TokenStream2 {
        let name = self.name.to_string();
        let ty = self.ty.to_interface_port_type_tokens();

        match self.direction {
            InterfaceDirection::Input => quote! {
                ::rrtl::iface_input(#name, #ty)
            },
            InterfaceDirection::Output => quote! {
                ::rrtl::iface_output(#name, #ty)
            },
        }
    }
}

impl Parse for InterfacePortSpec {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let direction_ident: Ident = input.parse()?;
        let direction = match direction_ident.to_string().as_str() {
            "input" => InterfaceDirection::Input,
            "output" => InterfaceDirection::Output,
            _ => {
                return Err(syn::Error::new(
                    direction_ident.span(),
                    "expected interface port direction `input` or `output`",
                ))
            }
        };

        let name = input.parse()?;
        input.parse::<Token![:]>()?;
        let ty = input.parse()?;

        Ok(Self {
            direction,
            name,
            ty,
        })
    }
}

impl ExternModuleSpec {
    fn to_tokens(&self) -> TokenStream2 {
        let design = &self.design;
        let module_name = self.name.to_string();
        let ports = self.ports.iter().map(ExternPortSpec::to_tokens);

        quote! {
            {
                let mut ext = #design.extern_module(#module_name);
                #(#ports)*
            }
        }
    }
}

impl Parse for ExternModuleSpec {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let design: Ident = input.parse()?;
        let name: Ident = input.parse()?;
        let content;
        braced!(content in input);
        let ports = content.parse_terminated(ExternPortSpec::parse, Token![,])?;
        if ports.is_empty() {
            return Err(syn::Error::new(
                name.span(),
                "extern_module! must contain at least one port",
            ));
        }

        Ok(Self {
            design,
            name,
            ports,
        })
    }
}

impl ExternPortSpec {
    fn to_tokens(&self) -> TokenStream2 {
        let name = self.name.to_string();
        let ty = self.ty.to_tokens();

        match self.direction {
            ExternPortDirection::Input => quote! {
                ext.input(#name, #ty);
            },
            ExternPortDirection::Output => quote! {
                ext.output(#name, #ty);
            },
            ExternPortDirection::Inout => quote! {
                ext.inout(#name, #ty);
            },
        }
    }
}

impl Parse for ExternPortSpec {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let direction_ident: Ident = input.parse()?;
        let direction = match direction_ident.to_string().as_str() {
            "input" => ExternPortDirection::Input,
            "output" => ExternPortDirection::Output,
            "inout" => ExternPortDirection::Inout,
            _ => {
                return Err(syn::Error::new(
                    direction_ident.span(),
                    "expected external port direction `input`, `output`, or `inout`",
                ));
            }
        };

        let name = input.parse()?;
        input.parse::<Token![:]>()?;
        let ty = input.parse()?;

        Ok(Self {
            direction,
            name,
            ty,
        })
    }
}

impl ExternPortTypeSpec {
    fn to_tokens(&self) -> TokenStream2 {
        match self {
            Self::Scalar { ctor, width } => scalar_type_tokens(ctor, width),
        }
    }
}

impl Parse for ExternPortTypeSpec {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let ty: TypeSpec = input.parse()?;
        match ty {
            TypeSpec::Scalar { ctor, width } => Ok(Self::Scalar { ctor, width }),
            TypeSpec::Bundle(bundle) => Err(syn::Error::new(
                bundle.name.span(),
                "external module ports require uint(width) or sint(width)",
            )),
        }
    }
}

impl InstancesSpec {
    fn to_tokens(&self) -> TokenStream2 {
        let builder = &self.builder;
        let stmts = self.stmts.iter().map(|stmt| stmt.to_tokens(builder));

        quote! {
            #(#stmts)*
        }
    }
}

impl Parse for InstancesSpec {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let builder: Ident = input.parse()?;
        let content;
        braced!(content in input);
        let stmts = content.parse_terminated(InstanceStmtSpec::parse, Token![,])?;
        if stmts.is_empty() {
            return Err(syn::Error::new(
                builder.span(),
                "instances! must contain at least one instance",
            ));
        }

        Ok(Self { builder, stmts })
    }
}

impl InstanceStmtSpec {
    fn to_tokens(&self, builder: &Ident) -> TokenStream2 {
        let name = self.name.to_string();
        let module = self.module.to_string();
        let connections = self
            .connections
            .iter()
            .map(InstanceConnectionSpec::to_tokens);

        match self.kind {
            InstanceStmtKind::Scalar => quote! {
                #builder.instance(#name, #module, [#(#connections),*]);
            },
            InstanceStmtKind::Bundles => quote! {
                #builder.instance_bundles(#name, #module, [#(#connections),*]);
            },
            InstanceStmtKind::Interfaces => quote! {
                #builder.instance_interfaces(#name, #module, [#(#connections),*]);
            },
        }
    }
}

impl Parse for InstanceStmtSpec {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let kind_ident: Ident = input.parse()?;
        let kind = match kind_ident.to_string().as_str() {
            "instance" => InstanceStmtKind::Scalar,
            "instance_bundles" => InstanceStmtKind::Bundles,
            "instance_interfaces" => InstanceStmtKind::Interfaces,
            _ => {
                return Err(syn::Error::new(
                    kind_ident.span(),
                    "expected instance statement kind `instance`, `instance_bundles`, or `instance_interfaces`",
                ));
            }
        };

        let name: Ident = input.parse()?;
        input.parse::<Token![:]>()?;
        let module: Ident = input.parse()?;

        let content;
        braced!(content in input);
        let connections = content.parse_terminated(InstanceConnectionSpec::parse, Token![,])?;
        if connections.is_empty() {
            return Err(syn::Error::new(
                name.span(),
                "instance declarations must contain at least one connection",
            ));
        }

        Ok(Self {
            kind,
            name,
            module,
            connections,
        })
    }
}

impl InstanceConnectionSpec {
    fn to_tokens(&self) -> TokenStream2 {
        let port = self.port.to_string();
        let signal = &self.signal;

        quote! {
            (#port, #signal)
        }
    }
}

impl Parse for InstanceConnectionSpec {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let port: Ident = input.parse()?;
        input.parse::<Token![:]>()?;
        let signal: Expr = input.parse()?;

        Ok(Self { port, signal })
    }
}

fn scalar_type_tokens(ctor: &Ident, width: &LitInt) -> TokenStream2 {
    match ctor.to_string().as_str() {
        "uint" => quote! {
            ::rrtl::uint(#width)
        },
        "sint" => quote! {
            ::rrtl::sint(#width)
        },
        _ => unreachable!("unsupported scalar constructors are rejected during parsing"),
    }
}

impl StateSpec {
    fn to_tokens(&self) -> TokenStream2 {
        let name = self.name.to_string();
        let ty = self.ty.to_tokens();
        let variants = self.variants.iter().map(StateVariantSpec::to_tokens);

        quote! {
            ::rrtl::state_type(#name, #ty, [#(#variants),*])
        }
    }
}

impl Parse for StateSpec {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let name: Ident = input.parse()?;
        input.parse::<Token![:]>()?;
        let ty = input.parse()?;

        let content;
        braced!(content in input);
        let variants = content.parse_terminated(StateVariantSpec::parse, Token![,])?;
        if variants.is_empty() {
            return Err(syn::Error::new(
                name.span(),
                "state! must contain at least one variant",
            ));
        }

        Ok(Self { name, ty, variants })
    }
}

impl StateTypeSpec {
    fn to_tokens(&self) -> TokenStream2 {
        match self {
            StateTypeSpec::Scalar { ctor, width } => scalar_type_tokens(ctor, width),
        }
    }
}

impl Parse for StateTypeSpec {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let ty: TypeSpec = input.parse()?;
        match ty {
            TypeSpec::Scalar { ctor, width } => Ok(Self::Scalar { ctor, width }),
            TypeSpec::Bundle(bundle) => Err(syn::Error::new(
                bundle.name.span(),
                "state types require uint(width) or sint(width)",
            )),
        }
    }
}

impl StateVariantSpec {
    fn to_tokens(&self) -> TokenStream2 {
        let name = self.name.to_string();
        let value = &self.value;

        quote! {
            (#name, #value)
        }
    }
}

impl Parse for StateVariantSpec {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let name = input.parse()?;
        input.parse::<Token![=]>()?;
        let value = input.parse()?;

        Ok(Self { name, value })
    }
}

impl StateTransitionSpec {
    fn to_tokens(&self) -> TokenStream2 {
        let cond = &self.cond;
        let variant = self.variant.to_string();

        quote! {
            (#cond, #variant)
        }
    }
}

impl Parse for StateTransitionSpec {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let cond = input.parse()?;
        input.parse::<Token![=>]>()?;
        let variant = input.parse()?;

        Ok(Self { cond, variant })
    }
}

impl MemReadSpec {
    fn to_tokens(&self) -> TokenStream2 {
        let builder = &self.builder;
        let mem = &self.mem;
        let addr = &self.addr;

        quote! {
            #builder.mem_read(#mem, #addr)
        }
    }
}

impl Parse for MemReadSpec {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let builder = input.parse()?;
        input.parse::<Token![,]>()?;
        let mem = input.parse()?;
        input.parse::<Token![,]>()?;
        let addr = input.parse()?;

        Ok(Self { builder, mem, addr })
    }
}

impl ReadyValidSpec {
    fn to_tokens(&self) -> TokenStream2 {
        let builder = &self.builder;
        let stmts = self.stmts.iter().map(|stmt| stmt.to_tokens(builder));

        quote! {
            #(#stmts)*
        }
    }
}

impl Parse for ReadyValidSpec {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let builder: Ident = input.parse()?;
        let content;
        braced!(content in input);

        let mut stmts = Vec::new();
        while !content.is_empty() {
            stmts.push(content.parse()?);
        }

        if stmts.is_empty() {
            return Err(syn::Error::new(
                builder.span(),
                "ready_valid! must contain at least one helper statement",
            ));
        }

        Ok(Self { builder, stmts })
    }
}

impl ReadyValidStmtSpec {
    fn to_tokens(&self, builder: &Ident) -> TokenStream2 {
        match self {
            Self::Connect { input, output } => {
                quote! {
                    #builder.rv_connect(&#input, &#output);
                }
            }
            Self::RegisterSlice {
                name,
                input,
                output,
                clock,
                reset,
            } => {
                let name = name.to_string();
                quote! {
                    #builder.rv_register_slice(#name, &#input, &#output, #clock, #reset);
                }
            }
            Self::SkidBuffer {
                name,
                input,
                output,
                clock,
                reset,
            } => {
                let name = name.to_string();
                quote! {
                    #builder.rv_skid_buffer(#name, &#input, &#output, #clock, #reset);
                }
            }
            Self::Fifo {
                name,
                input,
                output,
                clock,
                reset,
                depth,
            } => {
                let name = name.to_string();
                quote! {
                    #builder.rv_fifo(#name, &#input, &#output, #clock, #reset, #depth);
                }
            }
            Self::MemFifo {
                name,
                input,
                output,
                clock,
                reset,
                depth,
            } => {
                let name = name.to_string();
                quote! {
                    #builder.rv_mem_fifo(#name, &#input, &#output, #clock, #reset, #depth);
                }
            }
        }
    }
}

impl Parse for ReadyValidStmtSpec {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let kind_ident: Ident = input.parse()?;
        match kind_ident.to_string().as_str() {
            "connect" => {
                let input_channel = input.parse()?;
                input.parse::<Token![=>]>()?;
                let output = input.parse()?;
                input.parse::<Token![;]>()?;
                Ok(Self::Connect {
                    input: input_channel,
                    output,
                })
            }
            "register_slice" => {
                let (name, input_channel, output, clock, reset) =
                    parse_ready_valid_common(input)?;
                input.parse::<Token![;]>()?;
                Ok(Self::RegisterSlice {
                    name,
                    input: input_channel,
                    output,
                    clock,
                    reset,
                })
            }
            "skid_buffer" => {
                let (name, input_channel, output, clock, reset) =
                    parse_ready_valid_common(input)?;
                input.parse::<Token![;]>()?;
                Ok(Self::SkidBuffer {
                    name,
                    input: input_channel,
                    output,
                    clock,
                    reset,
                })
            }
            "fifo" => {
                let (name, input_channel, output, clock, reset) =
                    parse_ready_valid_common(input)?;
                let depth = parse_ready_valid_depth(input)?;
                input.parse::<Token![;]>()?;
                Ok(Self::Fifo {
                    name,
                    input: input_channel,
                    output,
                    clock,
                    reset,
                    depth,
                })
            }
            "mem_fifo" => {
                let (name, input_channel, output, clock, reset) =
                    parse_ready_valid_common(input)?;
                let depth = parse_ready_valid_depth(input)?;
                input.parse::<Token![;]>()?;
                Ok(Self::MemFifo {
                    name,
                    input: input_channel,
                    output,
                    clock,
                    reset,
                    depth,
                })
            }
            _ => Err(syn::Error::new(
                kind_ident.span(),
                "expected ready/valid helper kind `connect`, `register_slice`, `skid_buffer`, `fifo`, or `mem_fifo`",
            )),
        }
    }
}

fn parse_ready_valid_common(input: ParseStream<'_>) -> Result<(Ident, Ident, Ident, Ident, Ident)> {
    let name = input.parse()?;
    input.parse::<Token![:]>()?;
    let input_channel = input.parse()?;
    input.parse::<Token![=>]>()?;
    let output = input.parse()?;
    input.parse::<Token![,]>()?;
    let clock = input.parse()?;
    input.parse::<Token![,]>()?;
    let reset = input.parse()?;

    Ok((name, input_channel, output, clock, reset))
}

fn parse_ready_valid_depth(input: ParseStream<'_>) -> Result<LitInt> {
    input.parse::<Token![,]>()?;
    let depth_keyword: Ident = input.parse()?;
    if depth_keyword != "depth" {
        return Err(syn::Error::new(
            depth_keyword.span(),
            "expected ready/valid FIFO depth <integer literal>",
        ));
    }
    input.parse()
}

impl SignalsSpec {
    fn to_tokens(&self) -> TokenStream2 {
        let builder = &self.builder;
        let decls = self.decls.iter().map(|decl| decl.to_tokens(builder));

        quote! {
            (#(#decls),*,)
        }
    }
}

impl Parse for SignalsSpec {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let builder: Ident = input.parse()?;
        let content;
        braced!(content in input);
        let decls = content.parse_terminated(SignalDeclSpec::parse, Token![,])?;
        if decls.is_empty() {
            return Err(syn::Error::new(
                builder.span(),
                "signals! must contain at least one declaration",
            ));
        }

        Ok(Self { builder, decls })
    }
}

impl SignalDeclSpec {
    fn to_tokens(&self, builder: &Ident) -> TokenStream2 {
        let name = self.name.to_string();
        match (&self.kind, &self.ty) {
            (SignalDeclKind::Input, SignalDeclType::Scalar { ctor, width }) => {
                let ty = scalar_type_tokens(ctor, width);
                quote! {
                    #builder.input(#name, #ty)
                }
            }
            (SignalDeclKind::Output, SignalDeclType::Scalar { ctor, width }) => {
                let ty = scalar_type_tokens(ctor, width);
                quote! {
                    #builder.output(#name, #ty)
                }
            }
            (SignalDeclKind::Wire, SignalDeclType::Scalar { ctor, width }) => {
                let ty = scalar_type_tokens(ctor, width);
                quote! {
                    #builder.wire(#name, #ty)
                }
            }
            (SignalDeclKind::Reg, SignalDeclType::Scalar { ctor, width }) => {
                let ty = scalar_type_tokens(ctor, width);
                quote! {
                    #builder.reg(#name, #ty)
                }
            }
            (SignalDeclKind::InputBundle, SignalDeclType::Bundle(bundle)) => {
                let ty = bundle.to_tokens();
                quote! {
                    #builder.input_bundle(#name, #ty)
                }
            }
            (SignalDeclKind::OutputBundle, SignalDeclType::Bundle(bundle)) => {
                let ty = bundle.to_tokens();
                quote! {
                    #builder.output_bundle(#name, #ty)
                }
            }
            (SignalDeclKind::WireBundle, SignalDeclType::Bundle(bundle)) => {
                let ty = bundle.to_tokens();
                quote! {
                    #builder.wire_bundle(#name, #ty)
                }
            }
            (SignalDeclKind::Interface, SignalDeclType::Interface(interface)) => {
                let ty = interface.to_tokens();
                quote! {
                    #builder.interface(#name, #ty)
                }
            }
            (
                SignalDeclKind::Mem,
                SignalDeclType::Mem {
                    addr_width,
                    data_ctor,
                    data_width,
                    depth,
                },
            ) => {
                let ty = scalar_type_tokens(data_ctor, data_width);
                quote! {
                    #builder.mem(#name, #addr_width, #ty, #depth)
                }
            }
            (SignalDeclKind::RvSink, SignalDeclType::ReadyValid(payload)) => {
                let payload = payload.to_tokens();
                quote! {
                    #builder.rv_sink(#name, #payload)
                }
            }
            (SignalDeclKind::RvSource, SignalDeclType::ReadyValid(payload)) => {
                let payload = payload.to_tokens();
                quote! {
                    #builder.rv_source(#name, #payload)
                }
            }
            _ => unreachable!("declaration kind/type pairings are rejected during parsing"),
        }
    }
}

impl Parse for SignalDeclSpec {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let kind_ident: Ident = input.parse()?;
        let kind = SignalDeclKind::parse_ident(kind_ident)?;
        let name = input.parse()?;
        input.parse::<Token![:]>()?;
        let ty = kind.parse_type(input)?;

        Ok(Self { kind, name, ty })
    }
}

impl SignalDeclKind {
    fn parse_ident(ident: Ident) -> Result<Self> {
        match ident.to_string().as_str() {
            "input" => Ok(Self::Input),
            "output" => Ok(Self::Output),
            "wire" => Ok(Self::Wire),
            "reg" => Ok(Self::Reg),
            "input_bundle" => Ok(Self::InputBundle),
            "output_bundle" => Ok(Self::OutputBundle),
            "wire_bundle" => Ok(Self::WireBundle),
            "interface" => Ok(Self::Interface),
            "mem" => Ok(Self::Mem),
            "rv_sink" => Ok(Self::RvSink),
            "rv_source" => Ok(Self::RvSource),
            _ => Err(syn::Error::new(
                ident.span(),
                "expected signal declaration kind `input`, `output`, `wire`, `reg`, `input_bundle`, `output_bundle`, `wire_bundle`, `interface`, `mem`, `rv_sink`, or `rv_source`",
            )),
        }
    }

    fn parse_type(&self, input: ParseStream<'_>) -> Result<SignalDeclType> {
        match self {
            Self::Input | Self::Output | Self::Wire | Self::Reg => {
                let ty: TypeSpec = input.parse()?;
                match ty {
                    TypeSpec::Scalar { ctor, width } => Ok(SignalDeclType::Scalar { ctor, width }),
                    TypeSpec::Bundle(bundle) => Err(syn::Error::new(
                        bundle.name.span(),
                        "scalar signal declarations require uint(width) or sint(width)",
                    )),
                }
            }
            Self::InputBundle | Self::OutputBundle | Self::WireBundle => {
                let ty: TypeSpec = input.parse()?;
                match ty {
                    TypeSpec::Bundle(bundle) => Ok(SignalDeclType::Bundle(bundle)),
                    TypeSpec::Scalar { ctor, .. } => Err(syn::Error::new(
                        ctor.span(),
                        "bundle signal declarations require BundleName { ... }",
                    )),
                }
            }
            Self::Interface => {
                let interface = input.parse()?;
                Ok(SignalDeclType::Interface(interface))
            }
            Self::Mem => parse_mem_decl_type(input),
            Self::RvSink | Self::RvSource => {
                let payload = input.parse()?;
                Ok(SignalDeclType::ReadyValid(payload))
            }
        }
    }
}

impl ReadyValidPayloadSpec {
    fn to_tokens(&self) -> TokenStream2 {
        match self {
            ReadyValidPayloadSpec::Scalar { ctor, width } => {
                let ty = scalar_type_tokens(ctor, width);
                quote! {
                    ::rrtl::rv_scalar(#ty)
                }
            }
            ReadyValidPayloadSpec::Bundle(bundle) => {
                let ty = bundle.to_tokens();
                quote! {
                    ::rrtl::rv_bundle(#ty)
                }
            }
        }
    }
}

impl Parse for ReadyValidPayloadSpec {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let payload_kind: Ident = input.parse()?;
        match payload_kind.to_string().as_str() {
            "scalar" => {
                let ty: TypeSpec = input.parse()?;
                match ty {
                    TypeSpec::Scalar { ctor, width } => Ok(Self::Scalar { ctor, width }),
                    TypeSpec::Bundle(bundle) => Err(syn::Error::new(
                        bundle.name.span(),
                        "ready/valid scalar payloads require uint(width) or sint(width)",
                    )),
                }
            }
            "bundle" => {
                let name: Ident = input.parse()?;
                Ok(Self::Bundle(BundleSpec::parse_with_name(name, input)?))
            }
            _ => Err(syn::Error::new(
                payload_kind.span(),
                "expected ready/valid payload `scalar uint(width)`, `scalar sint(width)`, or `bundle BundleName { ... }`",
            )),
        }
    }
}

fn parse_mem_decl_type(input: ParseStream<'_>) -> Result<SignalDeclType> {
    let addr_keyword: Ident = input.parse()?;
    if addr_keyword != "addr" {
        return Err(syn::Error::new(
            addr_keyword.span(),
            "expected memory addr(<integer literal>)",
        ));
    }

    let content;
    parenthesized!(content in input);
    let addr_width = content.parse()?;
    if !content.is_empty() {
        return Err(syn::Error::new(
            content.span(),
            "expected memory addr(<integer literal>)",
        ));
    }

    input.parse::<Token![,]>()?;
    let data_keyword: Ident = input.parse()?;
    if data_keyword != "data" {
        return Err(syn::Error::new(
            data_keyword.span(),
            "expected memory data uint(width) or sint(width)",
        ));
    }

    let data_ty: TypeSpec = input.parse()?;
    let (data_ctor, data_width) = match data_ty {
        TypeSpec::Scalar { ctor, width } => (ctor, width),
        TypeSpec::Bundle(bundle) => {
            return Err(syn::Error::new(
                bundle.name.span(),
                "memory data types require uint(width) or sint(width)",
            ));
        }
    };

    input.parse::<Token![,]>()?;
    let depth_keyword: Ident = input.parse()?;
    if depth_keyword != "depth" {
        return Err(syn::Error::new(
            depth_keyword.span(),
            "expected memory depth <integer literal>",
        ));
    }
    let depth = input.parse()?;

    Ok(SignalDeclType::Mem {
        addr_width,
        data_ctor,
        data_width,
        depth,
    })
}

impl LogicSpec {
    fn to_tokens(&self) -> TokenStream2 {
        let builder = &self.builder;
        let stmts = self.stmts.iter().map(|stmt| stmt.to_tokens(builder));

        quote! {
            #(#stmts)*
        }
    }
}

impl Parse for LogicSpec {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let builder: Ident = input.parse()?;
        let content;
        braced!(content in input);

        let mut stmts = Vec::new();
        while !content.is_empty() {
            stmts.push(content.parse()?);
        }

        if stmts.is_empty() {
            return Err(syn::Error::new(
                builder.span(),
                "logic! must contain at least one statement",
            ));
        }

        Ok(Self { builder, stmts })
    }
}

impl LogicStmt {
    fn to_tokens(&self, builder: &Ident) -> TokenStream2 {
        match self {
            LogicStmt::Assign { dst, expr } => quote! {
                {
                    let __rrtl_assign_expr = #expr;
                    #builder.assign(#dst, __rrtl_assign_expr);
                }
            },
            LogicStmt::AssignBundle { dst, src } => quote! {
                #builder.assign_bundle(&#dst, &#src);
            },
            LogicStmt::AssignBundleWhen { dst, enable, src } => quote! {
                {
                    let __rrtl_assign_bundle_enable = #enable;
                    #builder.assign_bundle_when(&#dst, __rrtl_assign_bundle_enable, &#src);
                }
            },
            LogicStmt::Clock { reg, clock } => quote! {
                #builder.clock(#reg, #clock);
            },
            LogicStmt::Reset {
                kind,
                reg,
                reset,
                value,
            } => match kind {
                ResetStmtKind::SyncHigh => quote! {
                    {
                        let __rrtl_reset_value = #value;
                        #builder.reset(#reg, #reset, __rrtl_reset_value);
                    }
                },
                ResetStmtKind::SyncLow => quote! {
                    {
                        let __rrtl_reset_value = #value;
                        #builder.reset_low(#reg, #reset, __rrtl_reset_value);
                    }
                },
                ResetStmtKind::AsyncHigh => quote! {
                    {
                        let __rrtl_reset_value = #value;
                        #builder.async_reset(#reg, #reset, __rrtl_reset_value);
                    }
                },
                ResetStmtKind::AsyncLow => quote! {
                    {
                        let __rrtl_reset_value = #value;
                        #builder.async_reset_low(#reg, #reset, __rrtl_reset_value);
                    }
                },
            },
            LogicStmt::Next { reg, expr } => quote! {
                {
                    let __rrtl_next_expr = #expr;
                    #builder.next(#reg, __rrtl_next_expr);
                }
            },
            LogicStmt::StateNextHold { state, transitions } => {
                let transitions = transitions.iter().map(StateTransitionSpec::to_tokens);
                quote! {
                    #builder.state_next_hold(&#state, [#(#transitions),*]);
                }
            }
            LogicStmt::StateNext {
                state,
                default_variant,
                transitions,
            } => {
                let default_variant = default_variant.to_string();
                let transitions = transitions.iter().map(StateTransitionSpec::to_tokens);
                quote! {
                    #builder.state_next(&#state, #default_variant, [#(#transitions),*]);
                }
            }
            LogicStmt::MemWrite {
                mem,
                clock,
                enable,
                addr,
                data,
            } => quote! {
                {
                    let __rrtl_mem_write_enable = #enable;
                    let __rrtl_mem_write_addr = #addr;
                    let __rrtl_mem_write_data = #data;
                    #builder.mem_write(
                        #mem,
                        #clock,
                        __rrtl_mem_write_enable,
                        __rrtl_mem_write_addr,
                        __rrtl_mem_write_data,
                    );
                }
            },
            LogicStmt::Assert { name, condition } => {
                let name = name.to_string();
                quote! {
                    {
                        let __rrtl_assert_condition = #condition;
                        #builder.assert(#name, __rrtl_assert_condition);
                    }
                }
            }
            LogicStmt::AssertWhen {
                name,
                enable,
                condition,
            } => {
                let name = name.to_string();
                quote! {
                    {
                        let __rrtl_assert_enable = #enable;
                        let __rrtl_assert_condition = #condition;
                        #builder.assert_when(#name, __rrtl_assert_enable, __rrtl_assert_condition);
                    }
                }
            }
            LogicStmt::AssertClocked {
                name,
                clock,
                condition,
            } => {
                let name = name.to_string();
                quote! {
                    {
                        let __rrtl_assert_condition = #condition;
                        #builder.assert_clocked(#name, #clock, __rrtl_assert_condition);
                    }
                }
            }
            LogicStmt::AssertMsg {
                name,
                condition,
                message,
            } => {
                let name = name.to_string();
                quote! {
                    {
                        let __rrtl_assert_condition = #condition;
                        #builder.assert_msg(#name, __rrtl_assert_condition, #message);
                    }
                }
            }
            LogicStmt::Cover { name, condition } => {
                let name = name.to_string();
                quote! {
                    {
                        let __rrtl_cover_condition = #condition;
                        #builder.cover(#name, __rrtl_cover_condition);
                    }
                }
            }
            LogicStmt::CoverWhen {
                name,
                enable,
                condition,
            } => {
                let name = name.to_string();
                quote! {
                    {
                        let __rrtl_cover_enable = #enable;
                        let __rrtl_cover_condition = #condition;
                        #builder.cover_when(#name, __rrtl_cover_enable, __rrtl_cover_condition);
                    }
                }
            }
            LogicStmt::CoverClocked {
                name,
                clock,
                condition,
            } => {
                let name = name.to_string();
                quote! {
                    {
                        let __rrtl_cover_condition = #condition;
                        #builder.cover_clocked(#name, #clock, __rrtl_cover_condition);
                    }
                }
            }
            LogicStmt::CoverMsg {
                name,
                condition,
                message,
            } => {
                let name = name.to_string();
                quote! {
                    {
                        let __rrtl_cover_condition = #condition;
                        #builder.cover_msg(#name, __rrtl_cover_condition, #message);
                    }
                }
            }
        }
    }
}

impl Parse for LogicStmt {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let kind_ident: Ident = input.parse()?;
        match kind_ident.to_string().as_str() {
            "assign" => {
                let dst = input.parse()?;
                input.parse::<Token![=]>()?;
                let expr = input.parse()?;
                input.parse::<Token![;]>()?;
                Ok(Self::Assign { dst, expr })
            }
            "assign_bundle" => {
                let dst = input.parse()?;
                input.parse::<Token![=]>()?;
                let src = input.parse()?;
                input.parse::<Token![;]>()?;
                Ok(Self::AssignBundle { dst, src })
            }
            "assign_bundle_when" => {
                let dst = input.parse()?;
                input.parse::<Token![:]>()?;
                let enable = input.parse()?;
                input.parse::<Token![,]>()?;
                let src = input.parse()?;
                input.parse::<Token![;]>()?;
                Ok(Self::AssignBundleWhen { dst, enable, src })
            }
            "clock" => {
                let reg = input.parse()?;
                input.parse::<Token![:]>()?;
                let clock = input.parse()?;
                input.parse::<Token![;]>()?;
                Ok(Self::Clock { reg, clock })
            }
            "reset" => parse_reset_stmt(input, ResetStmtKind::SyncHigh),
            "reset_low" => parse_reset_stmt(input, ResetStmtKind::SyncLow),
            "async_reset" => parse_reset_stmt(input, ResetStmtKind::AsyncHigh),
            "async_reset_low" => parse_reset_stmt(input, ResetStmtKind::AsyncLow),
            "next" => {
                let reg = input.parse()?;
                input.parse::<Token![=]>()?;
                let expr = input.parse()?;
                input.parse::<Token![;]>()?;
                Ok(Self::Next { reg, expr })
            }
            "state_next_hold" => {
                let state = input.parse()?;
                let transitions = parse_state_transitions(input, &state)?;
                input.parse::<Token![;]>()?;
                Ok(Self::StateNextHold { state, transitions })
            }
            "state_next" => {
                let state = input.parse()?;
                let default_keyword: Ident = input.parse()?;
                if default_keyword != "default" {
                    return Err(syn::Error::new(
                        default_keyword.span(),
                        "expected `default` before the default state variant",
                    ));
                }
                let default_variant = input.parse()?;
                let transitions = parse_state_transitions(input, &state)?;
                input.parse::<Token![;]>()?;
                Ok(Self::StateNext {
                    state,
                    default_variant,
                    transitions,
                })
            }
            "mem_write" => {
                let mem = input.parse()?;
                input.parse::<Token![:]>()?;
                let clock = input.parse()?;
                input.parse::<Token![,]>()?;
                let enable = input.parse()?;
                input.parse::<Token![,]>()?;
                let addr = input.parse()?;
                input.parse::<Token![,]>()?;
                let data = input.parse()?;
                input.parse::<Token![;]>()?;
                Ok(Self::MemWrite {
                    mem,
                    clock,
                    enable,
                    addr,
                    data,
                })
            }
            "assert" => {
                let name = input.parse()?;
                input.parse::<Token![:]>()?;
                let condition = input.parse()?;
                input.parse::<Token![;]>()?;
                Ok(Self::Assert { name, condition })
            }
            "assert_when" => {
                let name = input.parse()?;
                input.parse::<Token![:]>()?;
                let enable = input.parse()?;
                input.parse::<Token![,]>()?;
                let condition = input.parse()?;
                input.parse::<Token![;]>()?;
                Ok(Self::AssertWhen {
                    name,
                    enable,
                    condition,
                })
            }
            "assert_clocked" => {
                let name = input.parse()?;
                input.parse::<Token![:]>()?;
                let clock = input.parse()?;
                input.parse::<Token![,]>()?;
                let condition = input.parse()?;
                input.parse::<Token![;]>()?;
                Ok(Self::AssertClocked {
                    name,
                    clock,
                    condition,
                })
            }
            "assert_msg" => {
                let name = input.parse()?;
                input.parse::<Token![:]>()?;
                let condition = input.parse()?;
                input.parse::<Token![,]>()?;
                let message = input.parse()?;
                input.parse::<Token![;]>()?;
                Ok(Self::AssertMsg {
                    name,
                    condition,
                    message,
                })
            }
            "cover" => {
                let name = input.parse()?;
                input.parse::<Token![:]>()?;
                let condition = input.parse()?;
                input.parse::<Token![;]>()?;
                Ok(Self::Cover { name, condition })
            }
            "cover_when" => {
                let name = input.parse()?;
                input.parse::<Token![:]>()?;
                let enable = input.parse()?;
                input.parse::<Token![,]>()?;
                let condition = input.parse()?;
                input.parse::<Token![;]>()?;
                Ok(Self::CoverWhen {
                    name,
                    enable,
                    condition,
                })
            }
            "cover_clocked" => {
                let name = input.parse()?;
                input.parse::<Token![:]>()?;
                let clock = input.parse()?;
                input.parse::<Token![,]>()?;
                let condition = input.parse()?;
                input.parse::<Token![;]>()?;
                Ok(Self::CoverClocked {
                    name,
                    clock,
                    condition,
                })
            }
            "cover_msg" => {
                let name = input.parse()?;
                input.parse::<Token![:]>()?;
                let condition = input.parse()?;
                input.parse::<Token![,]>()?;
                let message = input.parse()?;
                input.parse::<Token![;]>()?;
                Ok(Self::CoverMsg {
                    name,
                    condition,
                    message,
                })
            }
            _ => Err(syn::Error::new(
                kind_ident.span(),
                "expected logic statement kind `assign`, `assign_bundle`, `assign_bundle_when`, `clock`, `reset`, `reset_low`, `async_reset`, `async_reset_low`, `next`, `state_next_hold`, `state_next`, `mem_write`, `assert`, `assert_when`, `assert_clocked`, `assert_msg`, `cover`, `cover_when`, `cover_clocked`, or `cover_msg`",
            )),
        }
    }
}

fn parse_reset_stmt(input: ParseStream<'_>, kind: ResetStmtKind) -> Result<LogicStmt> {
    let reg = input.parse()?;
    input.parse::<Token![:]>()?;
    let reset = input.parse()?;
    input.parse::<Token![=]>()?;
    let value = input.parse()?;
    input.parse::<Token![;]>()?;

    Ok(LogicStmt::Reset {
        kind,
        reg,
        reset,
        value,
    })
}

fn parse_state_transitions(
    input: ParseStream<'_>,
    state: &Ident,
) -> Result<Punctuated<StateTransitionSpec, Token![,]>> {
    let content;
    braced!(content in input);
    let transitions = content.parse_terminated(StateTransitionSpec::parse, Token![,])?;
    if transitions.is_empty() {
        return Err(syn::Error::new(
            state.span(),
            "state transition blocks must contain at least one transition",
        ));
    }

    Ok(transitions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;

    #[test]
    fn parses_scalar_bundle() {
        let bundle: BundleSpec = syn::parse2(quote! {
            Req {
                valid: uint(1),
                data: sint(8),
            }
        })
        .unwrap();

        assert_eq!(bundle.name.to_string(), "Req");
        assert_eq!(bundle.fields.len(), 2);
    }

    #[test]
    fn parses_nested_bundle() {
        let bundle: BundleSpec = syn::parse2(quote! {
            Req {
                valid: uint(1),
                meta: ReqMeta {
                    tag: uint(4),
                },
            }
        })
        .unwrap();

        assert_eq!(bundle.name.to_string(), "Req");
        assert_eq!(bundle.fields.len(), 2);
    }

    #[test]
    fn rejects_empty_bundle() {
        let err = match syn::parse2::<BundleSpec>(quote! {
            Empty {}
        }) {
            Ok(_) => panic!("expected empty bundle to fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("at least one field"));
    }

    #[test]
    fn rejects_unsupported_constructor() {
        let err = match syn::parse2::<BundleSpec>(quote! {
            Req {
                data: bits(8),
            }
        }) {
            Ok(_) => panic!("expected unsupported constructor to fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("expected uint"));
    }

    #[test]
    fn parses_state_type() {
        let state: StateSpec = syn::parse2(quote! {
            ControllerState: uint(2) {
                Idle = 0,
                Run = 1,
                Done = 2,
            }
        })
        .unwrap();

        assert_eq!(state.name.to_string(), "ControllerState");
        assert_eq!(state.variants.len(), 3);
    }

    #[test]
    fn rejects_empty_state_type() {
        let err = match syn::parse2::<StateSpec>(quote! {
            EmptyState: uint(1) {}
        }) {
            Ok(_) => panic!("expected empty state! variant list to fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("at least one variant"));
    }

    #[test]
    fn parses_scalar_interface() {
        let interface: InterfaceSpec = syn::parse2(quote! {
            Bus {
                input ready: uint(1),
                output data: sint(8),
            }
        })
        .unwrap();

        assert_eq!(interface.name.to_string(), "Bus");
        assert_eq!(interface.ports.len(), 2);
    }

    #[test]
    fn parses_bundle_interface() {
        let interface: InterfaceSpec = syn::parse2(quote! {
            Bus {
                input req: Req {
                    valid: uint(1),
                    addr: uint(8),
                },
                output resp: Resp {
                    data: uint(8),
                },
            }
        })
        .unwrap();

        assert_eq!(interface.name.to_string(), "Bus");
        assert_eq!(interface.ports.len(), 2);
    }

    #[test]
    fn rejects_empty_interface() {
        let err = match syn::parse2::<InterfaceSpec>(quote! {
            Empty {}
        }) {
            Ok(_) => panic!("expected empty interface to fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("at least one port"));
    }

    #[test]
    fn rejects_unsupported_interface_direction() {
        let err = match syn::parse2::<InterfaceSpec>(quote! {
            Bus {
                sink req: uint(1),
            }
        }) {
            Ok(_) => panic!("expected unsupported interface direction to fail"),
            Err(err) => err,
        };

        assert!(err
            .to_string()
            .contains("expected interface port direction"));
    }

    #[test]
    fn parses_external_module_declaration() {
        let extern_module: ExternModuleSpec = syn::parse2(quote! {
            design IOBUF {
                inout PAD: uint(1),
                input I: uint(1),
                input T: uint(1),
                output O: uint(1),
            }
        })
        .unwrap();

        assert_eq!(extern_module.design.to_string(), "design");
        assert_eq!(extern_module.name.to_string(), "IOBUF");
        assert_eq!(extern_module.ports.len(), 4);
    }

    #[test]
    fn rejects_empty_external_module_declaration() {
        let err = match syn::parse2::<ExternModuleSpec>(quote! {
            design EmptyVendor {}
        }) {
            Ok(_) => panic!("expected empty external module declaration to fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("at least one port"));
    }

    #[test]
    fn rejects_unsupported_external_port_direction() {
        let err = match syn::parse2::<ExternModuleSpec>(quote! {
            design Vendor {
                clock clk: uint(1),
            }
        }) {
            Ok(_) => panic!("expected unsupported external port direction to fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("expected external port direction"));
    }

    #[test]
    fn parses_instance_declarations() {
        let instances: InstancesSpec = syn::parse2(quote! {
            m {
                instance u_child: Child {
                    a: a,
                    y: y,
                },
                instance_bundles u_bundle: BundleChild {
                    req: req,
                    resp: resp,
                },
                instance_interfaces u_bus: BusChild {
                    bus: bus,
                },
            }
        })
        .unwrap();

        assert_eq!(instances.builder.to_string(), "m");
        assert_eq!(instances.stmts.len(), 3);
    }

    #[test]
    fn rejects_empty_instance_declarations() {
        let err = match syn::parse2::<InstancesSpec>(quote! {
            m {}
        }) {
            Ok(_) => panic!("expected empty instances! block to fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("at least one instance"));
    }

    #[test]
    fn rejects_unsupported_instance_statement_kind() {
        let err = match syn::parse2::<InstancesSpec>(quote! {
            m {
                connect u_child: Child {
                    a: a,
                },
            }
        }) {
            Ok(_) => panic!("expected unsupported instance kind to fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("expected instance statement kind"));
    }

    #[test]
    fn rejects_empty_instance_connection_list() {
        let err = match syn::parse2::<InstancesSpec>(quote! {
            m {
                instance u_child: Child {},
            }
        }) {
            Ok(_) => panic!("expected empty instance connection list to fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("at least one connection"));
    }

    #[test]
    fn parses_scalar_signal_declarations() {
        let signals: SignalsSpec = syn::parse2(quote! {
            m {
                input clk: uint(1),
                output data: sint(8),
                wire tmp: uint(8),
                reg count: uint(8),
            }
        })
        .unwrap();

        assert_eq!(signals.builder.to_string(), "m");
        assert_eq!(signals.decls.len(), 4);
    }

    #[test]
    fn parses_structured_signal_declarations() {
        let signals: SignalsSpec = syn::parse2(quote! {
            m {
                input_bundle req: Req {
                    valid: uint(1),
                },
                output_bundle resp: Resp {
                    data: uint(8),
                },
                interface bus: Bus {
                    input ready: uint(1),
                    output data: uint(8),
                },
            }
        })
        .unwrap();

        assert_eq!(signals.builder.to_string(), "m");
        assert_eq!(signals.decls.len(), 3);
    }

    #[test]
    fn parses_memory_signal_declaration() {
        let signals: SignalsSpec = syn::parse2(quote! {
            m {
                input clk: uint(1),
                mem regs: addr(2), data uint(8), depth 4,
            }
        })
        .unwrap();

        assert_eq!(signals.builder.to_string(), "m");
        assert_eq!(signals.decls.len(), 2);
    }

    #[test]
    fn parses_ready_valid_signal_declarations() {
        let signals: SignalsSpec = syn::parse2(quote! {
            m {
                rv_sink input: scalar uint(8),
                rv_source output: bundle Payload {
                    data: uint(8),
                    last: uint(1),
                },
            }
        })
        .unwrap();

        assert_eq!(signals.builder.to_string(), "m");
        assert_eq!(signals.decls.len(), 2);
    }

    #[test]
    fn rejects_memory_signal_declaration_without_addr_keyword() {
        let err = match syn::parse2::<SignalsSpec>(quote! {
            m {
                mem regs: data uint(8), depth 4,
            }
        }) {
            Ok(_) => panic!("expected malformed memory declaration to fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("expected memory addr"));
    }

    #[test]
    fn rejects_empty_signal_declarations() {
        let err = match syn::parse2::<SignalsSpec>(quote! {
            m {}
        }) {
            Ok(_) => panic!("expected empty signals! declaration list to fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("at least one declaration"));
    }

    #[test]
    fn rejects_unsupported_signal_declaration_kind() {
        let err = match syn::parse2::<SignalsSpec>(quote! {
            m {
                sink clk: uint(1),
            }
        }) {
            Ok(_) => panic!("expected unsupported signal declaration kind to fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("expected signal declaration kind"));
    }

    #[test]
    fn parses_logic_statements() {
        let logic: LogicSpec = syn::parse2(quote! {
            m {
                clock count: clk;
                reset count: rst = 0;
                reset_low low_count: rst_n = 0;
                async_reset async_count: arst = 0;
                async_reset_low async_low_count: arst_n = 0;
                next count = mux(en, count + lit_u(1, 8), count);
                state_next_hold flow {
                    start.value() => Run,
                    done.value() => Done,
                };
                state_next flow default Idle {
                    start.value() => Run,
                    done.value() => Done,
                };
                mem_write regs: clk, we, waddr, wdata;
                assert count_small: count.value().lt_expr(lit_u(10, 8));
                assert_when accepted_small: accepted, count.value().lt_expr(lit_u(10, 8));
                assert_clocked count_checked: clk, count.value().lt_expr(lit_u(10, 8));
                assert_msg count_message: count.value().lt_expr(lit_u(10, 8)), "count too large";
                cover count_ten: count.value().eq_expr(lit_u(10, 8));
                cover_when accepted_ten: accepted, count.value().eq_expr(lit_u(10, 8));
                cover_clocked count_ten_clocked: clk, count.value().eq_expr(lit_u(10, 8));
                cover_msg count_ten_message: count.value().eq_expr(lit_u(10, 8)), "count reached ten";
                assign out = count;
                assign_bundle resp = req;
                assign_bundle_when next_resp: accepted, req;
            }
        })
        .unwrap();

        assert_eq!(logic.builder.to_string(), "m");
        assert_eq!(logic.stmts.len(), 20);
    }

    #[test]
    fn parses_mem_read_expression_macro() {
        let read: MemReadSpec = syn::parse2(quote! {
            m, regs, raddr + lit_u(1, 2)
        })
        .unwrap();

        assert_eq!(read.builder.to_string(), "m");
        assert_eq!(read.mem.to_string(), "regs");
    }

    #[test]
    fn parses_ready_valid_helper_statements() {
        let ready_valid: ReadyValidSpec = syn::parse2(quote! {
            m {
                connect input => output;
                register_slice slice: input => output, clk, rst;
                skid_buffer skid: input => output, clk, rst;
                fifo fifo: input => output, clk, rst, depth 3;
                mem_fifo mem_fifo: input => output, clk, rst, depth 4;
            }
        })
        .unwrap();

        assert_eq!(ready_valid.builder.to_string(), "m");
        assert_eq!(ready_valid.stmts.len(), 5);
    }

    #[test]
    fn rejects_unsupported_ready_valid_helper_kind() {
        let err = match syn::parse2::<ReadyValidSpec>(quote! {
            m {
                pipe slice: input => output, clk, rst;
            }
        }) {
            Ok(_) => panic!("expected unsupported ready/valid helper kind to fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("expected ready/valid helper kind"));
    }

    #[test]
    fn rejects_empty_logic_block() {
        let err = match syn::parse2::<LogicSpec>(quote! {
            m {}
        }) {
            Ok(_) => panic!("expected empty logic! block to fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("at least one statement"));
    }

    #[test]
    fn rejects_unsupported_logic_statement_kind() {
        let err = match syn::parse2::<LogicSpec>(quote! {
            m {
                drive out = count;
            }
        }) {
            Ok(_) => panic!("expected unsupported logic statement kind to fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("expected logic statement kind"));
    }

    #[test]
    fn rejects_empty_state_transition_block() {
        let err = match syn::parse2::<LogicSpec>(quote! {
            m {
                state_next_hold state {};
            }
        }) {
            Ok(_) => panic!("expected empty state transition block to fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("at least one transition"));
    }
}
