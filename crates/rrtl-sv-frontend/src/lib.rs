use rrtl_ir::{Diagnostic, ErrorReport};

mod ast;
mod lower;
pub mod netlist;
mod parser;
mod preprocess;

pub use ast::*;
pub use lower::import_source;
pub use netlist::import_yosys_netlist;

use lower::specialize_source;
use parser::Parser;
use preprocess::preprocess;

/// Maximum static unroll for `for`/generate loops.
pub(crate) const SV_FOR_UNROLL_LIMIT: usize = 4096;

pub fn parse_sv(source: &str) -> Result<SvSource, ErrorReport> {
    let pp = preprocess(source)?;
    Parser::new(&pp)?.parse_source()
}

pub fn import_sv(source: &str, top: Option<&str>) -> Result<SvImport, ErrorReport> {
    let pp = preprocess(source)?;
    let parsed = Parser::new(&pp)?.parse_source()?;
    import_source(specialize_source(&pp, parsed)?, top)
}

pub(crate) fn err(code: &'static str, message: impl Into<String>) -> ErrorReport {
    ErrorReport::new(vec![Diagnostic::new(code, message)])
}

#[cfg(test)]
mod tests;
