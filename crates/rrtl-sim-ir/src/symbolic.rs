//! Symbolic execution of gate-level RTL via BDDs.
//!
//! The bit-parallel engine (`bitparallel.rs`) advances 64 *concrete* lanes per
//! u64. Replace each signal's concrete value with a **reduced ordered binary
//! decision diagram** (ROBDD) — a canonical boolean function of symbolic input
//! variables — and the very same gate schedule becomes a *symbolic* simulation:
//! every output signal ends up as the boolean function it computes over the
//! symbolic inputs. The gate ops (And/Or/Xor/Not/Mux) map 1:1 onto BDD `apply`,
//! so the engine mirrors `BitParallelSimulator` structurally, one BDD per signal
//! instead of `ceil(L/64)` u64 words.
//!
//! This turns the simulator into a lightweight formal engine:
//!   * **ATPG** — a stuck-at fault's *detection function* is `OR_o(golden_o XOR
//!     faulty_o)`; a satisfying assignment is a test vector, and the *false* BDD
//!     is a proof the fault is untestable (redundant) — something simulation can
//!     never conclude, only fail to disprove.
//!   * **Equivalence / property checking** — two cones are equal iff the XOR of
//!     their outputs is the false BDD.
//!
//! Scope matches the bit-parallel engine: all-1-bit gate-level designs
//! (And/Or/Xor/Not/Mux/Signal/Lit, regs with optional reset). It is exponential
//! in the worst case (BDDs are); it targets bounded cones, not unbounded designs.

use std::collections::HashMap;

use rrtl_ir::{Diagnostic, ErrorReport, ResetPolarity};

use crate::{PackedBlock, PackedEffect, PackedInstrKind, PackedMachineProgram, PackedReset};

fn sym_err(msg: impl Into<String>) -> ErrorReport {
    ErrorReport::new(vec![Diagnostic::new("E_SYMBOLIC", msg.into())])
}

/// A BDD node handle. 0 = constant false, 1 = constant true; higher indices are
/// decision nodes in the manager's `nodes` table.
pub type Bdd = u32;
pub const FALSE: Bdd = 0;
pub const TRUE: Bdd = 1;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Node {
    var: u32, // decision variable; u32::MAX for the two terminals
    lo: Bdd,  // cofactor when var = 0
    hi: Bdd,  // cofactor when var = 1
}

/// A reduced, ordered, hash-consed BDD manager (variables ordered by index —
/// smaller index nearer the root).
#[derive(Default)]
pub struct BddManager {
    nodes: Vec<Node>,
    unique: HashMap<(u32, Bdd, Bdd), Bdd>,
    ite_cache: HashMap<(Bdd, Bdd, Bdd), Bdd>,
}

impl BddManager {
    pub fn new() -> Self {
        let term = Node { var: u32::MAX, lo: 0, hi: 0 };
        BddManager { nodes: vec![term, term], unique: HashMap::new(), ite_cache: HashMap::new() }
    }

    fn var_of(&self, f: Bdd) -> u32 {
        self.nodes[f as usize].var
    }

    /// Hash-consed node constructor with reduction (`lo == hi` collapses).
    fn mk(&mut self, var: u32, lo: Bdd, hi: Bdd) -> Bdd {
        if lo == hi {
            return lo;
        }
        if let Some(&n) = self.unique.get(&(var, lo, hi)) {
            return n;
        }
        let id = self.nodes.len() as Bdd;
        self.nodes.push(Node { var, lo, hi });
        self.unique.insert((var, lo, hi), id);
        id
    }

    /// The BDD for a single variable `i` (true iff variable `i` is 1).
    pub fn var(&mut self, i: u32) -> Bdd {
        self.mk(i, FALSE, TRUE)
    }
    pub fn constant(&mut self, b: bool) -> Bdd {
        if b {
            TRUE
        } else {
            FALSE
        }
    }

    /// Shannon cofactor of `f` w.r.t. `var = bit`, only descending nodes whose
    /// top variable *is* `var` (others are independent of it).
    fn cofactor(&self, f: Bdd, var: u32, bit: bool) -> Bdd {
        let n = self.nodes[f as usize];
        if n.var == var {
            if bit {
                n.hi
            } else {
                n.lo
            }
        } else {
            f
        }
    }

    /// If-then-else: `f ? g : h`. The universal BDD operation.
    pub fn ite(&mut self, f: Bdd, g: Bdd, h: Bdd) -> Bdd {
        // Terminal / trivial cases.
        if f == TRUE {
            return g;
        }
        if f == FALSE {
            return h;
        }
        if g == h {
            return g;
        }
        if g == TRUE && h == FALSE {
            return f;
        }
        if let Some(&r) = self.ite_cache.get(&(f, g, h)) {
            return r;
        }
        // Split on the top variable among f, g, h.
        let v = self.var_of(f).min(self.var_of(g)).min(self.var_of(h));
        let (fl, fh) = (self.cofactor(f, v, false), self.cofactor(f, v, true));
        let (gl, gh) = (self.cofactor(g, v, false), self.cofactor(g, v, true));
        let (hl, hh) = (self.cofactor(h, v, false), self.cofactor(h, v, true));
        let lo = self.ite(fl, gl, hl);
        let hi = self.ite(fh, gh, hh);
        let r = self.mk(v, lo, hi);
        self.ite_cache.insert((f, g, h), r);
        r
    }

    pub fn not(&mut self, f: Bdd) -> Bdd {
        self.ite(f, FALSE, TRUE)
    }
    pub fn and(&mut self, f: Bdd, g: Bdd) -> Bdd {
        self.ite(f, g, FALSE)
    }
    pub fn or(&mut self, f: Bdd, g: Bdd) -> Bdd {
        self.ite(f, TRUE, g)
    }
    pub fn xor(&mut self, f: Bdd, g: Bdd) -> Bdd {
        let ng = self.not(g);
        self.ite(f, ng, g)
    }

    pub fn is_false(&self, f: Bdd) -> bool {
        f == FALSE
    }
    pub fn is_true(&self, f: Bdd) -> bool {
        f == TRUE
    }

    /// One satisfying assignment (variable → bit) for `f`, or `None` if `f` is the
    /// false BDD. Variables not returned are don't-cares.
    pub fn sat_one(&self, f: Bdd) -> Option<Vec<(u32, bool)>> {
        if f == FALSE {
            return None;
        }
        let mut out = Vec::new();
        let mut cur = f;
        while cur != TRUE {
            let n = self.nodes[cur as usize];
            // Prefer the hi branch (var = 1) when it can still reach true.
            if n.hi != FALSE {
                out.push((n.var, true));
                cur = n.hi;
            } else {
                out.push((n.var, false));
                cur = n.lo;
            }
        }
        Some(out)
    }

    /// Evaluate `f` under a total assignment `assign(var) -> bit`.
    pub fn eval(&self, f: Bdd, assign: impl Fn(u32) -> bool) -> bool {
        let mut cur = f;
        while cur > TRUE {
            let n = self.nodes[cur as usize];
            cur = if assign(n.var) { n.hi } else { n.lo };
        }
        cur == TRUE
    }

    /// Number of distinct decision nodes reachable from `f` (BDD size).
    pub fn size(&self, f: Bdd) -> usize {
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![f];
        while let Some(x) = stack.pop() {
            if x <= TRUE || !seen.insert(x) {
                continue;
            }
            let n = self.nodes[x as usize];
            stack.push(n.lo);
            stack.push(n.hi);
        }
        seen.len()
    }
}

/// A symbolic gate-level simulator: one BDD per signal, evaluated on the same
/// packed stream schedule as [`crate::bitparallel::BitParallelSimulator`].
pub struct SymbolicSimulator {
    machine: PackedMachineProgram,
    pub mgr: BddManager,
    /// BDD per signal, indexed by signal position (== `signal_index`).
    state: Vec<Bdd>,
    /// Stuck-at / clamped signals `signal -> BDD`, re-applied after every store
    /// and after each commit so the value propagates (comb-wire or register).
    forces: HashMap<usize, Bdd>,
}

impl SymbolicSimulator {
    pub fn new(machine: &PackedMachineProgram) -> Result<Self, ErrorReport> {
        for s in &machine.source.signals {
            if s.layout.width != 1 {
                return Err(sym_err(format!(
                    "signal `{}` is {} bits — symbolic engine needs 1-bit gate-level signals",
                    s.name, s.layout.width
                )));
            }
        }
        if !machine.source.memories.is_empty() {
            return Err(sym_err("memories are not supported by the symbolic engine"));
        }
        for blk in Self::streams(machine) {
            for pkt in &blk.packets {
                for instr in &pkt.instrs {
                    use PackedInstrKind::*;
                    match &instr.kind {
                        And(..) | Or(..) | Xor(..) | Not(..) | Mux { .. } | Signal(_) | Lit(_) => {}
                        other => {
                            return Err(sym_err(format!(
                                "op {:?} is not a gate-level bitwise op (symbolic scope)",
                                std::mem::discriminant(other)
                            )))
                        }
                    }
                }
            }
        }
        let n = machine.source.signals.len();
        Ok(Self {
            machine: machine.clone(),
            mgr: BddManager::new(),
            state: vec![FALSE; n],
            forces: HashMap::new(),
        })
    }

    fn streams(machine: &PackedMachineProgram) -> [&PackedBlock; 4] {
        [
            &machine.streams.async_reset_comb,
            &machine.streams.comb,
            &machine.streams.tick_next,
            &machine.streams.tick_commit,
        ]
    }

    /// Assign a signal (input or initial register) a fresh symbolic variable.
    pub fn set_var(&mut self, signal: usize, var: u32) {
        let v = self.mgr.var(var);
        self.state[signal] = v;
    }
    /// Assign a signal a constant.
    pub fn set_const(&mut self, signal: usize, value: bool) {
        self.state[signal] = self.mgr.constant(value);
    }
    /// Set a signal to an arbitrary BDD.
    pub fn set_bdd(&mut self, signal: usize, f: Bdd) {
        self.state[signal] = f;
    }
    pub fn get(&self, signal: usize) -> Bdd {
        self.state[signal]
    }

    /// Clamp `signal` to a constant every cycle (stuck-at fault). Propagates
    /// through the settle (comb-wire or register).
    pub fn set_force_const(&mut self, signal: usize, value: bool) {
        let b = self.mgr.constant(value);
        self.forces.insert(signal, b);
    }
    pub fn clear_forces(&mut self) {
        self.forces.clear();
    }
    fn clamp(&mut self, signal: usize) {
        if let Some(&f) = self.forces.get(&signal) {
            self.state[signal] = f;
        }
    }

    fn nvals(block: &PackedBlock) -> usize {
        block.packets.iter().flat_map(|p| p.instrs.iter()).map(|i| i.dst.0 + 1).max().unwrap_or(0)
    }

    fn eval_packet(&mut self, pkt: &crate::PackedMachinePacket, work: &mut [Bdd]) {
        for instr in &pkt.instrs {
            let d = instr.dst.0;
            use PackedInstrKind::*;
            work[d] = match &instr.kind {
                Signal(s) => self.state[*s],
                Lit(w) => self.mgr.constant(w.first().copied().unwrap_or(0) & 1 == 1),
                Not(a) => {
                    let a = work[a.0];
                    self.mgr.not(a)
                }
                And(a, b) => {
                    let (a, b) = (work[a.0], work[b.0]);
                    self.mgr.and(a, b)
                }
                Or(a, b) => {
                    let (a, b) = (work[a.0], work[b.0]);
                    self.mgr.or(a, b)
                }
                Xor(a, b) => {
                    let (a, b) = (work[a.0], work[b.0]);
                    self.mgr.xor(a, b)
                }
                Mux { cond, then_value, else_value } => {
                    let (c, t, e) = (work[cond.0], work[then_value.0], work[else_value.0]);
                    self.mgr.ite(c, t, e)
                }
                _ => unreachable!("validated in new()"),
            };
        }
    }

    fn reset_asserted(&mut self, reset: &PackedReset) -> Bdd {
        let base = self.state[reset.signal];
        match reset.polarity {
            ResetPolarity::ActiveHigh => base,
            ResetPolarity::ActiveLow => self.mgr.not(base),
        }
    }

    /// Combinational settle (async-reset-comb then comb), storing comb signals and
    /// re-applying any forces so stuck-at wires propagate.
    pub fn settle(&mut self) {
        let blocks = [self.machine.streams.async_reset_comb.clone(), self.machine.streams.comb.clone()];
        for block in blocks {
            let nvals = Self::nvals(&block);
            let mut work = vec![FALSE; nvals];
            for pkt in &block.packets {
                self.eval_packet(pkt, &mut work);
                for eff in &pkt.effects {
                    match eff {
                        PackedEffect::StoreSignal { dst, value } => {
                            self.state[*dst] = work[value.0];
                            self.clamp(*dst);
                        }
                        PackedEffect::CaptureReg { dst, value, reset } => {
                            if let Some(r) = reset {
                                let asserted = self.reset_asserted(r);
                                let rv = self.mgr.constant(r.value.first().copied().unwrap_or(0) & 1 == 1);
                                let nv = work[value.0];
                                self.state[*dst] = self.mgr.ite(asserted, rv, nv);
                                self.clamp(*dst);
                            }
                        }
                        PackedEffect::MemoryWrite { .. } => {}
                    }
                }
            }
        }
    }

    /// One clock tick: settle, capture register next-states, commit, re-clamp
    /// forces, settle.
    pub fn tick(&mut self) {
        self.settle();
        let block = self.machine.streams.tick_next.clone();
        let nvals = Self::nvals(&block);
        let mut work = vec![FALSE; nvals];
        for pkt in &block.packets {
            self.eval_packet(pkt, &mut work);
        }
        let mut next: Vec<(usize, Bdd)> = Vec::new();
        for pkt in &block.packets {
            for eff in &pkt.effects {
                if let PackedEffect::CaptureReg { dst, value, reset } = eff {
                    let mut nv = work[value.0];
                    if let Some(r) = reset {
                        let asserted = self.reset_asserted(r);
                        let rv = self.mgr.constant(r.value.first().copied().unwrap_or(0) & 1 == 1);
                        nv = self.mgr.ite(asserted, rv, nv);
                    }
                    next.push((*dst, nv));
                }
            }
        }
        for (dst, nv) in next {
            self.state[dst] = nv;
        }
        for &sig in self.forces.keys().copied().collect::<Vec<_>>().iter() {
            self.clamp(sig);
        }
        self.settle();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{lower_to_machine_program, lower_to_packed_program};
    use rrtl_core::{compile, uint, Design};

    #[test]
    fn bdd_core_ops_and_sat() {
        let mut m = BddManager::new();
        let a = m.var(0);
        let b = m.var(1);
        let ab = m.and(a, b);
        // a & b is satisfiable only with a=1,b=1.
        let sol = m.sat_one(ab).unwrap();
        assert!(m.eval(ab, |v| sol.iter().find(|(x, _)| *x == v).map(|(_, x)| *x).unwrap_or(false)));
        // (a & b) is false when a=0.
        assert!(!m.eval(ab, |v| v == 1));
        // a ^ a = false; a | !a = true (canonicity).
        let na = m.not(a);
        let axa = m.xor(a, a);
        assert!(m.is_false(axa));
        let taut = m.or(a, na);
        assert!(m.is_true(taut));
    }

    // Symbolic simulation of a combinational gate cone must agree with a concrete
    // run on EVERY assignment of the symbolic inputs (exhaustive equivalence).
    #[test]
    fn symbolic_matches_concrete_exhaustively() {
        let mut design = Design::new();
        {
            let mut d = design.module("Cone");
            let a = d.input("a", uint(1));
            let b = d.input("b", uint(1));
            let c = d.input("c", uint(1));
            let w = d.wire("w", uint(1));
            let y = d.output("y", uint(1));
            // y = (a & b) ^ (b | c)
            d.assign(w, a.value() & b.value());
            d.assign(y, w.value() ^ (b.value() | c.value()));
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Cone").unwrap();
        let machine = lower_to_machine_program(&program);
        let h = |n: &str| compiled.find_module("Cone").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle;
        let idx = |n: &str| program.signal_index(h(n)).unwrap();

        let mut sym = SymbolicSimulator::new(&machine).unwrap();
        sym.set_var(idx("a"), 0);
        sym.set_var(idx("b"), 1);
        sym.set_var(idx("c"), 2);
        sym.settle();
        let y = sym.get(idx("y"));

        for bits in 0..8u32 {
            let assign = |v: u32| (bits >> v) & 1 == 1;
            let expect = {
                let (a, b, c) = (assign(0), assign(1), assign(2));
                (a & b) ^ (b | c)
            };
            assert_eq!(sym.mgr.eval(y, assign), expect, "mismatch at {bits:03b}");
        }
    }

    // Symbolic ATPG: a stuck-at on a REDUNDANT wire has a false detection BDD
    // (provably untestable), while a testable fault yields a satisfiable one.
    #[test]
    fn symbolic_atpg_proves_redundancy() {
        let mut design = Design::new();
        {
            let mut d = design.module("Redund");
            let a = d.input("a", uint(1));
            let b = d.input("b", uint(1));
            let nb = d.wire("nb", uint(1));
            let w = d.wire("w", uint(1));
            let v = d.wire("v", uint(1));
            let y = d.output("y", uint(1));
            d.assign(nb, !b.value()); // y = (a&b) | (a&~b) = a
            d.assign(w, a.value() & b.value());
            d.assign(v, a.value() & nb.value());
            d.assign(y, w.value() | v.value());
        }
        let compiled = compile(&design).unwrap();
        let program = lower_to_packed_program(&compiled, "Redund").unwrap();
        let machine = lower_to_machine_program(&program);
        let h = |n: &str| compiled.find_module("Redund").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle;
        let idx = |n: &str| program.signal_index(h(n)).unwrap();

        let detect = |signal: usize, stuck: bool| -> bool {
            let mut s = SymbolicSimulator::new(&machine).unwrap();
            s.set_var(idx("a"), 0);
            s.set_var(idx("b"), 1);
            s.settle();
            let golden = s.get(idx("y"));
            s.set_force_const(signal, stuck);
            s.settle();
            let faulty = s.get(idx("y"));
            let d = s.mgr.xor(golden, faulty);
            !s.mgr.is_false(d) // true = testable
        };
        assert!(!detect(idx("nb"), true), "nb stuck-at-1 is redundant → untestable");
        assert!(detect(idx("nb"), false), "nb stuck-at-0 is testable");
        assert!(detect(idx("w"), false), "w stuck-at-0 is testable");
    }
}
