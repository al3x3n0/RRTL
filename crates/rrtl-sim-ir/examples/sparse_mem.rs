//! Sparse / lazy huge memory for the canonical RTL case: a CPU's address space.
//! A 32-bit space is 4 GB — a dense array can't hold it — yet a program touches a
//! handful of pages (code low, stack high, a little MMIO). A sparse page table
//! allocates only the pages actually written, returns 0 elsewhere, and so models
//! the FULL address space in a few KB. Here picorv32 runs with code at address 0
//! and its stack near 1 GB; the dense harness (a flat Vec) would need a gigabyte.
//! Build: cargo run --release --features jit -p rrtl-sim-ir --example sparse_mem -- bench/sv/picorv32.v
fn main() {
    #[cfg(not(feature = "jit"))]
    println!("build with --features jit");
    #[cfg(feature = "jit")]
    run();
}

#[cfg(feature = "jit")]
const PAGE_WORDS: usize = 1024; // 4 KiB pages (1024 × 32-bit words)

/// Demand-paged sparse memory over the full 32-bit (word-addressed) space: a page
/// is allocated on first write; unwritten addresses read as 0.
#[cfg(feature = "jit")]
#[derive(Default)]
struct SparseMem {
    pages: std::collections::HashMap<u32, Box<[u32; PAGE_WORDS]>>,
}
#[cfg(feature = "jit")]
impl SparseMem {
    fn read(&self, word: u32) -> u32 {
        let (pg, off) = (word / PAGE_WORDS as u32, (word % PAGE_WORDS as u32) as usize);
        self.pages.get(&pg).map_or(0, |p| p[off])
    }
    fn write(&mut self, word: u32, wstrb: u32, data: u32) {
        let (pg, off) = (word / PAGE_WORDS as u32, (word % PAGE_WORDS as u32) as usize);
        let p = self.pages.entry(pg).or_insert_with(|| Box::new([0u32; PAGE_WORDS]));
        let mut w = p[off];
        for b in 0..4 {
            if wstrb & (1 << b) != 0 {
                let sh = b * 8;
                w = (w & !(0xFFu32 << sh)) | (data & (0xFFu32 << sh));
            }
        }
        p[off] = w;
    }
    fn load_words(&mut self, base_word: u32, prog: &[u32]) {
        for (i, &w) in prog.iter().enumerate() {
            self.write(base_word + i as u32, 0xF, w);
        }
    }
    fn ram_bytes(&self) -> usize {
        self.pages.len() * PAGE_WORDS * 4
    }
}

#[cfg(feature = "jit")]
fn run() {
    use rrtl_sim_ir::jit::JitSimulator;
    use rrtl_sim_ir::{lower_to_machine_program, lower_to_packed_program};
    use rrtl_sv_frontend::import_sv;

    let path = std::env::args().nth(1).unwrap_or_else(|| "bench/sv/picorv32.v".into());
    let src = std::fs::read_to_string(&path).expect("read picorv32.v");
    let imported = import_sv(&src, Some("picorv32")).unwrap();
    let compiled = rrtl_core::compile(&imported.design).unwrap();
    let program = lower_to_packed_program(&compiled, "picorv32").unwrap();
    let machine = lower_to_machine_program(&program);
    let idx = |n: &str| {
        let h = compiled.find_module("picorv32").unwrap().signals.iter().find(|s| s.name == n).unwrap().handle;
        program.signal_index(h).unwrap()
    };
    let (clk, resetn, mr, rd, mv, ma, mw, md, trap) = (
        idx("clk"), idx("resetn"), idx("mem_ready"), idx("mem_rdata"), idx("mem_valid"),
        idx("mem_addr"), idx("mem_wstrb"), idx("mem_wdata"), idx("trap"),
    );

    // Stack near 1 GB, code at 0, result to a low address — so the touched pages
    // are far apart and a dense array would have to span the whole gap.
    let prog = rrtl_riscv_asm::assemble(
        "
        li   sp, 0x40000000  # sp (=x2) = ~1 GiB; keep x2 reserved as sp
        li   x5, 0           # sum
        li   x6, 1           # i
        li   x7, 6           # limit
    loop:
        addi sp, sp, -4
        sw   x6, 0(sp)       # push i (high-address writes)
        add  x5, x5, x6
        addi x6, x6, 1
        blt  x6, x7, loop
        lw   x8, 0(sp)       # pop top (high-address read = 5)
        add  x5, x5, x8      # 15 + 5
        sw   x5, 0x40(x0)    # result -> mem[0x40]
    spin:
        j spin
        ",
    )
    .unwrap();

    let mut mem = SparseMem::default();
    mem.load_words(0, &prog);

    let mut jit = JitSimulator::compile(&machine).unwrap();
    let mut prev_ready = 0u64;
    let mut max_addr = 0u32;
    let mut result = None;
    for c in 0..20_000u32 {
        let resetn_v = (c >= 4) as u64;
        let valid = jit.get_signal(mv);
        let addr = jit.get_signal(ma) as u32;
        let wstrb = jit.get_signal(mw) as u32;
        let wdata = jit.get_signal(md) as u32;
        let ready = (valid != 0 && prev_ready == 0) as u64;
        let mut rdata = 0u64;
        if ready != 0 {
            max_addr = max_addr.max(addr);
            let word = addr >> 2;
            if wstrb != 0 {
                mem.write(word, wstrb, wdata);
                if addr == 0x40 {
                    result = Some(wdata);
                }
            } else {
                rdata = mem.read(word) as u64;
            }
        }
        prev_ready = ready;
        jit.set_signal(clk, 1);
        jit.set_signal(resetn, resetn_v);
        jit.set_signal(mr, ready);
        jit.set_signal(rd, rdata);
        jit.tick();
        assert!(jit.get_signal(trap) == 0, "trapped at cycle {c}");
        if result.is_some() && c > 200 {
            break;
        }
    }

    let dense_bytes = (max_addr as usize + 4).next_power_of_two();
    println!("sparse huge memory — picorv32, code @ 0, stack @ ~1 GiB");
    println!("  highest address touched : 0x{max_addr:08x} ({} MiB into the space)", max_addr as usize / (1 << 20));
    println!("  sparse pages allocated  : {} ({} KiB)", mem.pages.len(), mem.ram_bytes() / 1024);
    println!("  a dense array would need : ~{} MiB (to span up to the stack)", dense_bytes / (1 << 20));
    println!("  result mem[0x40]         : {:?} (want 20 = sum 1..5 + popped top 5)", result);
    println!("  => the full 32-bit space is modeled in {} KiB; only touched pages cost RAM.", mem.ram_bytes() / 1024);
    println!("     In a bulk-synchronous batch (§4g.13) each instance carries its own sparse memory,");
    println!("     so N CPUs in a 4 GiB space cost N×(touched pages), not N×4 GiB.");
    println!("  {}", if result == Some(20) { "[PASS] high-address stack + low code, bit-exact" } else { "[FAIL]" });
}
