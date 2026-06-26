#!/usr/bin/env bash
# Cross-simulator benchmark: RRTL vs Verilator vs PyRTL FastSimulation on an
# identical design + stimulus. Prints each tool's `out` (must match) and rate.
# Usage: ./run.sh [W D N]
set -e
W=${1:-16}; D=${2:-8}; N=${3:-1000000}; PYN=${4:-20000}
HERE="$(cd "$(dirname "$0")" && pwd)"
RRTL_ROOT="$(cd "$HERE/../.." && pwd)"
VENV=/tmp/rrtl_bench_venv

echo "=== design W=$W D=$D, N=$N cycles, stimulus din[c]=c*2654435761 mod 2^32 ==="

echo "--- Verilator $(verilator --version | awk '{print $2}') (-O3 -march=native) ---"
cd "$HERE"
rm -rf obj_dir
# Verilate then make with OPT_FAST overridden to -O3 (Verilator's default is -Os,
# size not speed). The DUT is UNOPTFLAT-free, so eval() is single-pass.
verilator --cc --exe -O3 -Wno-WIDTH -GW="$W" -GD="$D" \
  --top-module dut --Mdir obj_dir -o vsim dut.v sim_main.cpp > /tmp/verilate.log 2>&1 \
  || { echo "verilate failed:"; tail -20 /tmp/verilate.log; exit 1; }
make -C obj_dir -f Vdut.mk OPT_FAST="-O3 -march=native" -j8 vsim > /tmp/vmake.log 2>&1 \
  || { echo "build failed:"; tail -20 /tmp/vmake.log; exit 1; }
./obj_dir/vsim "$N"

echo "--- PyRTL FastSimulation (N=$PYN; rate-only, it is ~100x slower) ---"
"$VENV/bin/python" "$HERE/dut_pyrtl.py" "$W" "$D" "${PYN:-$N}"

echo "--- RRTL (Cranelift JIT) ---"
cd "$RRTL_ROOT"
cargo run --release --quiet --features jit -p rrtl-sim-ir --example extsim_compare -- "$W" "$D" "$N" 1024
