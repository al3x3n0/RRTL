#!/usr/bin/env bash
# Apples-to-apples picorv32 throughput: RRTL JIT vs Verilator on the SAME core,
# program, memory protocol, and cycle count. The RRTL bench writes the shared
# program hex (picorv32_prog.hex); this script then verilates + runs Verilator.
#
# Usage: bench/sv/run_picorv32.sh [CYCLES]   (default 5,000,000)
set -euo pipefail
cd "$(dirname "$0")"
CYCLES="${1:-5000000}"

echo "== RRTL JIT (Cranelift) =="
( cd ../.. && cargo run --release --quiet --features jit -p rrtl-sim-ir \
    --example picorv32_bench -- "$CYCLES" )

echo
echo "== RRTL AOT (clang -O3) =="
( cd ../.. && cargo run --release --quiet --features aot -p rrtl-sim-ir \
    --example picorv32_aot -- "$CYCLES" )

if [ ! -f picorv32_prog.hex ]; then
  echo "picorv32_prog.hex missing (run the RRTL bench first)"; exit 1
fi

echo
echo "== Verilator (5.x) =="
# -O3 / --x-assign 0 to match RRTL's 2-state semantics; -Wno-* to silence the
# core's UNOPTFLAT/WIDTH style warnings (picorv32 is intentionally written that way).
# -j 2 (not -j 0=all-cores): the generated picorv32 C++ compiled at -O3 by many
# parallel g++ processes OOMs on memory-constrained machines; 2 jobs is safe.
verilator --cc --exe --build -j 2 -O3 \
  --x-assign 0 --x-initial 0 \
  -Wno-fatal -Wno-WIDTH -Wno-UNOPTFLAT -Wno-CASEINCOMPLETE -Wno-UNUSED -Wno-PINMISSING \
  --top-module picorv32 \
  -CFLAGS "-O3" \
  picorv32.v picorv32_tb.cpp >/dev/null 2>&1 || \
  verilator --cc --exe --build -j 0 -O3 --x-assign 0 --x-initial 0 \
    -Wno-fatal -Wno-WIDTH -Wno-UNOPTFLAT -Wno-CASEINCOMPLETE -Wno-UNUSED -Wno-PINMISSING \
    --top-module picorv32 -CFLAGS "-O3" picorv32.v picorv32_tb.cpp

./obj_dir/Vpicorv32 "$CYCLES" picorv32_prog.hex
