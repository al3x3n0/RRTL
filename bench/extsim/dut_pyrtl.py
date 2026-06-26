"""PyRTL FastSimulation side of the cross-simulator benchmark. Same design and
stimulus as dut.v / extsim_compare.rs; prints `out` (cross-check) and cycle rate.
Usage: python dut_pyrtl.py [W D N]"""
import sys
import time

import pyrtl

W = int(sys.argv[1]) if len(sys.argv) > 1 else 16
D = int(sys.argv[2]) if len(sys.argv) > 2 else 8
N = int(sys.argv[3]) if len(sys.argv) > 3 else 500000
C = 0x9E3779B9

din = pyrtl.Input(32, "din")
accs = []
for i in range(W):
    acc = pyrtl.Register(32, f"acc{i}")
    t = acc
    for _ in range(D):
        t = (t * C + din)[:32]          # 32-bit wrap, same as Verilog/RRTL
    acc.next <<= (t + acc + i)[:32]   # per-channel salt
    accs.append(acc)

x = accs[0]
for a in accs[1:]:
    x = x ^ a
out = pyrtl.Output(32, "out")
out <<= x

sim = pyrtl.FastSimulation()
t0 = time.perf_counter()
for c in range(N):
    sim.step({"din": (c * 2654435761) & 0xFFFFFFFF})
sec = time.perf_counter() - t0

# PyRTL's registered output lags one step (RRTL/Verilator re-settle comb after
# commit). One extra flush step — outside the timer — exposes the Nth update so
# `out` matches the others. The N timed steps did N register updates regardless.
sim.step({"din": 0})
val = sim.inspect("out")
print(f"pyrtl out={val} cycles={N} {N / sec / 1e6:.3f} Mcyc/s {sec * 1e3:.1f} ms")
