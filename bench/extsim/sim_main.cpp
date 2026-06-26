// Verilator C++ harness for the cross-simulator benchmark. Drives N clock
// cycles with din[c] = c*2654435761 (mod 2^32), then prints `out` (for the
// cross-check) and the cycle rate. One cycle = clk low eval + clk high eval
// (the posedge that captures the registers) — the standard edge-accurate loop.
#include "Vdut.h"
#include "verilated.h"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    long n = argc > 1 ? atol(argv[1]) : 500000;
    Verilated::commandArgs(argc, argv);
    Vdut* top = new Vdut;

    // Initialize (Verilator 2-state zero-inits regs by default).
    top->clk = 0;
    top->din = 0;
    top->eval();

    // warmup + best-of-3 (matches the RRTL harness for a fair comparison)
    double best = 1e30;
    for (int rep = 0; rep < 4; rep++) {
        auto t0 = std::chrono::steady_clock::now();
        for (long c = 0; c < n; c++) {
            top->din = (uint32_t)((uint64_t)c * 2654435761ull);
            top->clk = 0;
            top->eval();
            top->clk = 1;
            top->eval();
        }
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        if (rep > 0 && sec < best) best = sec;  // rep 0 = warm
    }
    printf("verilator out=%u cycles=%ld %.2f Mcyc/s %.1f ms\n",
           (unsigned)top->out, n, n / best / 1e6, best * 1e3);
    delete top;
    return 0;
}
