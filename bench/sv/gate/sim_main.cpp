
#include "Vgates.h"
#include "verilated.h"
#include <cstdio>
#include <chrono>
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    long cycles = argc > 1 ? atol(argv[1]) : 20000000;
    Vgates* dut = new Vgates;
    dut->a = 1; dut->b = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (long i = 0; i < cycles; i++) {
        dut->clk = 0; dut->eval();
        dut->clk = 1; dut->eval();
        dut->a = (i & 1); dut->b = ((i >> 1) & 1); // vary stimulus
    }
    auto t1 = std::chrono::steady_clock::now();
    double s = std::chrono::duration<double>(t1 - t0).count();
    printf("Verilator gates: %ld cycles, %.1f Mcyc/s, o[0]=%d\n", cycles, cycles/s/1e6, (int)(dut->o & 1));
    delete dut;
    return 0;
}
