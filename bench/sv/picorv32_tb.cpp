// Verilator throughput harness for picorv32, mirroring the RRTL JIT bench
// (crates/rrtl-sim-ir/examples/picorv32_bench.rs) exactly: same program (loaded
// from picorv32_prog.hex), same 1-cycle-pulse memory protocol, same cycle count.
// Reports Mcyc/s for an apples-to-apples comparison.
#include "Vpicorv32.h"
#include "verilated.h"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static const size_t MEMW = 4096; // words (power of two)

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    long n = (argc > 1) ? std::strtol(argv[1], nullptr, 10) : 5000000;
    const char* hex = (argc > 2) ? argv[2] : "picorv32_prog.hex";

    std::vector<uint32_t> mem(MEMW, 0);
    {
        std::ifstream f(hex);
        if (!f) { std::fprintf(stderr, "cannot open %s\n", hex); return 1; }
        std::string line;
        size_t i = 0;
        while (std::getline(f, line) && i < MEMW) {
            if (line.empty()) continue;
            mem[i++] = (uint32_t)std::strtoul(line.c_str(), nullptr, 16);
        }
    }

    Vpicorv32* top = new Vpicorv32;
    top->clk = 0;
    top->resetn = 0;
    top->mem_ready = 0;
    top->mem_rdata = 0;
    top->eval();

    uint64_t prev_ready = 0, fetches = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (long c = 0; c < n; c++) {
        int resetn = (c >= 4) ? 1 : 0;
        uint64_t valid = top->mem_valid;
        uint64_t addr = top->mem_addr;
        uint64_t wstrb = top->mem_wstrb;
        uint64_t wdata = top->mem_wdata;

        uint64_t ready = (valid && !prev_ready) ? 1 : 0;
        uint64_t rdata = 0;
        if (ready) {
            size_t widx = ((size_t)(addr >> 2)) & (MEMW - 1);
            if (wstrb) {
                uint32_t w = mem[widx];
                for (int b = 0; b < 4; b++)
                    if (wstrb & (1u << b)) {
                        int sh = b * 8;
                        w = (w & ~(0xFFu << sh)) | (((uint32_t)(wdata >> sh) & 0xFF) << sh);
                    }
                mem[widx] = w;
            } else {
                rdata = mem[widx];
                fetches++;
            }
        }
        prev_ready = ready;

        top->resetn = resetn;
        top->mem_ready = ready;
        top->mem_rdata = (uint32_t)rdata;
        // one full clock period (posedge updates registers)
        top->clk = 0;
        top->eval();
        top->clk = 1;
        top->eval();
    }
    auto t1 = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();
    std::printf("Verilator picorv32: %ld cycles, %llu mem reads, %.2f Mcyc/s (%.1f ms)\n",
                n, (unsigned long long)fetches, n / dt / 1e6, dt * 1e3);
    delete top;
    return 0;
}
