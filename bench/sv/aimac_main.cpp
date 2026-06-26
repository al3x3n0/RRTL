#include "Vaimac.h"
#include "verilated.h"
#include <chrono>
#include <cstdio>
int main(int argc, char** argv){ long n=argc>1?atol(argv[1]):2000000; Verilated::commandArgs(argc,argv);
 Vaimac* t=new Vaimac; for(int w=0;w<16;w++){ t->a[w]=0x11223344u*(w+1); t->b[w]=0x55667788u*(w+3); }
 t->clk=0; t->eval();
 double best=1e18; for(int r=0;r<4;r++){ auto s=std::chrono::steady_clock::now();
  for(long c=0;c<n;c++){ t->clk=0; t->eval(); t->clk=1; t->eval(); }
  double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-s).count(); if(dt<best)best=dt; }
 printf("aimac out=%u %.2f Mcyc/s\n",(unsigned)t->o,n/best/1e6); delete t; return 0; }
