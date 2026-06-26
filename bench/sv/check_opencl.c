// Build-validate an OpenCL kernel on the system OpenCL runtime (Apple's here).
#include <OpenCL/opencl.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char** argv) {
  FILE* f = fopen(argv[1], "rb");
  if (!f) { printf("cannot open %s\n", argv[1]); return 2; }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  char* src = malloc(n + 1); fread(src, 1, n, f); src[n] = 0; fclose(f);
  cl_platform_id plat; if (clGetPlatformIDs(1, &plat, 0) != CL_SUCCESS) { printf("no OpenCL platform\n"); return 3; }
  cl_device_id dev; clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 1, &dev, 0);
  char dn[256]; clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(dn), dn, 0);
  cl_context ctx = clCreateContext(0, 1, &dev, 0, 0, 0);
  cl_program prog = clCreateProgramWithSource(ctx, 1, (const char**)&src, 0, 0);
  cl_int err = clBuildProgram(prog, 1, &dev, "", 0, 0);
  if (err != CL_SUCCESS) {
    char log[16384]; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, 0);
    printf("OpenCL BUILD FAILED (%d) on %s:\n%s\n", err, dn, log); return 1;
  }
  clCreateKernel(prog, "tick", &err);
  printf("OpenCL build: SUCCESS — kernel 'tick' compiled on '%s'%s\n", dn, err == CL_SUCCESS ? "" : " (kernel missing!)");
  return 0;
}
