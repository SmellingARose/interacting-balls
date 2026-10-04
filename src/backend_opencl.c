// OpenCL backend (AMD Radeon such as the RX 6700 XT on Windows/Linux; also works on macOS).
// The OpenCL library is loaded at runtime from the GPU driver, so no SDK is needed to build or run.
// One work-item per flight, `chunk` physics steps per dispatch; flights that have ended are dropped from later dispatches.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "br_cl.h"
#include "threads.h"
#ifdef _WIN32
  /* windows.h comes from threads.h */
#else
  #include <dlfcn.h>
#endif

BrCL CL;
static int clLoaded = -1;

int br_cl_load(void) {
  if (clLoaded >= 0) return clLoaded;
  void* lib = NULL;
#if defined(_WIN32)
  lib = (void*)LoadLibraryA("OpenCL.dll");
  #define SYM(n) GetProcAddress((HMODULE)lib, n)
#elif defined(__APPLE__)
  lib = dlopen("/System/Library/Frameworks/OpenCL.framework/OpenCL", RTLD_NOW);
  #define SYM(n) dlsym(lib, n)
#else
  lib = dlopen("libOpenCL.so.1", RTLD_NOW); if (!lib) lib = dlopen("libOpenCL.so", RTLD_NOW);
  #define SYM(n) dlsym(lib, n)
#endif
  if (!lib) return clLoaded = 0;
  void** f = (void**)&CL; const char* names[] = { "clGetPlatformIDs", "clGetDeviceIDs", "clGetDeviceInfo", "clCreateContext", "clCreateCommandQueue",
    "clCreateProgramWithSource", "clBuildProgram", "clGetProgramBuildInfo", "clCreateKernel", "clCreateBuffer", "clSetKernelArg",
    "clEnqueueWriteBuffer", "clEnqueueReadBuffer", "clEnqueueNDRangeKernel", "clFinish", "clReleaseMemObject", "clReleaseKernel",
    "clReleaseProgram", "clReleaseCommandQueue", "clReleaseContext" };
  for (size_t i = 0; i < sizeof names / sizeof names[0]; i++) { f[i] = (void*)SYM(names[i]); if (!f[i]) return clLoaded = 0; }
  return clLoaded = 1;
}

typedef struct {
  cl_context ctx; cl_command_queue q; cl_device_id dev;
  cl_program prog; cl_kernel k, swk; int progMaxW, progSwW;
  cl_mem state, weights, scen, params, alive, idx, flags, traj, wD, bat; size_t capState, capW, capScen, capIdx, capFlags, capTraj, capWD, capBat;
  int chunk;
} ClImpl;

static int collect_devices(cl_device_id* out, int max) {
  if (!br_cl_load()) return 0;
  cl_platform_id plats[16]; cl_uint np = 0; int n = 0;
  if (CL.GetPlatformIDs(16, plats, &np) != CL_SUCCESS) return 0;
  for (cl_uint p = 0; p < np && n < max; p++) {
    cl_uint nd = 0; cl_device_id devs[16];
    if (CL.GetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 16, devs, &nd) != CL_SUCCESS) continue;
    for (cl_uint d = 0; d < nd && n < max; d++) out[n++] = devs[d];
  }
  return n;
}

int opencl_devices(char names[][160], int isGpu[], int max) {
  cl_device_id devs[32]; int n = collect_devices(devs, max < 32 ? max : 32);
  for (int i = 0; i < n; i++) {
    char name[120] = ""; cl_device_type t = 0; cl_uint cu = 0;
    CL.GetDeviceInfo(devs[i], CL_DEVICE_NAME, sizeof name, name, NULL);
    CL.GetDeviceInfo(devs[i], CL_DEVICE_TYPE, sizeof t, &t, NULL);
    CL.GetDeviceInfo(devs[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof cu, &cu, NULL);
    snprintf(names[i], 160, "%s (%u compute units)", name, cu); isGpu[i] = (t & CL_DEVICE_TYPE_GPU) != 0;
  }
  return n;
}

void opencl_list_devices(void) {
  char names[32][160]; int gpu[32]; int n = opencl_devices(names, gpu, 32);
  if (!n) { printf("No OpenCL devices found.\n"); return; }
  for (int i = 0; i < n; i++) printf("  [%d] %s %s\n", i, names[i], gpu[i] ? "GPU" : "CPU/other");
}

static int build_program(ClImpl* c, int maxW, int swW) {
  if (c->prog && c->progMaxW == maxW && c->progSwW == swW) return 0;
  if (c->k) CL.ReleaseKernel(c->k);
  if (c->swk) CL.ReleaseKernel(c->swk);
  if (c->prog) CL.ReleaseProgram(c->prog);
  c->k = c->swk = NULL; c->prog = NULL;
  const char* parts[2] = { BR_SRC_SIM_CORE, BR_SRC_OPENCL };
  cl_int e; c->prog = CL.CreateProgramWithSource(c->ctx, 2, parts, NULL, &e);
  char opts[160]; snprintf(opts, sizeof opts, "-DBR_OPENCL=1 -DMAXW=%d -DSW_MAXW=%d -cl-fast-relaxed-math -cl-mad-enable", maxW, swW);
  e = CL.BuildProgram(c->prog, 1, &c->dev, opts, NULL, NULL);
  if (e != CL_SUCCESS) {
    size_t len = 0; CL.GetProgramBuildInfo(c->prog, c->dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
    char* log = (char*)malloc(len + 1); CL.GetProgramBuildInfo(c->prog, c->dev, CL_PROGRAM_BUILD_LOG, len, log, NULL); log[len] = 0;
    fprintf(stderr, "OpenCL build failed (%d):\n%s\n", e, log); free(log); return -1;
  }
  c->k = CL.CreateKernel(c->prog, "br_kernel", &e);
  if (e != CL_SUCCESS) { fprintf(stderr, "OpenCL kernel create failed (%d)\n", e); return -1; }
  c->swk = CL.CreateKernel(c->prog, "br_sw_kernel", &e);
  if (e != CL_SUCCESS) { fprintf(stderr, "OpenCL swarm kernel create failed (%d)\n", e); return -1; }
  c->progMaxW = maxW; c->progSwW = swW; return 0;
}

static cl_mem grow(ClImpl* c, cl_mem m, size_t* cap, size_t bytes) {
  if (m && *cap >= bytes) return m;
  if (m) CL.ReleaseMemObject(m);
  cl_int e; *cap = bytes < 16 ? 16 : bytes;
  return CL.CreateBuffer(c->ctx, CL_MEM_READ_WRITE, *cap, NULL, &e);
}

static int opencl_eval(Backend* b, const float* weights, int nGenomes, const BrScen* scen, const float* traj, size_t trajFloats, BrParams* P, float* out) {
  ClImpl* c = (ClImpl*)b->impl;
  if (build_program(c, br_max_width(P), c->prog ? c->progSwW : br_sw_max_width(P))) return -1;
  P->nRoll = nGenomes * P->S;
  size_t sBytes = sizeof(float) * (size_t)P->nRoll * P->stride, wBytes = sizeof(float) * (size_t)nGenomes * P->nw;
  c->state = grow(c, c->state, &c->capState, sBytes);
  c->weights = grow(c, c->weights, &c->capW, wBytes);
  c->scen = grow(c, c->scen, &c->capScen, sizeof(BrScen) * (size_t)P->S);
  c->idx = grow(c, c->idx, &c->capIdx, sizeof(int) * (size_t)P->nRoll);
  c->flags = grow(c, c->flags, &c->capFlags, sizeof(int) * (size_t)P->nRoll);
  c->traj = grow(c, c->traj, &c->capTraj, sizeof(float) * (traj ? trajFloats : 4));
  if (!c->state || !c->weights || !c->idx || !c->flags || !c->traj) { fprintf(stderr, "OpenCL: out of GPU memory\n"); return -1; }
  int* idx = (int*)malloc(sizeof(int) * (size_t)P->nRoll); int* flags = (int*)malloc(sizeof(int) * (size_t)P->nRoll);
  for (int r = 0; r < P->nRoll; r++) idx[r] = r;
  P->nActive = P->nRoll;
  float* host = (float*)malloc(sBytes);
  br_init_states(host, P, scen);
  CL.EnqueueWriteBuffer(c->q, c->idx, CL_FALSE, 0, sizeof(int) * (size_t)P->nRoll, idx, 0, NULL, NULL);
  CL.EnqueueWriteBuffer(c->q, c->state, CL_FALSE, 0, sBytes, host, 0, NULL, NULL);
  CL.EnqueueWriteBuffer(c->q, c->weights, CL_FALSE, 0, wBytes, weights, 0, NULL, NULL);
  CL.EnqueueWriteBuffer(c->q, c->scen, CL_FALSE, 0, sizeof(BrScen) * (size_t)P->S, scen, 0, NULL, NULL);
  if (traj && trajFloats) CL.EnqueueWriteBuffer(c->q, c->traj, CL_FALSE, 0, sizeof(float) * trajFloats, traj, 0, NULL, NULL);
  size_t local = (size_t)(b->wg > 0 ? b->wg : 64); if ((int)local > b->maxWg) local = (size_t)b->maxWg;
  double target = b->chunkMs > 0 ? b->chunkMs : 40;
  int maxSteps = (int)(P->maxT / P->dt) + 8, done = 0; cl_uint alive = 0, zero = 0;
  while (done < maxSteps) {
    P->chunk = c->chunk;
    CL.EnqueueWriteBuffer(c->q, c->params, CL_FALSE, 0, sizeof(BrParams), P, 0, NULL, NULL);
    CL.EnqueueWriteBuffer(c->q, c->alive, CL_FALSE, 0, sizeof zero, &zero, 0, NULL, NULL);
    CL.SetKernelArg(c->k, 0, sizeof(cl_mem), &c->state); CL.SetKernelArg(c->k, 1, sizeof(cl_mem), &c->weights);
    CL.SetKernelArg(c->k, 2, sizeof(cl_mem), &c->scen); CL.SetKernelArg(c->k, 3, sizeof(cl_mem), &c->params);
    CL.SetKernelArg(c->k, 4, sizeof(cl_mem), &c->alive); CL.SetKernelArg(c->k, 5, sizeof(cl_mem), &c->idx); CL.SetKernelArg(c->k, 6, sizeof(cl_mem), &c->flags);
    CL.SetKernelArg(c->k, 7, sizeof(cl_mem), &c->traj);
    size_t global = ((size_t)P->nActive + local - 1) / local * local;
    double t0 = br_now();
    cl_int e = CL.EnqueueNDRangeKernel(c->q, c->k, 1, NULL, &global, &local, 0, NULL, NULL);
    if (e != CL_SUCCESS) { fprintf(stderr, "OpenCL launch failed (%d)\n", e); free(host); free(idx); free(flags); return -1; }
    CL.EnqueueReadBuffer(c->q, c->alive, CL_TRUE, 0, sizeof alive, &alive, 0, NULL, NULL);
    double ms = (br_now() - t0) * 1000;
    done += c->chunk;
    // keep each dispatch near the target time: long enough to amortise overhead, far below driver timeouts (~2 s on Windows)
    double scale = target / (ms > 0.5 ? ms : 0.5);
    int next = (int)(c->chunk * (scale > 4 ? 4 : scale < 0.25 ? 0.25 : scale));
    c->chunk = next < 16 ? 16 : next > 100000 ? 100000 : next;
    if (alive == 0) break;
    if (alive < 0.7 * P->nActive) {   // compact once ≥30% of launched flights have ended
      CL.EnqueueReadBuffer(c->q, c->flags, CL_TRUE, 0, sizeof(int) * (size_t)P->nRoll, flags, 0, NULL, NULL);
      int n = 0; for (int i = 0; i < P->nActive; i++) { int r = idx[i]; if (flags[r]) idx[n++] = r; }
      P->nActive = n;
      CL.EnqueueWriteBuffer(c->q, c->idx, CL_FALSE, 0, sizeof(int) * (size_t)n, idx, 0, NULL, NULL);
    }
  }
  CL.EnqueueReadBuffer(c->q, c->state, CL_TRUE, 0, sBytes, host, 0, NULL, NULL);
  br_collect(host, P, out);
  free(host); free(idx); free(flags);
  return 0;
}

// Swarm battles: same chunked dispatch and compaction as flights; one work-item per battle.
static int opencl_eval_battles(Backend* b, const float* attW, const float* defW, const int* bat, int nBattles, const BrScen* scen,
                               const float* start, size_t startFloats, BrParams* P, float* out) {
  (void)startFloats;
  ClImpl* c = (ClImpl*)b->impl;
  if (build_program(c, c->prog ? c->progMaxW : br_max_width(P), br_sw_max_width(P))) return -1;
  int nS = 0, nA = 0, nD = 0;
  for (int r = 0; r < nBattles; r++) { if (bat[r * 3 + 2] >= nS) nS = bat[r * 3 + 2] + 1; if (bat[r * 3] >= nA) nA = bat[r * 3] + 1; if (bat[r * 3 + 1] >= nD) nD = bat[r * 3 + 1] + 1; }
  P->nRoll = nBattles;
  size_t sBytes = sizeof(float) * (size_t)nBattles * P->stride;
  size_t aBytes = sizeof(float) * (P->attAI ? (size_t)nA * P->attNw : 1), dBytes = sizeof(float) * (P->defAI ? (size_t)nD * P->defNw : 1);
  c->state = grow(c, c->state, &c->capState, sBytes);
  c->weights = grow(c, c->weights, &c->capW, aBytes);
  c->wD = grow(c, c->wD, &c->capWD, dBytes);
  c->scen = grow(c, c->scen, &c->capScen, sizeof(BrScen) * (size_t)nS);
  c->idx = grow(c, c->idx, &c->capIdx, sizeof(int) * (size_t)nBattles);
  c->flags = grow(c, c->flags, &c->capFlags, sizeof(int) * (size_t)nBattles);
  c->bat = grow(c, c->bat, &c->capBat, sizeof(int) * 3 * (size_t)nBattles);
  if (!c->state || !c->weights || !c->wD || !c->idx || !c->flags || !c->bat) { fprintf(stderr, "OpenCL: out of GPU memory\n"); return -1; }
  int* idx = (int*)malloc(sizeof(int) * (size_t)nBattles); int* flags = (int*)malloc(sizeof(int) * (size_t)nBattles);
  for (int r = 0; r < nBattles; r++) idx[r] = r;
  P->nActive = nBattles;
  float* host = (float*)malloc(sBytes);
  br_sw_init_states(host, P, scen, bat, nBattles, start);
  CL.EnqueueWriteBuffer(c->q, c->idx, CL_FALSE, 0, sizeof(int) * (size_t)nBattles, idx, 0, NULL, NULL);
  CL.EnqueueWriteBuffer(c->q, c->state, CL_FALSE, 0, sBytes, host, 0, NULL, NULL);
  if (P->attAI) CL.EnqueueWriteBuffer(c->q, c->weights, CL_FALSE, 0, aBytes, attW, 0, NULL, NULL);
  if (P->defAI) CL.EnqueueWriteBuffer(c->q, c->wD, CL_FALSE, 0, dBytes, defW, 0, NULL, NULL);
  CL.EnqueueWriteBuffer(c->q, c->scen, CL_FALSE, 0, sizeof(BrScen) * (size_t)nS, scen, 0, NULL, NULL);
  CL.EnqueueWriteBuffer(c->q, c->bat, CL_FALSE, 0, sizeof(int) * 3 * (size_t)nBattles, bat, 0, NULL, NULL);
  size_t local = (size_t)(b->wg > 0 ? b->wg : 64); if ((int)local > b->maxWg) local = (size_t)b->maxWg;
  double target = b->chunkMs > 0 ? b->chunkMs : 40; int chunk = 64;
  int maxSteps = (int)(P->maxT / P->dt) + 8, done = 0; cl_uint alive = 0, zero = 0;
  while (done < maxSteps) {
    P->chunk = chunk;
    CL.EnqueueWriteBuffer(c->q, c->params, CL_FALSE, 0, sizeof(BrParams), P, 0, NULL, NULL);
    CL.EnqueueWriteBuffer(c->q, c->alive, CL_FALSE, 0, sizeof zero, &zero, 0, NULL, NULL);
    cl_mem args[9] = { c->state, c->weights, c->wD, c->scen, c->params, c->alive, c->idx, c->flags, c->bat };
    for (int i = 0; i < 9; i++) CL.SetKernelArg(c->swk, i, sizeof(cl_mem), &args[i]);
    size_t global = ((size_t)P->nActive + local - 1) / local * local;
    double t0 = br_now();
    cl_int e = CL.EnqueueNDRangeKernel(c->q, c->swk, 1, NULL, &global, &local, 0, NULL, NULL);
    if (e != CL_SUCCESS) { fprintf(stderr, "OpenCL launch failed (%d)\n", e); free(host); free(idx); free(flags); return -1; }
    CL.EnqueueReadBuffer(c->q, c->alive, CL_TRUE, 0, sizeof alive, &alive, 0, NULL, NULL);
    double ms = (br_now() - t0) * 1000;
    done += chunk;
    double scale = target / (ms > 0.5 ? ms : 0.5);
    int next = (int)(chunk * (scale > 4 ? 4 : scale < 0.25 ? 0.25 : scale));
    chunk = next < 8 ? 8 : next > 100000 ? 100000 : next;
    if (alive == 0) break;
    if (alive < 0.7 * P->nActive) {
      CL.EnqueueReadBuffer(c->q, c->flags, CL_TRUE, 0, sizeof(int) * (size_t)nBattles, flags, 0, NULL, NULL);
      int n = 0; for (int i = 0; i < P->nActive; i++) { int r = idx[i]; if (flags[r]) idx[n++] = r; }
      P->nActive = n;
      CL.EnqueueWriteBuffer(c->q, c->idx, CL_FALSE, 0, sizeof(int) * (size_t)n, idx, 0, NULL, NULL);
    }
  }
  CL.EnqueueReadBuffer(c->q, c->state, CL_TRUE, 0, sBytes, host, 0, NULL, NULL);
  br_sw_collect(host, P, nBattles, out);
  free(host); free(idx); free(flags);
  return 0;
}

static void opencl_destroy(Backend* b) { ClImpl* c = (ClImpl*)b->impl;
  cl_mem ms[10] = { c->state, c->weights, c->scen, c->params, c->alive, c->idx, c->flags, c->traj, c->wD, c->bat };
  for (int i = 0; i < 10; i++) if (ms[i]) CL.ReleaseMemObject(ms[i]);
  if (c->k) CL.ReleaseKernel(c->k);
  if (c->swk) CL.ReleaseKernel(c->swk);
  if (c->prog) CL.ReleaseProgram(c->prog);
  CL.ReleaseCommandQueue(c->q); CL.ReleaseContext(c->ctx); free(c); free(b); }

Backend* opencl_backend_create(int deviceIndex, char* err, int errLen) {
  if (!br_cl_load()) { snprintf(err, errLen, "OpenCL is not available (GPU driver missing?)"); return NULL; }
  cl_device_id devs[32]; int n = collect_devices(devs, 32);
  if (!n) { snprintf(err, errLen, "no OpenCL devices (is the GPU driver installed?)"); return NULL; }
  int pick = deviceIndex;
  if (pick < 0) { pick = 0; for (int i = 0; i < n; i++) { cl_device_type t = 0; CL.GetDeviceInfo(devs[i], CL_DEVICE_TYPE, sizeof t, &t, NULL); if (t & CL_DEVICE_TYPE_GPU) { pick = i; break; } } }
  if (pick >= n) { snprintf(err, errLen, "OpenCL device %d not found (have %d)", pick, n); return NULL; }
  ClImpl* c = (ClImpl*)calloc(1, sizeof(ClImpl)); cl_int e;
  c->dev = devs[pick];
  c->ctx = CL.CreateContext(NULL, 1, &c->dev, NULL, NULL, &e);
  if (e != CL_SUCCESS) { snprintf(err, errLen, "clCreateContext failed (%d)", e); free(c); return NULL; }
  c->q = CL.CreateCommandQueue(c->ctx, c->dev, 0, &e);
  c->params = CL.CreateBuffer(c->ctx, CL_MEM_READ_ONLY, sizeof(BrParams), NULL, &e);
  c->alive = CL.CreateBuffer(c->ctx, CL_MEM_READ_WRITE, sizeof(cl_uint), NULL, &e);
  c->chunk = 256;
  Backend* b = (Backend*)calloc(1, sizeof(Backend));
  b->name = "opencl"; b->eval = opencl_eval; b->evalBattles = opencl_eval_battles; b->destroy = opencl_destroy; b->impl = c;
  size_t mwg = 256; CL.GetDeviceInfo(c->dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof mwg, &mwg, NULL);
  cl_ulong mem = 0; CL.GetDeviceInfo(c->dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof mem, &mem, NULL);
  b->wg = 64; b->chunkMs = 40; b->maxWg = (int)mwg; b->memBytes = (double)mem;
  char name[200] = ""; CL.GetDeviceInfo(c->dev, CL_DEVICE_NAME, sizeof name, name, NULL);
  snprintf(b->info, sizeof b->info, "OpenCL, %s", name);
  return b;
}
