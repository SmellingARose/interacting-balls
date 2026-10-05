// OpenCL backend (AMD, NVIDIA and Intel GPUs on Windows and Linux; also works on macOS). The OpenCL library is loaded
// at runtime from the GPU driver, so no SDK is needed to build or run. One work-item per flight (or battle), `chunk`
// physics steps per dispatch; work-items that have ended are dropped from later dispatches, and the CPU can finish the
// rest (BrTail). Every call into the driver is checked: a failure (device reset, out of memory) returns -1.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "br_cl.h"
#include "threads.h"
#ifdef _WIN32
  /* windows.h comes from threads.h */
  #ifndef LOAD_LIBRARY_SEARCH_SYSTEM32
  #define LOAD_LIBRARY_SEARCH_SYSTEM32 0x00000800
  #endif
#else
  #include <dlfcn.h>
#endif

BrCL CL;
static int clLoaded = -1;

int br_cl_load(void) {
  if (clLoaded >= 0) return clLoaded;
  void* lib = NULL;
#if defined(_WIN32)
  // the driver's loader in System32, not a stray OpenCL.dll next to the program. The usual search only where Windows
  // lacks the flag (Windows 7 without its loader update rejects it as an invalid parameter); a missing driver is just missing.
  lib = (void*)LoadLibraryExA("OpenCL.dll", NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (!lib && GetLastError() == ERROR_INVALID_PARAMETER) lib = (void*)LoadLibraryA("OpenCL.dll");
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
    "clReleaseProgram", "clReleaseCommandQueue", "clReleaseContext", "clGetKernelWorkGroupInfo" };
  for (size_t i = 0; i < sizeof names / sizeof names[0]; i++) { f[i] = (void*)SYM(names[i]); if (!f[i]) return clLoaded = 0; }
  return clLoaded = 1;
}

typedef struct {
  cl_context ctx; cl_command_queue q; cl_device_id dev;
  cl_program prog, gprog; cl_kernel k, swk, swgk; int progMaxW, progSwW, grpSwW, grpStride, grpBadSwW, grpBadStride;
  size_t devWg, kWg, swkWg, swgkWg;   // work-group limits: the device's, then each kernel's own (register use can lower it)
  cl_ulong maxAlloc, localMem;
  cl_mem state, weights, scen, params, alive, idx, flags, traj, wD, bat; size_t capState, capW, capScen, capIdx, capFlags, capTraj, capWD, capBat;
} ClImpl;

// Every device of every platform, in a fixed order (the N of "opencl:N"). The counts the driver returns may exceed
// the room given, so they are clamped.
static int collect_devices(cl_device_id* out, int max) {
  if (!br_cl_load()) return 0;
  cl_platform_id plats[16]; cl_uint np = 0; int n = 0;
  if (CL.GetPlatformIDs(16, plats, &np) != CL_SUCCESS) return 0;
  if (np > 16) np = 16;
  for (cl_uint p = 0; p < np && n < max; p++) {
    cl_uint nd = 0; cl_device_id devs[16];
    if (CL.GetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 16, devs, &nd) != CL_SUCCESS) continue;
    if (nd > 16) nd = 16;
    for (cl_uint d = 0; d < nd && n < max; d++) out[n++] = devs[d];
  }
  return n;
}

int opencl_devices(char names[][160], int isGpu[], int max) {
  cl_device_id devs[32]; int n = collect_devices(devs, max < 32 ? max : 32);
  for (int i = 0; i < n; i++) {
    char name[120] = ""; cl_device_type t = 0; cl_uint cu = 0;
    CL.GetDeviceInfo(devs[i], CL_DEVICE_NAME, sizeof name, name, NULL); name[sizeof name - 1] = 0;
    CL.GetDeviceInfo(devs[i], CL_DEVICE_TYPE, sizeof t, &t, NULL);
    CL.GetDeviceInfo(devs[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof cu, &cu, NULL);
    snprintf(names[i], 160, "%s (%u compute units)", name, cu); isGpu[i] = (t & CL_DEVICE_TYPE_GPU) != 0;
  }
  return n;
}

// The fastest-looking GPU: a discrete one before any integrated one (those share the CPU's memory), then compute units
// × lanes per unit (NVIDIA SMs are 128 wide, AMD CUs 64, Intel EUs 8) × clock.
int opencl_best_gpu(void) {
  cl_device_id devs[32]; int n = collect_devices(devs, 32), best = -1; double top = -1;
  for (int i = 0; i < n; i++) {
    cl_device_type t = 0; cl_uint cu = 0, mhz = 0, vendor = 0; cl_bool uni = 0;
    if (CL.GetDeviceInfo(devs[i], CL_DEVICE_TYPE, sizeof t, &t, NULL) != CL_SUCCESS || !(t & CL_DEVICE_TYPE_GPU)) continue;
    CL.GetDeviceInfo(devs[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof cu, &cu, NULL);
    CL.GetDeviceInfo(devs[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof mhz, &mhz, NULL);
    CL.GetDeviceInfo(devs[i], CL_DEVICE_VENDOR_ID, sizeof vendor, &vendor, NULL);
    if (CL.GetDeviceInfo(devs[i], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof uni, &uni, NULL) != CL_SUCCESS) uni = 0;
    double lanes = vendor == 0x10DE ? 128 : vendor == 0x8086 ? 8 : 64;
    double s = (double)(cu ? cu : 1) * lanes * (mhz ? mhz : 1000) + (uni ? 0 : 1e12);
    if (s > top) { top = s; best = i; }
  }
  return best;
}

void opencl_list_devices(void) {
  char names[32][160]; int gpu[32]; int n = opencl_devices(names, gpu, 32), best = opencl_best_gpu();
  if (!n) { printf("No OpenCL devices found.\n"); return; }
  for (int i = 0; i < n; i++) printf("  [%d] %s %s%s\n", i, names[i], gpu[i] ? "GPU" : "CPU/other", i == best ? " (default)" : "");
}

static void build_log(ClImpl* c, cl_program p, const char* what, cl_int e) {
  size_t len = 0; CL.GetProgramBuildInfo(p, c->dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
  char* log = (char*)malloc(len + 1);
  if (log) { if (CL.GetProgramBuildInfo(p, c->dev, CL_PROGRAM_BUILD_LOG, len, log, NULL) != CL_SUCCESS) len = 0; log[len] = 0; }
  fprintf(stderr, "OpenCL %s build failed (%d):\n%s\n", what, e, log ? log : ""); free(log);
}
static size_t kernel_wg(ClImpl* c, cl_kernel k) {
  size_t v = 0;
  if (CL.GetKernelWorkGroupInfo(k, c->dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof v, &v, NULL) != CL_SUCCESS || !v) v = c->devWg;
  return v < c->devWg ? v : c->devWg;
}
static cl_program make_program(ClImpl* c, const char* opts, const char* what) {
  const char* parts[2] = { BR_SRC_SIM_CORE, BR_SRC_OPENCL }; cl_int e = CL_SUCCESS;
  cl_program p = CL.CreateProgramWithSource(c->ctx, 2, parts, NULL, &e);
  if (!p || e != CL_SUCCESS) { fprintf(stderr, "OpenCL %s program failed (%d)\n", what, e); return NULL; }
  if ((e = CL.BuildProgram(p, 1, &c->dev, opts, NULL, NULL)) != CL_SUCCESS) { build_log(c, p, what, e); CL.ReleaseProgram(p); return NULL; }
  return p;
}
static cl_kernel make_kernel(cl_program p, const char* name) {
  cl_int e = CL_SUCCESS; cl_kernel k = CL.CreateKernel(p, name, &e);
  if (!k || e != CL_SUCCESS) { fprintf(stderr, "OpenCL kernel %s failed (%d)\n", name, e); return NULL; }
  return k;
}

// Flight and thread-per-battle kernels, compiled for these layer widths. A failed build leaves nothing behind, so a
// later call cannot mistake it for a built one.
static int build_program(ClImpl* c, int maxW, int swW) {
  if (c->k && c->swk && c->progMaxW == maxW && c->progSwW == swW) return 0;
  if (c->k) CL.ReleaseKernel(c->k);
  if (c->swk) CL.ReleaseKernel(c->swk);
  if (c->prog) CL.ReleaseProgram(c->prog);
  c->k = c->swk = NULL; c->prog = NULL; c->progMaxW = c->progSwW = 0;
  char opts[160]; snprintf(opts, sizeof opts, "-DBR_OPENCL=1 -DMAXW=%d -DSW_MAXW=%d -cl-fast-relaxed-math -cl-mad-enable", maxW, swW);
  if (!(c->prog = make_program(c, opts, "kernel"))) return -1;
  c->k = make_kernel(c->prog, "br_kernel"); c->swk = c->k ? make_kernel(c->prog, "br_sw_kernel") : NULL;
  if (!c->k || !c->swk) {
    if (c->k) CL.ReleaseKernel(c->k);
    CL.ReleaseProgram(c->prog); c->k = NULL; c->prog = NULL; return -1;
  }
  c->kWg = kernel_wg(c, c->k); c->swkWg = kernel_wg(c, c->swk);
  c->progMaxW = maxW; c->progSwW = swW; return 0;
}

// The work-group battle kernel keeps the battle in __local memory: built per battle size (SW_STRIDE floats).
// 1: ready; 0: this size does not fit the device's local memory or failed to build (use a thread per battle).
static int build_group(ClImpl* c, int swW, int stride) {
  if (c->swgk && c->grpSwW == swW && c->grpStride == stride) return 1;
  if ((c->grpBadSwW == swW && c->grpBadStride == stride) || sizeof(float) * ((size_t)stride + 2 * (size_t)swW) > c->localMem) return 0;
  if (c->swgk) CL.ReleaseKernel(c->swgk);
  if (c->gprog) CL.ReleaseProgram(c->gprog);
  c->swgk = NULL; c->gprog = NULL; c->grpSwW = c->grpStride = 0;
  char opts[220]; snprintf(opts, sizeof opts, "-DBR_OPENCL=1 -DBR_SW_LOCAL=1 -DBR_SG=__local -DMAXW=32 -DSW_MAXW=%d -DSW_STRIDE=%d -cl-fast-relaxed-math -cl-mad-enable", swW, stride);
  cl_ulong lm = 0;
  if ((c->gprog = make_program(c, opts, "swarm group")) && (c->swgk = make_kernel(c->gprog, "br_sw_group_kernel")) &&
      (CL.GetKernelWorkGroupInfo(c->swgk, c->dev, CL_KERNEL_LOCAL_MEM_SIZE, sizeof lm, &lm, NULL) != CL_SUCCESS || lm <= c->localMem)) {
    c->swgkWg = kernel_wg(c, c->swgk); c->grpSwW = swW; c->grpStride = stride; return 1;
  }
  if (c->swgk) CL.ReleaseKernel(c->swgk);
  if (c->gprog) CL.ReleaseProgram(c->gprog);
  c->swgk = NULL; c->gprog = NULL; c->grpBadSwW = swW; c->grpBadStride = stride;
  fprintf(stderr, "OpenCL: no work-group kernel for this battle size; one work-item per battle instead\n");
  return 0;
}

// A buffer of at least `bytes` (NULL when the device cannot hold one that big).
static cl_mem grow(ClImpl* c, cl_mem m, size_t* cap, size_t bytes) {
  if (m && *cap >= bytes) return m;
  if (m) CL.ReleaseMemObject(m);
  *cap = 0; if (bytes < 16) bytes = 16;
  if (bytes > c->maxAlloc) return NULL;
  cl_int e = CL_SUCCESS; cl_mem r = CL.CreateBuffer(c->ctx, CL_MEM_READ_WRITE, bytes, NULL, &e);
  if (!r || e != CL_SUCCESS) return NULL;
  *cap = bytes; return r;
}
static int no_buffers(ClImpl* c, size_t bytes) {
  fprintf(stderr, "OpenCL: out of GPU memory for this batch (%.0f MB; largest buffer allowed %.0f MB)\n", bytes / 1e6, (double)c->maxAlloc / 1e6);
  return -1;
}
static int cl_fail(ClImpl* c, cl_int e) {
  fprintf(stderr, "OpenCL error %d%s\n", e, e == -5 || e == -36 || e == -4 ? " (GPU out of resources or reset)" : "");
  CL.Finish(c->q);   // writes from host memory may still be queued: they must end before that memory is freed
  return -1;
}
#define CK(x) do { if ((e = (x)) != CL_SUCCESS) { cl_fail(c, e); goto fail; } } while (0)

static int opencl_eval(Backend* b, const float* weights, int nGenomes, const BrScen* scen, const float* traj, size_t trajFloats, BrParams* P, float* out) {
  ClImpl* c = (ClImpl*)b->impl;
  P->nRoll = nGenomes * P->S;
  if (P->nRoll <= 0) return 0;
  if (build_program(c, br_max_width(P), c->k ? c->progSwW : br_sw_max_width(P))) return -1;
  b->maxWg = (int)c->kWg;
  size_t sBytes = sizeof(float) * (size_t)P->nRoll * P->stride, wBytes = sizeof(float) * (size_t)nGenomes * P->nw;
  size_t cBytes = sizeof(BrScen) * (size_t)P->S, iBytes = sizeof(int) * (size_t)P->nRoll, tBytes = sizeof(float) * (traj && trajFloats ? trajFloats : 4);
  c->state = grow(c, c->state, &c->capState, sBytes); c->weights = grow(c, c->weights, &c->capW, wBytes);
  c->scen = grow(c, c->scen, &c->capScen, cBytes); c->idx = grow(c, c->idx, &c->capIdx, iBytes);
  c->flags = grow(c, c->flags, &c->capFlags, iBytes); c->traj = grow(c, c->traj, &c->capTraj, tBytes);
  if (!c->state || !c->weights || !c->scen || !c->idx || !c->flags || !c->traj) return no_buffers(c, sBytes > tBytes ? sBytes : tBytes);
  int* idx = (int*)malloc(iBytes); int* flags = (int*)malloc(iBytes); float* host = (float*)malloc(sBytes); cl_int e = CL_SUCCESS;
  if (!idx || !flags || !host) { free(idx); free(flags); free(host); fprintf(stderr, "OpenCL: out of memory\n"); return -1; }
  for (int r = 0; r < P->nRoll; r++) idx[r] = r;
  P->nActive = P->nRoll;
  br_init_states(host, P, scen);
  br_tail_begin(&b->tail, host, P, weights, NULL, NULL, NULL, scen, traj);
  CK(CL.EnqueueWriteBuffer(c->q, c->idx, CL_FALSE, 0, iBytes, idx, 0, NULL, NULL));
  CK(CL.EnqueueWriteBuffer(c->q, c->state, CL_FALSE, 0, sBytes, host, 0, NULL, NULL));
  CK(CL.EnqueueWriteBuffer(c->q, c->weights, CL_FALSE, 0, wBytes, weights, 0, NULL, NULL));
  CK(CL.EnqueueWriteBuffer(c->q, c->scen, CL_FALSE, 0, cBytes, scen, 0, NULL, NULL));
  if (traj && trajFloats) CK(CL.EnqueueWriteBuffer(c->q, c->traj, CL_FALSE, 0, tBytes, traj, 0, NULL, NULL));
  { cl_mem args[8] = { c->state, c->weights, c->scen, c->params, c->alive, c->idx, c->flags, c->traj };
    for (cl_uint i = 0; i < 8; i++) CK(CL.SetKernelArg(c->k, i, sizeof(cl_mem), &args[i])); }
  size_t local = (size_t)(b->wg > 0 ? b->wg : 64); if (local > c->kWg) local = c->kWg;
  double target = b->chunkMs > 0 ? b->chunkMs : 40;
  int maxSteps = (int)(P->maxT / P->dt) + 8, done = 0, chunk = br_chunk_first(b, P, 0, target), onHost = 0; cl_uint alive = 0, zero = 0;
  while (done < maxSteps) {
    P->chunk = chunk;
    CK(CL.EnqueueWriteBuffer(c->q, c->params, CL_FALSE, 0, sizeof(BrParams), P, 0, NULL, NULL));
    CK(CL.EnqueueWriteBuffer(c->q, c->alive, CL_FALSE, 0, sizeof zero, &zero, 0, NULL, NULL));
    size_t global = ((size_t)P->nActive + local - 1) / local * local;
    double t0 = br_now();
    CK(CL.EnqueueNDRangeKernel(c->q, c->k, 1, NULL, &global, &local, 0, NULL, NULL));
    CK(CL.EnqueueReadBuffer(c->q, c->alive, CL_TRUE, 0, sizeof alive, &alive, 0, NULL, NULL));   // kernel failures surface here
    double ms = (br_now() - t0) * 1000;
    if (!done || alive == (cl_uint)P->nRoll) br_chunk_seen(b, P, 0, chunk, ms);
    done += chunk;
    if (alive == 0) break;
    if (br_tail_due(&b->tail, P, (int)alive, P->nActive, ms, chunk)) {
      CK(CL.EnqueueReadBuffer(c->q, c->state, CL_TRUE, 0, sBytes, host, 0, NULL, NULL)); onHost = 1;
      if (br_tail_run(&b->tail, host, P, idx, P->nActive, weights, NULL, NULL, NULL, scen, traj)) goto fail;
      break;
    }
    chunk = br_chunk_next(P, chunk, ms, target);
    if (alive < 0.7 * P->nActive) {   // compact once ≥30% of launched flights have ended
      CK(CL.EnqueueReadBuffer(c->q, c->flags, CL_TRUE, 0, iBytes, flags, 0, NULL, NULL));
      int n = 0; for (int i = 0; i < P->nActive; i++) { int r = idx[i]; if (flags[r]) idx[n++] = r; }
      P->nActive = n;
      CK(CL.EnqueueWriteBuffer(c->q, c->idx, CL_FALSE, 0, sizeof(int) * (size_t)n, idx, 0, NULL, NULL));
    }
  }
  if (!onHost) CK(CL.EnqueueReadBuffer(c->q, c->state, CL_TRUE, 0, sBytes, host, 0, NULL, NULL));
  br_collect(host, P, out);
  free(host); free(idx); free(flags);
  return 0;
fail:
  free(host); free(idx); free(flags);
  return -1;
}

// Swarm battles: same chunked dispatch, compaction and CPU tail as flights; a work-item or a work-group per battle.
static int opencl_eval_battles(Backend* b, const float* attW, const float* defW, const int* bat, int nBattles, const BrScen* scen,
                               const float* start, size_t startFloats, BrParams* P, float* out) {
  (void)startFloats;
  ClImpl* c = (ClImpl*)b->impl;
  P->nRoll = nBattles;
  if (nBattles <= 0) return 0;
  int swW = br_sw_max_width(P);
  if (build_program(c, c->k ? c->progMaxW : br_max_width(P), swW)) return -1;
  int gThreads = 32, group = br_sw_group_layout(b, P, &gThreads);
  if (group) group = build_group(c, swW, P->stride);
  cl_kernel kern = group ? c->swgk : c->swk; size_t kmax = group ? c->swgkWg : c->swkWg;
  if (!group) b->maxWg = (int)c->swkWg;
  int nS = 0, nA = 0, nD = 0;
  for (int r = 0; r < nBattles; r++) { if (bat[r * 3 + 2] >= nS) nS = bat[r * 3 + 2] + 1; if (bat[r * 3] >= nA) nA = bat[r * 3] + 1; if (bat[r * 3 + 1] >= nD) nD = bat[r * 3 + 1] + 1; }
  size_t sBytes = sizeof(float) * (size_t)nBattles * P->stride, iBytes = sizeof(int) * (size_t)nBattles, cBytes = sizeof(BrScen) * (size_t)nS;
  size_t aBytes = sizeof(float) * (P->attAI ? (size_t)nA * P->attNw : 1), dBytes = sizeof(float) * (P->defAI ? (size_t)nD * P->defNw : 1);
  c->state = grow(c, c->state, &c->capState, sBytes); c->weights = grow(c, c->weights, &c->capW, aBytes); c->wD = grow(c, c->wD, &c->capWD, dBytes);
  c->scen = grow(c, c->scen, &c->capScen, cBytes); c->idx = grow(c, c->idx, &c->capIdx, iBytes); c->flags = grow(c, c->flags, &c->capFlags, iBytes);
  c->bat = grow(c, c->bat, &c->capBat, 3 * iBytes);
  if (!c->state || !c->weights || !c->wD || !c->scen || !c->idx || !c->flags || !c->bat) return no_buffers(c, sBytes);
  int* idx = (int*)malloc(iBytes); int* flags = (int*)malloc(iBytes); float* host = (float*)malloc(sBytes); cl_int e = CL_SUCCESS;
  if (!idx || !flags || !host) { free(idx); free(flags); free(host); fprintf(stderr, "OpenCL: out of memory\n"); return -1; }
  for (int r = 0; r < nBattles; r++) idx[r] = r;
  P->nActive = nBattles;
  br_sw_init_states(host, P, scen, bat, nBattles, start);
  br_tail_begin(&b->tail, host, P, NULL, attW, defW, bat, scen, NULL);
  CK(CL.EnqueueWriteBuffer(c->q, c->idx, CL_FALSE, 0, iBytes, idx, 0, NULL, NULL));
  CK(CL.EnqueueWriteBuffer(c->q, c->state, CL_FALSE, 0, sBytes, host, 0, NULL, NULL));
  if (P->attAI) CK(CL.EnqueueWriteBuffer(c->q, c->weights, CL_FALSE, 0, aBytes, attW, 0, NULL, NULL));
  if (P->defAI) CK(CL.EnqueueWriteBuffer(c->q, c->wD, CL_FALSE, 0, dBytes, defW, 0, NULL, NULL));
  CK(CL.EnqueueWriteBuffer(c->q, c->scen, CL_FALSE, 0, cBytes, scen, 0, NULL, NULL));
  CK(CL.EnqueueWriteBuffer(c->q, c->bat, CL_FALSE, 0, 3 * iBytes, bat, 0, NULL, NULL));
  { cl_mem args[9] = { c->state, c->weights, c->wD, c->scen, c->params, c->alive, c->idx, c->flags, c->bat };
    for (cl_uint i = 0; i < 9; i++) CK(CL.SetKernelArg(kern, i, sizeof(cl_mem), &args[i])); }
  size_t local = group ? (size_t)gThreads : (size_t)(b->wg > 0 ? b->wg : 64); if (local > kmax) local = kmax;
  double target = b->chunkMs > 0 ? b->chunkMs : 40; int layout = group ? (int)local : 0, chunk = br_chunk_first(b, P, layout, target), onHost = 0;
  int maxSteps = (int)(P->maxT / P->dt) + 8, done = 0; cl_uint alive = 0, zero = 0;
  while (done < maxSteps) {
    P->chunk = chunk;
    CK(CL.EnqueueWriteBuffer(c->q, c->params, CL_FALSE, 0, sizeof(BrParams), P, 0, NULL, NULL));
    CK(CL.EnqueueWriteBuffer(c->q, c->alive, CL_FALSE, 0, sizeof zero, &zero, 0, NULL, NULL));
    size_t global = group ? (size_t)P->nActive * local : ((size_t)P->nActive + local - 1) / local * local;
    double t0 = br_now();
    CK(CL.EnqueueNDRangeKernel(c->q, kern, 1, NULL, &global, &local, 0, NULL, NULL));
    CK(CL.EnqueueReadBuffer(c->q, c->alive, CL_TRUE, 0, sizeof alive, &alive, 0, NULL, NULL));
    double ms = (br_now() - t0) * 1000;
    if (!done || alive == (cl_uint)P->nRoll) br_chunk_seen(b, P, layout, chunk, ms);
    done += chunk;
    if (alive == 0) break;
    if (br_tail_due(&b->tail, P, (int)alive, P->nActive, ms, chunk)) {
      CK(CL.EnqueueReadBuffer(c->q, c->state, CL_TRUE, 0, sBytes, host, 0, NULL, NULL)); onHost = 1;
      if (br_tail_run(&b->tail, host, P, idx, P->nActive, NULL, attW, defW, bat, scen, NULL)) goto fail;
      break;
    }
    chunk = br_chunk_next(P, chunk, ms, target);
    if (alive < 0.7 * P->nActive) {
      CK(CL.EnqueueReadBuffer(c->q, c->flags, CL_TRUE, 0, iBytes, flags, 0, NULL, NULL));
      int n = 0; for (int i = 0; i < P->nActive; i++) { int r = idx[i]; if (flags[r]) idx[n++] = r; }
      P->nActive = n;
      CK(CL.EnqueueWriteBuffer(c->q, c->idx, CL_FALSE, 0, sizeof(int) * (size_t)n, idx, 0, NULL, NULL));
    }
  }
  if (!onHost) CK(CL.EnqueueReadBuffer(c->q, c->state, CL_TRUE, 0, sBytes, host, 0, NULL, NULL));
  br_sw_collect(host, P, nBattles, out);
  free(host); free(idx); free(flags);
  return 0;
fail:
  free(host); free(idx); free(flags);
  return -1;
}

static void opencl_destroy(Backend* b) { ClImpl* c = (ClImpl*)b->impl;
  cl_mem ms[10] = { c->state, c->weights, c->scen, c->params, c->alive, c->idx, c->flags, c->traj, c->wD, c->bat };
  for (int i = 0; i < 10; i++) if (ms[i]) CL.ReleaseMemObject(ms[i]);
  if (c->k) CL.ReleaseKernel(c->k);
  if (c->swk) CL.ReleaseKernel(c->swk);
  if (c->swgk) CL.ReleaseKernel(c->swgk);
  if (c->gprog) CL.ReleaseProgram(c->gprog);
  if (c->prog) CL.ReleaseProgram(c->prog);
  if (c->q) CL.ReleaseCommandQueue(c->q);
  if (c->ctx) CL.ReleaseContext(c->ctx);
  free(c); free(b); }

Backend* opencl_backend_create(int deviceIndex, char* err, int errLen) {
  if (!br_cl_load()) { snprintf(err, errLen, "OpenCL is not available (GPU driver missing?)"); return NULL; }
  cl_device_id devs[32]; int n = collect_devices(devs, 32);
  if (!n) { snprintf(err, errLen, "no OpenCL devices (is the GPU driver installed?)"); return NULL; }
  int pick = deviceIndex >= 0 ? deviceIndex : opencl_best_gpu(); if (pick < 0) pick = 0;
  if (pick >= n) { snprintf(err, errLen, "OpenCL device %d not found (have %d)", pick, n); return NULL; }
  ClImpl* c = (ClImpl*)calloc(1, sizeof(ClImpl)); Backend* b = (Backend*)calloc(1, sizeof(Backend)); cl_int e = CL_SUCCESS;
  if (!c || !b) { free(c); free(b); snprintf(err, errLen, "out of memory"); return NULL; }
  b->impl = c; c->dev = devs[pick];
  c->ctx = CL.CreateContext(NULL, 1, &c->dev, NULL, NULL, &e);
  if (!c->ctx || e != CL_SUCCESS) { snprintf(err, errLen, "clCreateContext failed (%d)", e); c->ctx = NULL; opencl_destroy(b); return NULL; }
  c->q = CL.CreateCommandQueue(c->ctx, c->dev, 0, &e);
  if (!c->q || e != CL_SUCCESS) { snprintf(err, errLen, "clCreateCommandQueue failed (%d)", e); c->q = NULL; opencl_destroy(b); return NULL; }
  c->params = CL.CreateBuffer(c->ctx, CL_MEM_READ_ONLY, sizeof(BrParams), NULL, &e); if (e != CL_SUCCESS) c->params = NULL;
  c->alive = CL.CreateBuffer(c->ctx, CL_MEM_READ_WRITE, sizeof(cl_uint), NULL, &e); if (e != CL_SUCCESS) c->alive = NULL;
  if (!c->params || !c->alive) { snprintf(err, errLen, "OpenCL buffers failed (%d)", e); opencl_destroy(b); return NULL; }
  size_t mwg = 0; cl_ulong mem = 0; cl_uint cu = 0;
  if (CL.GetDeviceInfo(c->dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof mwg, &mwg, NULL) != CL_SUCCESS || !mwg) mwg = 64;
  if (CL.GetDeviceInfo(c->dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof c->maxAlloc, &c->maxAlloc, NULL) != CL_SUCCESS || !c->maxAlloc) c->maxAlloc = (cl_ulong)-1;
  if (CL.GetDeviceInfo(c->dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof c->localMem, &c->localMem, NULL) != CL_SUCCESS) c->localMem = 0;
  CL.GetDeviceInfo(c->dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof mem, &mem, NULL);
  CL.GetDeviceInfo(c->dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof cu, &cu, NULL);
  c->devWg = c->kWg = c->swkWg = c->swgkWg = mwg;
  b->name = "opencl"; b->eval = opencl_eval; b->evalBattles = opencl_eval_battles; b->destroy = opencl_destroy;
  b->wg = 64; b->chunkMs = 40; b->maxWg = (int)mwg; b->memBytes = (double)mem;
  // hand-off bound: about four 32-wide groups per compute unit (well short of filling the GPU), at least 2048
  b->tail.on = 1; b->tail.cap = cu * 128 > 2048 ? (int)(cu * 128) : 2048; b->tail.threads = br_cpu_count();
  char name[200] = ""; CL.GetDeviceInfo(c->dev, CL_DEVICE_NAME, sizeof name, name, NULL); name[sizeof name - 1] = 0;
  snprintf(b->info, sizeof b->info, "OpenCL, %s", name);
  return b;
}
