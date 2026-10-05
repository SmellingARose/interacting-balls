// Metal backend (Apple GPUs; the AMD and Intel GPUs of Intel Macs). Flights run as one GPU thread each, `chunk` physics
// steps per dispatch, until all have ended or the CPU takes the rest (BrTail). Apple GPUs share memory with the
// CPU, so buffers are shared; a GPU with its own memory gets managed buffers (a copy on each side, synchronised
// explicitly), since it would otherwise read weights and flight states across the bus on every step.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>
#include "backend.h"
#include "threads.h"

typedef struct {
  id<MTLDevice> dev; id<MTLCommandQueue> q;
  id<MTLComputePipelineState> pipe, swPipe, swGroupPipe; int pipeMaxW, pipeSwW, grpSwW, grpStride, grpBadSwW, grpBadStride;
  id<MTLBuffer> state, weights, scen, params, alive, idx, flags, traj, wD, bat;
  int managed;
} MetalImpl;

// A buffer of at least `bytes` (nil when the device cannot hold one that big).
static id<MTLBuffer> ensure(MetalImpl* m, id<MTLBuffer> b, size_t bytes) {
  if (b && b.length >= bytes) return b;
  if (bytes > m->dev.maxBufferLength) return nil;
  return [m->dev newBufferWithLength:(bytes < 16 ? 16 : bytes) options:m->managed ? MTLResourceStorageModeManaged : MTLResourceStorageModeShared];
}
// The CPU wrote the first n bytes (managed buffers: the GPU copy must be updated).
static void wrote(MetalImpl* m, id<MTLBuffer> b, size_t n) { if (m->managed && n) [b didModifyRange:NSMakeRange(0, n)]; }

static MTLCompileOptions* compile_options(void) {
  MTLCompileOptions* opt = [MTLCompileOptions new];
#if defined(__MAC_15_0) && __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_15_0   // older SDKs have no mathMode
  if (@available(macOS 15.0, *)) { opt.mathMode = MTLMathModeFast; return opt; }
#endif
  _Pragma("clang diagnostic push") _Pragma("clang diagnostic ignored \"-Wdeprecated-declarations\"") opt.fastMathEnabled = YES; _Pragma("clang diagnostic pop")
  return opt;
}
static id<MTLComputePipelineState> make_pipe(MetalImpl* m, id<MTLLibrary> lib, NSString* name) {
  id<MTLFunction> fn = [lib newFunctionWithName:name]; NSError* e = nil;
  id<MTLComputePipelineState> p = fn ? [m->dev newComputePipelineStateWithFunction:fn error:&e] : nil;
  if (!p) fprintf(stderr, "Metal pipeline %s failed: %s\n", name.UTF8String, e ? e.localizedDescription.UTF8String : "kernel missing");
  return p;
}

// Flight and thread-per-battle kernels, compiled for these layer widths. A failed build leaves no pipeline behind, so a
// later call cannot mistake it for a built one.
static int build_pipeline(MetalImpl* m, int maxW, int swW) {
  if (m->pipe && m->swPipe && m->pipeMaxW == maxW && m->pipeSwW == swW) return 0;
  m->pipe = m->swPipe = nil; m->pipeMaxW = m->pipeSwW = 0;
  NSString* src = [NSString stringWithFormat:@"#include <metal_stdlib>\nusing namespace metal;\n#define BR_METAL 1\n#define MAXW %d\n#define SW_MAXW %d\n%s\n%s\n",
                   maxW, swW, BR_SRC_SIM_CORE, BR_SRC_METAL];
  NSError* e = nil;
  id<MTLLibrary> lib = [m->dev newLibraryWithSource:src options:compile_options() error:&e];
  if (!lib) { fprintf(stderr, "Metal compile failed: %s\n", e.localizedDescription.UTF8String); return -1; }
  id<MTLComputePipelineState> p = make_pipe(m, lib, @"br_kernel"), sp = p ? make_pipe(m, lib, @"br_sw_kernel") : nil;
  if (!p || !sp) return -1;
  m->pipe = p; m->swPipe = sp; m->pipeMaxW = maxW; m->pipeSwW = swW;
  return 0;
}

// The work-group battle kernel keeps the battle in threadgroup memory: built per battle size (SW_STRIDE floats).
// 1: ready; 0: this size does not fit the device's threadgroup memory or failed to build (use a thread per battle).
static int build_group(MetalImpl* m, int swW, int stride) {
  if (m->swGroupPipe && m->grpSwW == swW && m->grpStride == stride) return 1;
  if ((m->grpBadSwW == swW && m->grpBadStride == stride) || sizeof(float) * ((size_t)stride + 2 * (size_t)swW) > m->dev.maxThreadgroupMemoryLength) return 0;
  m->swGroupPipe = nil; m->grpSwW = m->grpStride = 0;
  NSString* src = [NSString stringWithFormat:@"#include <metal_stdlib>\nusing namespace metal;\n#define BR_METAL 1\n#define BR_SW_LOCAL 1\n#define BR_SG threadgroup\n#define MAXW 32\n#define SW_MAXW %d\n#define SW_STRIDE %d\n%s\n%s\n",
                   swW, stride, BR_SRC_SIM_CORE, BR_SRC_METAL];
  NSError* e = nil;
  id<MTLLibrary> lib = [m->dev newLibraryWithSource:src options:compile_options() error:&e];
  id<MTLComputePipelineState> p = lib ? make_pipe(m, lib, @"br_sw_group_kernel") : nil;
  if (!lib) fprintf(stderr, "Metal swarm group compile failed: %s\n", e.localizedDescription.UTF8String);
  if (!p || p.staticThreadgroupMemoryLength > m->dev.maxThreadgroupMemoryLength) {
    fprintf(stderr, "Metal: no work-group kernel for this battle size; one thread per battle instead\n");
    m->grpBadSwW = swW; m->grpBadStride = stride; return 0;
  }
  m->swGroupPipe = p; m->grpSwW = swW; m->grpStride = stride;
  return 1;
}

// One dispatch over the P->nActive items listed in idx (group: a threadgroup per item); then reads the live count.
// Returns it, or -1 when the GPU failed (the message says why).
static long run_chunk(MetalImpl* m, id<MTLComputePipelineState> pipe, __unsafe_unretained id<MTLBuffer> const* bufs, int nb,
                     NSUInteger tg, int group, const BrParams* P, double* ms) {
  memcpy(m->params.contents, P, sizeof(BrParams)); wrote(m, m->params, sizeof(BrParams));
  *(uint32_t*)m->alive.contents = 0; wrote(m, m->alive, sizeof(uint32_t));
  id<MTLCommandBuffer> cb = [m->q commandBuffer];
  if (!cb) { fprintf(stderr, "Metal: no command buffer (GPU removed?)\n"); return -1; }
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:pipe];
  for (int i = 0; i < nb; i++) [enc setBuffer:bufs[i] offset:0 atIndex:(NSUInteger)i];
  NSUInteger n = (NSUInteger)P->nActive, groups = group ? n : (n + tg - 1) / tg;   // the kernels skip threads past nActive
  [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  [enc endEncoding];
  if (m->managed) { id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder]; [bl synchronizeResource:m->alive]; [bl synchronizeResource:m->flags]; [bl endEncoding]; }
  double t0 = br_now();
  [cb commit]; [cb waitUntilCompleted];
  *ms = (br_now() - t0) * 1000;
  if (cb.status != MTLCommandBufferStatusCompleted) { fprintf(stderr, "Metal error: %s\n", cb.error ? cb.error.localizedDescription.UTF8String : "command buffer failed"); return -1; }
  return (long)*(const uint32_t*)m->alive.contents;
}
// Managed buffers: bring the GPU's flight states to the CPU copy (once per call: a second time would overwrite what the
// CPU tail flew).
static int sync_state(MetalImpl* m) {
  if (!m->managed) return 0;
  id<MTLCommandBuffer> cb = [m->q commandBuffer]; if (!cb) return -1;
  id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder]; [bl synchronizeResource:m->state]; [bl endEncoding];
  [cb commit]; [cb waitUntilCompleted];
  if (cb.status != MTLCommandBufferStatusCompleted) { fprintf(stderr, "Metal error: %s\n", cb.error ? cb.error.localizedDescription.UTF8String : "copy failed"); return -1; }
  return 0;
}
static int no_buffers(MetalImpl* m, size_t bytes) {
  fprintf(stderr, "Metal: cannot allocate GPU buffers for this batch (%.0f MB; largest allowed %.0f MB)\n", bytes / 1e6, m->dev.maxBufferLength / 1e6);
  return -1;
}
static int metal_eval(Backend* b, const float* weights, int nGenomes, const BrScen* scen, const float* traj, size_t trajFloats, BrParams* P, float* out) {
  @autoreleasepool {
    MetalImpl* m = (MetalImpl*)b->impl;
    P->nRoll = nGenomes * P->S;
    if (P->nRoll <= 0) return 0;
    if (build_pipeline(m, br_max_width(P), m->pipe ? m->pipeSwW : br_sw_max_width(P))) return -1;
    size_t sB = sizeof(float) * (size_t)P->nRoll * P->stride, wB = sizeof(float) * (size_t)nGenomes * P->nw, cB = sizeof(BrScen) * (size_t)P->S;
    size_t iB = sizeof(int) * (size_t)P->nRoll, tB = sizeof(float) * (traj && trajFloats ? trajFloats : 4);
    m->state = ensure(m, m->state, sB); m->weights = ensure(m, m->weights, wB); m->scen = ensure(m, m->scen, cB);
    m->params = ensure(m, m->params, sizeof(BrParams)); m->alive = ensure(m, m->alive, sizeof(uint32_t));
    m->idx = ensure(m, m->idx, iB); m->flags = ensure(m, m->flags, iB); m->traj = ensure(m, m->traj, tB);
    if (!m->state || !m->weights || !m->scen || !m->params || !m->alive || !m->idx || !m->flags || !m->traj) return no_buffers(m, sB + wB + tB);
    if (traj && trajFloats) { memcpy(m->traj.contents, traj, tB); wrote(m, m->traj, tB); }
    int* idx = (int*)m->idx.contents; const int* flags = (const int*)m->flags.contents;
    for (int r = 0; r < P->nRoll; r++) idx[r] = r;
    P->nActive = P->nRoll;
    memcpy(m->weights.contents, weights, wB); memcpy(m->scen.contents, scen, cB);
    br_init_states((float*)m->state.contents, P, scen);
    wrote(m, m->idx, iB); wrote(m, m->weights, wB); wrote(m, m->scen, cB); wrote(m, m->state, sB);
    br_tail_begin(&b->tail, (const float*)m->state.contents, P, weights, NULL, NULL, NULL, scen, traj);
    NSUInteger tg = m->pipe.maxTotalThreadsPerThreadgroup; b->maxWg = (int)tg;
    NSUInteger want = b->wg > 0 ? (NSUInteger)b->wg : 64; if (tg > want) tg = want;
    __unsafe_unretained id<MTLBuffer> bufs[8] = { m->state, m->weights, m->scen, m->params, m->alive, m->idx, m->flags, m->traj };
    double target = b->chunkMs > 0 ? b->chunkMs : 40;
    int maxSteps = (int)(P->maxT / P->dt) + 8, done = 0, chunk = br_chunk_first(b, P, 0, target), synced = 0;
    while (done < maxSteps) {
      P->chunk = chunk; double ms = 0;
      long live = run_chunk(m, m->pipe, bufs, 8, tg, 0, P, &ms);
      if (live < 0) return -1;
      if (!done || live == P->nRoll) br_chunk_seen(b, P, 0, chunk, ms);
      done += chunk;
      if (live == 0) break;
      if (br_tail_due(&b->tail, P, (int)live, P->nActive, ms, chunk)) {
        if (sync_state(m)) return -1;
        synced = 1;
        if (br_tail_run(&b->tail, (float*)m->state.contents, P, idx, P->nActive, weights, NULL, NULL, NULL, scen, traj)) return -1;
        break;
      }
      chunk = br_chunk_next(P, chunk, ms, target);
      if (live < 0.7 * P->nActive) {   // compact once ≥30% have ended: keeps SIMD groups full of live flights
        int n = 0; for (int i = 0; i < P->nActive; i++) { int r = idx[i]; if (flags[r]) idx[n++] = r; }
        P->nActive = n; wrote(m, m->idx, sizeof(int) * (size_t)n);
      }
    }
    if (!synced && sync_state(m)) return -1;
    br_collect((const float*)m->state.contents, P, out);
    return 0;
  }
}

// Swarm battles: same chunked dispatch, compaction and CPU tail as flights; a thread or a threadgroup per battle.
static int metal_eval_battles(Backend* b, const float* attW, const float* defW, const int* bat, int nBattles, const BrScen* scen,
                              const float* start, size_t startFloats, BrParams* P, float* out) {
  (void)startFloats;
  @autoreleasepool {
    MetalImpl* m = (MetalImpl*)b->impl;
    P->nRoll = nBattles;
    if (nBattles <= 0) return 0;
    int swW = br_sw_max_width(P);
    if (build_pipeline(m, m->pipe ? m->pipeMaxW : br_max_width(P), swW)) return -1;
    int gThreads = 32, group = br_sw_group_layout(b, P, &gThreads);
    if (group) group = build_group(m, swW, P->stride);
    int nS = 0, nA = 0, nD = 0;
    for (int r = 0; r < nBattles; r++) { if (bat[r * 3 + 2] >= nS) nS = bat[r * 3 + 2] + 1; if (bat[r * 3] >= nA) nA = bat[r * 3] + 1; if (bat[r * 3 + 1] >= nD) nD = bat[r * 3 + 1] + 1; }
    size_t sB = sizeof(float) * (size_t)nBattles * P->stride, aB = sizeof(float) * (P->attAI ? (size_t)nA * P->attNw : 1), dB = sizeof(float) * (P->defAI ? (size_t)nD * P->defNw : 1);
    size_t cB = sizeof(BrScen) * (size_t)nS, iB = sizeof(int) * (size_t)nBattles;
    m->state = ensure(m, m->state, sB); m->weights = ensure(m, m->weights, aB); m->wD = ensure(m, m->wD, dB); m->scen = ensure(m, m->scen, cB);
    m->params = ensure(m, m->params, sizeof(BrParams)); m->alive = ensure(m, m->alive, sizeof(uint32_t));
    m->idx = ensure(m, m->idx, iB); m->flags = ensure(m, m->flags, iB); m->bat = ensure(m, m->bat, 3 * iB);
    if (!m->state || !m->weights || !m->wD || !m->scen || !m->params || !m->alive || !m->idx || !m->flags || !m->bat) return no_buffers(m, sB + aB + dB);
    if (P->attAI) { memcpy(m->weights.contents, attW, aB); wrote(m, m->weights, aB); }
    if (P->defAI) { memcpy(m->wD.contents, defW, dB); wrote(m, m->wD, dB); }
    memcpy(m->scen.contents, scen, cB); memcpy(m->bat.contents, bat, 3 * iB);
    int* idx = (int*)m->idx.contents; const int* flags = (const int*)m->flags.contents;
    for (int r = 0; r < nBattles; r++) idx[r] = r;
    P->nActive = nBattles;
    br_sw_init_states((float*)m->state.contents, P, scen, bat, nBattles, start);
    wrote(m, m->scen, cB); wrote(m, m->bat, 3 * iB); wrote(m, m->idx, iB); wrote(m, m->state, sB);
    br_tail_begin(&b->tail, (const float*)m->state.contents, P, NULL, attW, defW, bat, scen, NULL);
    id<MTLComputePipelineState> pipe = group ? m->swGroupPipe : m->swPipe;
    NSUInteger tg = pipe.maxTotalThreadsPerThreadgroup; if (!group) b->maxWg = (int)tg;
    NSUInteger want = group ? (NSUInteger)gThreads : b->wg > 0 ? (NSUInteger)b->wg : 64; if (tg > want) tg = want;
    __unsafe_unretained id<MTLBuffer> bufs[9] = { m->state, m->weights, m->wD, m->scen, m->params, m->alive, m->idx, m->flags, m->bat };
    double target = b->chunkMs > 0 ? b->chunkMs : 40; int layout = group ? (int)tg : 0, chunk = br_chunk_first(b, P, layout, target), synced = 0;
    int maxSteps = (int)(P->maxT / P->dt) + 8, done = 0;
    while (done < maxSteps) {
      P->chunk = chunk; double ms = 0;
      long live = run_chunk(m, pipe, bufs, 9, tg, group, P, &ms);
      if (live < 0) return -1;
      if (!done || live == P->nRoll) br_chunk_seen(b, P, layout, chunk, ms);
      done += chunk;
      if (live == 0) break;
      if (br_tail_due(&b->tail, P, (int)live, P->nActive, ms, chunk)) {
        if (sync_state(m)) return -1;
        synced = 1;
        if (br_tail_run(&b->tail, (float*)m->state.contents, P, idx, P->nActive, NULL, attW, defW, bat, scen, NULL)) return -1;
        break;
      }
      chunk = br_chunk_next(P, chunk, ms, target);
      if (live < 0.7 * P->nActive) { int n = 0; for (int i = 0; i < P->nActive; i++) { int r = idx[i]; if (flags[r]) idx[n++] = r; } P->nActive = n; wrote(m, m->idx, sizeof(int) * (size_t)n); }
    }
    if (!synced && sync_state(m)) return -1;
    br_sw_collect((const float*)m->state.contents, P, nBattles, out);
    return 0;
  }
}

static void metal_destroy(Backend* b) { MetalImpl* m = (MetalImpl*)b->impl;
  m->dev = nil; m->q = nil; m->pipe = m->swPipe = m->swGroupPipe = nil; m->state = m->weights = m->scen = m->params = m->alive = m->idx = m->flags = m->traj = m->wD = m->bat = nil; free(m); free(b); }

Backend* metal_backend_create(char* err, int errLen) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { snprintf(err, errLen, "no Metal device"); return NULL; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    if (!q) { snprintf(err, errLen, "no Metal command queue"); return NULL; }
    Backend* b = (Backend*)calloc(1, sizeof(Backend));
    MetalImpl* m = (MetalImpl*)calloc(1, sizeof(MetalImpl));
    if (!b || !m) { free(b); free(m); snprintf(err, errLen, "out of memory"); return NULL; }
    m->dev = dev; m->q = q;
    if (@available(macOS 10.15, *)) m->managed = !dev.hasUnifiedMemory; else m->managed = !dev.isLowPower;
    b->name = "metal"; b->eval = metal_eval; b->evalBattles = metal_eval_battles; b->destroy = metal_destroy; b->impl = m;
    b->wg = 64; b->chunkMs = 40; b->maxWg = 1024; b->memBytes = (double)dev.recommendedMaxWorkingSetSize;
    // Metal does not report the GPU's core count: 2048 threads (64 SIMD groups) bounds a hand-off the timing misjudged
    b->tail.on = 1; b->tail.cap = 2048; b->tail.threads = br_cpu_count();
    snprintf(b->info, sizeof b->info, "Metal, %s", dev.name.UTF8String);
    return b;
  }
}
