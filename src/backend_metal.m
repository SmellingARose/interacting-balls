// Metal backend (Apple GPUs). Flights run as one GPU thread each, `chunk` physics steps per dispatch, until all
// have ended. Buffers use shared (unified) memory, so the CPU writes weights and reads results without copies.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>
#include "backend.h"

typedef struct {
  id<MTLDevice> dev; id<MTLCommandQueue> q;
  id<MTLComputePipelineState> pipe, swPipe; int pipeMaxW, pipeSwW;
  id<MTLBuffer> state, weights, scen, params, alive, idx, flags, traj, wD, bat;
  double chunkMs;   // target GPU time per dispatch
  int chunk;
} MetalImpl;

static id<MTLBuffer> ensure(MetalImpl* m, id<MTLBuffer> b, size_t bytes) {
  if (b && b.length >= bytes) return b;
  return [m->dev newBufferWithLength:(bytes < 16 ? 16 : bytes) options:MTLResourceStorageModeShared];
}

static int build_pipeline(MetalImpl* m, int maxW, int swW, char* err, int errLen) {
  if (m->pipe && m->pipeMaxW == maxW && m->pipeSwW == swW) return 0;
  NSString* src = [NSString stringWithFormat:@"#include <metal_stdlib>\nusing namespace metal;\n#define BR_METAL 1\n#define MAXW %d\n#define SW_MAXW %d\n%s\n%s\n",
                   maxW, swW, BR_SRC_SIM_CORE, BR_SRC_METAL];
  MTLCompileOptions* opt = [MTLCompileOptions new];
  if (@available(macOS 15.0, *)) opt.mathMode = MTLMathModeFast;
  else { _Pragma("clang diagnostic push") _Pragma("clang diagnostic ignored \"-Wdeprecated-declarations\"") opt.fastMathEnabled = YES; _Pragma("clang diagnostic pop") }
  NSError* e = nil;
  id<MTLLibrary> lib = [m->dev newLibraryWithSource:src options:opt error:&e];
  if (!lib) { snprintf(err, errLen, "Metal compile failed: %s", e.localizedDescription.UTF8String); return -1; }
  id<MTLFunction> fn = [lib newFunctionWithName:@"br_kernel"];
  m->pipe = [m->dev newComputePipelineStateWithFunction:fn error:&e];
  if (!m->pipe) { snprintf(err, errLen, "Metal pipeline failed: %s", e.localizedDescription.UTF8String); return -1; }
  m->swPipe = [m->dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"br_sw_kernel"] error:&e];
  if (!m->swPipe) { snprintf(err, errLen, "Metal swarm pipeline failed: %s", e.localizedDescription.UTF8String); return -1; }
  m->pipeMaxW = maxW; m->pipeSwW = swW;
  return 0;
}

static int metal_eval(Backend* b, const float* weights, int nGenomes, const BrScen* scen, const float* traj, size_t trajFloats, BrParams* P, float* out) {
  @autoreleasepool {
    MetalImpl* m = (MetalImpl*)b->impl; char err[512];
    if (build_pipeline(m, br_max_width(P), m->pipe ? m->pipeSwW : br_sw_max_width(P), err, sizeof err)) { fprintf(stderr, "%s\n", err); return -1; }
    P->nRoll = nGenomes * P->S;
    m->state   = ensure(m, m->state,   sizeof(float) * (size_t)P->nRoll * P->stride);
    m->weights = ensure(m, m->weights, sizeof(float) * (size_t)nGenomes * P->nw);
    m->scen    = ensure(m, m->scen,    sizeof(BrScen) * (size_t)P->S);
    m->params  = ensure(m, m->params,  sizeof(BrParams));
    m->alive   = ensure(m, m->alive,   sizeof(uint32_t));
    m->idx     = ensure(m, m->idx,     sizeof(int) * (size_t)P->nRoll);
    m->flags   = ensure(m, m->flags,   sizeof(int) * (size_t)P->nRoll);
    m->traj    = ensure(m, m->traj,    sizeof(float) * (traj ? trajFloats : 4));
    if (traj && trajFloats) memcpy(m->traj.contents, traj, sizeof(float) * trajFloats);
    int* idx = (int*)m->idx.contents; const int* flags = (const int*)m->flags.contents;
    for (int r = 0; r < P->nRoll; r++) idx[r] = r;
    P->nActive = P->nRoll;
    memcpy(m->weights.contents, weights, sizeof(float) * (size_t)nGenomes * P->nw);
    memcpy(m->scen.contents, scen, sizeof(BrScen) * (size_t)P->S);
    br_init_states((float*)m->state.contents, P, scen);
    NSUInteger tg = m->pipe.maxTotalThreadsPerThreadgroup; b->maxWg = (int)tg;
    NSUInteger want = b->wg > 0 ? (NSUInteger)b->wg : 64; if (tg > want) tg = want;
    m->chunkMs = b->chunkMs > 0 ? b->chunkMs : 40;
    int maxSteps = (int)(P->maxT / P->dt) + 8, done = 0;
    while (done < maxSteps) {
      P->chunk = m->chunk;
      memcpy(m->params.contents, P, sizeof(BrParams));
      *(uint32_t*)m->alive.contents = 0;
      id<MTLCommandBuffer> cb = [m->q commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:m->pipe];
      [enc setBuffer:m->state offset:0 atIndex:0]; [enc setBuffer:m->weights offset:0 atIndex:1];
      [enc setBuffer:m->scen offset:0 atIndex:2]; [enc setBuffer:m->params offset:0 atIndex:3]; [enc setBuffer:m->alive offset:0 atIndex:4];
      [enc setBuffer:m->idx offset:0 atIndex:5]; [enc setBuffer:m->flags offset:0 atIndex:6]; [enc setBuffer:m->traj offset:0 atIndex:7];
      [enc dispatchThreads:MTLSizeMake((NSUInteger)P->nActive, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      [enc endEncoding];
      NSDate* t0 = [NSDate date];
      [cb commit]; [cb waitUntilCompleted];
      if (cb.error) { fprintf(stderr, "Metal error: %s\n", cb.error.localizedDescription.UTF8String); return -1; }
      double ms = -[t0 timeIntervalSinceNow] * 1000.0;
      done += m->chunk;
      // adapt the dispatch size toward the target time (long enough to amortise overhead, short enough to stay responsive)
      double scale = m->chunkMs / (ms > 0.5 ? ms : 0.5);
      int next = (int)(m->chunk * (scale > 4 ? 4 : scale < 0.25 ? 0.25 : scale));
      m->chunk = next < 16 ? 16 : next > 100000 ? 100000 : next;
      if (*(uint32_t*)m->alive.contents == 0) break;
      uint32_t live = *(uint32_t*)m->alive.contents;
      if (live < 0.7 * P->nActive) {   // compact once ≥30% have ended: keeps SIMD groups full of live flights
        int n = 0; for (int i = 0; i < P->nActive; i++) { int r = idx[i]; if (flags[r]) idx[n++] = r; }
        P->nActive = n;
      }
    }
    br_collect((const float*)m->state.contents, P, out);
    return 0;
  }
}

// Swarm battles: same chunked dispatch and compaction as flights; one thread per battle.
static int metal_eval_battles(Backend* b, const float* attW, const float* defW, const int* bat, int nBattles, const BrScen* scen,
                              const float* start, size_t startFloats, BrParams* P, float* out) {
  (void)startFloats;
  @autoreleasepool {
    MetalImpl* m = (MetalImpl*)b->impl; char err[512];
    if (build_pipeline(m, m->pipe ? m->pipeMaxW : br_max_width(P), br_sw_max_width(P), err, sizeof err)) { fprintf(stderr, "%s\n", err); return -1; }
    int nS = 0; for (int r = 0; r < nBattles; r++) if (bat[r * 3 + 2] >= nS) nS = bat[r * 3 + 2] + 1;
    int nA = 0, nD = 0; for (int r = 0; r < nBattles; r++) { if (bat[r * 3] >= nA) nA = bat[r * 3] + 1; if (bat[r * 3 + 1] >= nD) nD = bat[r * 3 + 1] + 1; }
    P->nRoll = nBattles;
    m->state   = ensure(m, m->state,   sizeof(float) * (size_t)nBattles * P->stride);
    m->weights = ensure(m, m->weights, sizeof(float) * (P->attAI ? (size_t)nA * P->attNw : 1));
    m->wD      = ensure(m, m->wD,      sizeof(float) * (P->defAI ? (size_t)nD * P->defNw : 1));
    m->scen    = ensure(m, m->scen,    sizeof(BrScen) * (size_t)nS);
    m->params  = ensure(m, m->params,  sizeof(BrParams));
    m->alive   = ensure(m, m->alive,   sizeof(uint32_t));
    m->idx     = ensure(m, m->idx,     sizeof(int) * (size_t)nBattles);
    m->flags   = ensure(m, m->flags,   sizeof(int) * (size_t)nBattles);
    m->bat     = ensure(m, m->bat,     sizeof(int) * 3 * (size_t)nBattles);
    if (P->attAI) memcpy(m->weights.contents, attW, sizeof(float) * (size_t)nA * P->attNw);
    if (P->defAI) memcpy(m->wD.contents, defW, sizeof(float) * (size_t)nD * P->defNw);
    memcpy(m->scen.contents, scen, sizeof(BrScen) * (size_t)nS);
    memcpy(m->bat.contents, bat, sizeof(int) * 3 * (size_t)nBattles);
    int* idx = (int*)m->idx.contents; const int* flags = (const int*)m->flags.contents;
    for (int r = 0; r < nBattles; r++) idx[r] = r;
    P->nActive = nBattles;
    br_sw_init_states((float*)m->state.contents, P, scen, bat, nBattles, start);
    NSUInteger tg = m->swPipe.maxTotalThreadsPerThreadgroup;
    NSUInteger want = b->wg > 0 ? (NSUInteger)b->wg : 64; if (tg > want) tg = want;
    double target = b->chunkMs > 0 ? b->chunkMs : 40; int chunk = 64;
    int maxSteps = (int)(P->maxT / P->dt) + 8, done = 0;
    while (done < maxSteps) {
      P->chunk = chunk;
      memcpy(m->params.contents, P, sizeof(BrParams));
      *(uint32_t*)m->alive.contents = 0;
      id<MTLCommandBuffer> cb = [m->q commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:m->swPipe];
      [enc setBuffer:m->state offset:0 atIndex:0]; [enc setBuffer:m->weights offset:0 atIndex:1]; [enc setBuffer:m->wD offset:0 atIndex:2];
      [enc setBuffer:m->scen offset:0 atIndex:3]; [enc setBuffer:m->params offset:0 atIndex:4]; [enc setBuffer:m->alive offset:0 atIndex:5];
      [enc setBuffer:m->idx offset:0 atIndex:6]; [enc setBuffer:m->flags offset:0 atIndex:7]; [enc setBuffer:m->bat offset:0 atIndex:8];
      [enc dispatchThreads:MTLSizeMake((NSUInteger)P->nActive, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      [enc endEncoding];
      NSDate* t0 = [NSDate date];
      [cb commit]; [cb waitUntilCompleted];
      if (cb.error) { fprintf(stderr, "Metal error: %s\n", cb.error.localizedDescription.UTF8String); return -1; }
      double ms = -[t0 timeIntervalSinceNow] * 1000.0;
      done += chunk;
      double scale = target / (ms > 0.5 ? ms : 0.5);
      int next = (int)(chunk * (scale > 4 ? 4 : scale < 0.25 ? 0.25 : scale));
      chunk = next < 8 ? 8 : next > 100000 ? 100000 : next;
      uint32_t live = *(uint32_t*)m->alive.contents;
      if (live == 0) break;
      if (live < 0.7 * P->nActive) { int n = 0; for (int i = 0; i < P->nActive; i++) { int r = idx[i]; if (flags[r]) idx[n++] = r; } P->nActive = n; }
    }
    br_sw_collect((const float*)m->state.contents, P, nBattles, out);
    return 0;
  }
}

static void metal_destroy(Backend* b) { MetalImpl* m = (MetalImpl*)b->impl;
  m->dev = nil; m->q = nil; m->pipe = m->swPipe = nil; m->state = m->weights = m->scen = m->params = m->alive = m->idx = m->flags = m->traj = m->wD = m->bat = nil; free(m); free(b); }

Backend* metal_backend_create(char* err, int errLen) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { snprintf(err, errLen, "no Metal device"); return NULL; }
    Backend* b = (Backend*)calloc(1, sizeof(Backend));
    MetalImpl* m = (MetalImpl*)calloc(1, sizeof(MetalImpl));
    m->dev = dev; m->q = [dev newCommandQueue]; m->chunkMs = 40; m->chunk = 256;
    b->name = "metal"; b->eval = metal_eval; b->evalBattles = metal_eval_battles; b->destroy = metal_destroy; b->impl = m;
    b->wg = 64; b->chunkMs = 40; b->maxWg = 1024; b->memBytes = (double)dev.recommendedMaxWorkingSetSize;
    snprintf(b->info, sizeof b->info, "Metal, %s", dev.name.UTF8String);
    return b;
  }
}
