// CPU backend: every core flies whole flights; work is handed out one flight at a time from an atomic counter.
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "backend.h"
#include "threads.h"

typedef struct { int threads; } CpuImpl;
typedef struct {
  const float* w; const BrScen* scen; const float* traj; const BrParams* P; float* out; br_atomic_int next;
} CpuJob;

// One flight's result: closest approach, hit, flight time, physics steps, fraction of mid-flight below 5 m.
static void br_result(const float* g, float* o) {
  o[0] = g[S_MIND]; o[1] = g[S_HIT]; o[2] = g[S_T]; o[3] = g[S_K]; o[4] = g[S_MID] > 0 ? g[S_LOW] / g[S_MID] : 0;
}

// Repack one genome per layer from [W out×in | b] to [b | Wᵀ in×out] for the CPU network loop.
void br_transpose_genome(const float* g, float* t, const BrParams* P) {
  int off = 0;
  for (int l = 0; l < P->nl; l++) { int ni = P->arch[l], no = P->arch[l + 1];
    for (int j = 0; j < no; j++) t[off + j] = g[off + no * ni + j];
    for (int j = 0; j < no; j++) for (int i = 0; i < ni; i++) t[off + no + i * no + j] = g[off + j * ni + i];
    off += no * ni + no; }
}

static void cpu_worker(void* arg) {
  CpuJob* j = (CpuJob*)arg; const BrParams* P = j->P;
  float* st = (float*)malloc(sizeof(float) * (size_t)P->stride);
  float* wt = (float*)malloc(sizeof(float) * (size_t)P->nw); int have = -1;
  for (;;) {
    int r = br_atomic_fetch_add(&j->next, 1);
    if (r >= P->nRoll) break;
    int gi = r / P->S, si = r % P->S;
    if (gi != have) { br_transpose_genome(j->w + (size_t)gi * P->nw, wt, P); have = gi; }
    br_init(st, P, &j->scen[si]);
    br_run(st, P, &j->scen[si], wt, j->traj, 1 << 30);
    br_result(st, j->out + (size_t)r * BR_OUT);
  }
  free(st); free(wt);
}

static int cpu_eval(Backend* b, const float* weights, int nGenomes, const BrScen* scen, const float* traj, size_t trajFloats, BrParams* P, float* out) {
  (void)trajFloats;
  CpuImpl* c = (CpuImpl*)b->impl;
  P->nRoll = nGenomes * P->S;
  CpuJob job = { weights, scen, traj, P, out, 0 };
  br_run_threads(c->threads, cpu_worker, &job);
  return 0;
}
static void cpu_destroy(Backend* b) { free(b->impl); free(b); }

Backend* cpu_backend_create(int threads) {
  Backend* b = (Backend*)calloc(1, sizeof(Backend));
  CpuImpl* c = (CpuImpl*)calloc(1, sizeof(CpuImpl));
  c->threads = threads > 0 ? threads : br_cpu_count();
  b->name = "cpu"; b->eval = cpu_eval; b->destroy = cpu_destroy; b->impl = c;
  snprintf(b->info, sizeof b->info, "CPU, %d threads", c->threads);
  return b;
}

// ---- helpers shared by the GPU hosts
int br_max_width(const BrParams* P) { int m = 4; for (int l = 0; l <= P->nl; l++) if (P->arch[l] > m) m = P->arch[l]; return (m + 3) & ~3; }
void br_init_states(float* state, const BrParams* P, const BrScen* scen) {
  for (int r = 0; r < P->nRoll; r++) br_init(state + (size_t)r * P->stride, P, &scen[r % P->S]);
}
void br_collect(const float* state, const BrParams* P, float* out) {
  for (int r = 0; r < P->nRoll; r++) br_result(state + (size_t)r * P->stride, out + (size_t)r * BR_OUT);
}
