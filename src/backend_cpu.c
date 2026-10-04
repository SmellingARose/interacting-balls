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
void br_transpose_layers(const float* g, float* t, int nl, const int* arch) {
  int off = 0;
  for (int l = 0; l < nl; l++) { int ni = arch[l], no = arch[l + 1];
    for (int j = 0; j < no; j++) t[off + j] = g[off + no * ni + j];
    for (int j = 0; j < no; j++) for (int i = 0; i < ni; i++) t[off + no + i * no + j] = g[off + j * ni + i];
    off += no * ni + no; }
}
void br_transpose_genome(const float* g, float* t, const BrParams* P) { br_transpose_layers(g, t, P->nl, P->arch); }

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
// ---- swarm: every core flies whole battles, handed out one at a time
typedef struct { const float *wA, *wD; const int* bat; int n; const BrScen* scen; const float* start; const BrParams* P; float* out; br_atomic_int next; } CpuBattleJob;
// A side's layer sizes: its own inputs and outputs around the shared hidden layers.
static void side_arch(const BrParams* P, int nin, int nout, int* arch) {
  for (int l = 0; l <= P->nl; l++) arch[l] = P->arch[l];
  arch[0] = nin; arch[P->nl] = nout;
}
static void cpu_battle_worker(void* arg) {
  CpuBattleJob* j = (CpuBattleJob*)arg; const BrParams* P = j->P;
  int aA[BR_MAXL + 1], aD[BR_MAXL + 1]; side_arch(P, P->attNin, P->attNout, aA); side_arch(P, P->defNin, P->defNout, aD);
  float* st = (float*)malloc(sizeof(float) * (size_t)br_sw_stride(P));
  float* tA = (float*)malloc(sizeof(float) * (size_t)(P->attNw > 0 ? P->attNw : 1));
  float* tD = (float*)malloc(sizeof(float) * (size_t)(P->defNw > 0 ? P->defNw : 1));
  int haveA = -1, haveD = -1;
  for (;;) {
    int r = br_atomic_fetch_add(&j->next, 1); if (r >= j->n) break;
    int ga = j->bat[r * 3], gd = j->bat[r * 3 + 1], si = j->bat[r * 3 + 2];
    if (P->attAI && ga != haveA) { br_transpose_layers(j->wA + (size_t)ga * P->attNw, tA, P->nl, aA); haveA = ga; }
    if (P->defAI && gd != haveD) { br_transpose_layers(j->wD + (size_t)gd * P->defNw, tD, P->nl, aD); haveD = gd; }
    br_sw_init(st, P, &j->scen[si], j->start);
    br_sw_run(st, P, &j->scen[si], tA, tD, 1 << 30);
    br_sw_result(st, P, j->out + (size_t)r * BR_SWOUT);
  }
  free(st); free(tA); free(tD);
}
static int cpu_eval_battles(Backend* b, const float* attW, const float* defW, const int* bat, int nBattles, const BrScen* scen,
                            const float* start, size_t startFloats, BrParams* P, float* out) {
  (void)startFloats;
  CpuImpl* c = (CpuImpl*)b->impl;
  CpuBattleJob job = { attW, defW, bat, nBattles, scen, start, P, out, 0 };
  br_run_threads(c->threads, cpu_battle_worker, &job);
  return 0;
}
static void cpu_destroy(Backend* b) { free(b->impl); free(b); }

Backend* cpu_backend_create(int threads) {
  Backend* b = (Backend*)calloc(1, sizeof(Backend));
  CpuImpl* c = (CpuImpl*)calloc(1, sizeof(CpuImpl));
  c->threads = threads > 0 ? threads : br_cpu_count();
  b->name = "cpu"; b->eval = cpu_eval; b->evalBattles = cpu_eval_battles; b->destroy = cpu_destroy; b->impl = c;
  snprintf(b->info, sizeof b->info, "CPU, %d threads", c->threads);
  return b;
}

// ---- helpers shared by the GPU hosts
int br_sw_max_width(const BrParams* P) {
  int m = BR_NI + 2 * BR_SW_MAXK * SW_NF_NB;
  if (P->mode == BR_MODE_SWARM) { int v[5] = { P->attNin, P->defNin, P->attNout, P->defNout, P->arch[1] }; m = 16; for (int i = 0; i < 5; i++) if (v[i] > m) m = v[i]; for (int l = 1; l < P->nl; l++) if (P->arch[l] > m) m = P->arch[l]; }
  return (m + 3) & ~3;
}
int br_max_width(const BrParams* P) { int m = 4; for (int l = 0; l <= P->nl; l++) if (P->arch[l] > m) m = P->arch[l]; return (m + 3) & ~3; }
void br_init_states(float* state, const BrParams* P, const BrScen* scen) {
  for (int r = 0; r < P->nRoll; r++) br_init(state + (size_t)r * P->stride, P, &scen[r % P->S]);
}
void br_collect(const float* state, const BrParams* P, float* out) {
  for (int r = 0; r < P->nRoll; r++) br_result(state + (size_t)r * P->stride, out + (size_t)r * BR_OUT);
}
// Small battles keep one GPU thread each (many battles per SIMD group); from about 12 balls a battle gets a
// work-group with a thread per ball (unless a side uses a commander brain).
int br_sw_group_layout(const Backend* b, const BrParams* P, int* threads) {
  int nB = P->attN + P->defN, t = (nB + 31) / 32 * 32; *threads = t > 64 ? 64 : t;
  if (b->swLayout) return b->swLayout == 2;
  return nB >= 12 && !(P->attAI && P->attCmd) && !(P->defAI && P->defCmd);   // a commander pass runs on one thread: per-battle threads suit it better
}
void br_sw_init_states(float* state, const BrParams* P, const BrScen* scen, const int* bat, int n, const float* start) {
  for (int r = 0; r < n; r++) br_sw_init(state + (size_t)r * P->stride, P, &scen[bat[r * 3 + 2]], start);
}
void br_sw_collect(const float* state, const BrParams* P, int n, float* out) {
  for (int r = 0; r < n; r++) br_sw_result(state + (size_t)r * P->stride, P, out + (size_t)r * BR_SWOUT);
}
