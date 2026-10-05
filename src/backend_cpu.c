// CPU backend: every core flies whole flights; work is handed out one flight at a time from an atomic counter.
// The same workers also finish the flights or battles still running in a GPU batch (the CPU tail, see BrTail).
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

// The CPU core keeps its layers in fixed arrays (MAXW, SW_MAXW wide) and its battles within BR_SW_MAX* balls; GPUs
// compile their sizes to fit. A shape beyond them (a foreign network file, an unclamped option) is refused, not overrun.
int br_cpu_fits(const BrParams* P) {
  if (P->nl < 1 || P->nl >= BR_MAXL || P->ctrl < 1 || P->S < 1) return 0;
  if (P->mode != BR_MODE_SWARM)
    return P->K >= 0 && P->memN >= 0 && BR_NI * (P->K + 1) + P->memN <= MAXW && br_max_width(P) <= MAXW && P->arch[P->nl] >= 3 &&
           P->stride >= S_HIST + BR_NI * P->K + P->memN;
  return P->attN >= 1 && P->attN <= BR_SW_MAXA && P->defN >= 1 && P->defN <= BR_SW_MAXD && P->swK >= 1 && P->swK <= BR_SW_MAXK &&
         br_sw_max_width(P) <= SW_MAXW && P->stride >= br_sw_stride(P) &&
         P->attNout >= (P->attCmd ? 3 * P->attN : 3) && P->defNout >= (P->defCmd ? 3 * P->defN : 3);
}

// A worker that cannot get its scratch memory takes no work: the others take its share, and the caller sees whether
// everything was flown from the work counter (each worker leaves it at or past the end).
static void cpu_worker(void* arg) {
  CpuJob* j = (CpuJob*)arg; const BrParams* P = j->P;
  float* st = (float*)malloc(sizeof(float) * (size_t)P->stride);
  float* wt = (float*)malloc(sizeof(float) * (size_t)P->nw); int have = -1;
  if (!st || !wt) { free(st); free(wt); return; }
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
  if (!br_cpu_fits(P)) { fprintf(stderr, "CPU: this network does not fit the CPU build (layers up to %d wide)\n", MAXW); return -1; }
  CpuJob job = { weights, scen, traj, P, out, 0 };
  br_run_threads(c->threads < P->nRoll ? c->threads : P->nRoll, cpu_worker, &job);
  if (job.next < P->nRoll) { fprintf(stderr, "CPU: out of memory\n"); return -1; }
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
  if (!st || !tA || !tD) { free(st); free(tA); free(tD); return; }
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
  if (!br_cpu_fits(P)) { fprintf(stderr, "CPU: this battle does not fit the CPU build (layers up to %d wide, %d balls a side)\n", SW_MAXW, BR_SW_MAXA); return -1; }
  CpuBattleJob job = { attW, defW, bat, nBattles, scen, start, P, out, 0 };
  br_run_threads(c->threads < nBattles ? c->threads : nBattles, cpu_battle_worker, &job);
  if (job.next < nBattles) { fprintf(stderr, "CPU: out of memory\n"); return -1; }
  return 0;
}
static void cpu_destroy(Backend* b) { free(b->impl); free(b); }

Backend* cpu_backend_create(int threads) {
  Backend* b = (Backend*)calloc(1, sizeof(Backend));
  CpuImpl* c = (CpuImpl*)calloc(1, sizeof(CpuImpl));
  if (!b || !c) { free(b); free(c); return NULL; }
  c->threads = threads > 0 ? threads : br_cpu_count();
  b->name = "cpu"; b->eval = cpu_eval; b->evalBattles = cpu_eval_battles; b->destroy = cpu_destroy; b->impl = c;
  snprintf(b->info, sizeof b->info, "CPU, %d threads", c->threads);
  return b;
}

// ---- CPU tail of the GPU backends. A dispatch over a few live threads takes about as long as over a full GPU, so
// once the CPU would advance the items still running faster than the GPU does, it flies them to the end (same core
// code, CPU weight layout): usually the last few, nearly all of a small batch that the CPU outruns the GPU on. The
// CPU's speed is measured per network shape (br_tail_begin) and refined by every tail it flies.
static unsigned tail_key(const BrParams* P) {
  int v[] = { P->mode, P->nl, P->nw, P->K, P->rec, P->ctrl, P->arch[0], P->arch[1], P->attN, P->defN, P->attAI, P->defAI,
              P->attCmd, P->defCmd, P->swK, P->attNw, P->defNw };
  unsigned h = 2166136261u; for (size_t i = 0; i < sizeof v / sizeof v[0]; i++) h = (h ^ (unsigned)v[i]) * 16777619u;
  return h;
}
typedef struct {
  float* state; const BrParams* P; const int* idx; int n; const float *w, *wA, *wD; const int* bat; const BrScen* scen; const float* traj;
  br_atomic_int next, slot; double* sec; double* steps;
} TailJob;
// Transposed weights for item r (a flight, or a battle's two sides), reusing what tA / tD already hold.
static void tail_weights(const TailJob* j, int r, float* tA, float* tD, int* haveA, int* haveD, const int* aA, const int* aD) {
  const BrParams* P = j->P;
  if (P->mode != BR_MODE_SWARM) { int gi = r / P->S; if (gi != *haveA) { br_transpose_genome(j->w + (size_t)gi * P->nw, tA, P); *haveA = gi; } return; }
  int ga = j->bat[r * 3], gd = j->bat[r * 3 + 1];
  if (P->attAI && ga != *haveA) { br_transpose_layers(j->wA + (size_t)ga * P->attNw, tA, P->nl, aA); *haveA = ga; }
  if (P->defAI && gd != *haveD) { br_transpose_layers(j->wD + (size_t)gd * P->defNw, tD, P->nl, aD); *haveD = gd; }
}
static void tail_worker(void* arg) {
  TailJob* j = (TailJob*)arg; const BrParams* P = j->P; int sw = P->mode == BR_MODE_SWARM, me = br_atomic_fetch_add(&j->slot, 1);
  int aA[BR_MAXL + 1], aD[BR_MAXL + 1], haveA = -1, haveD = -1;
  size_t nA = sw ? (size_t)P->attNw : (size_t)P->nw, nD = sw ? (size_t)P->defNw : 0;
  float* tA = (float*)malloc(sizeof(float) * (nA ? nA : 1)); float* tD = (float*)malloc(sizeof(float) * (nD ? nD : 1));
  double sec = 0, steps = 0;
  side_arch(P, P->attNin, P->attNout, aA); side_arch(P, P->defNin, P->defNout, aD);   // (read only for battles)
  if (tA && tD) for (;;) {
    int i = br_atomic_fetch_add(&j->next, 1); if (i >= j->n) break;
    int r = j->idx[i]; float* g = j->state + (size_t)r * P->stride;
    if (sw ? g[SWH_DONE] != 0 : g[S_ALIVE] == 0) continue;   // ended in the last dispatch
    tail_weights(j, r, tA, tD, &haveA, &haveD, aA, aD);
    float k0 = g[sw ? SWH_K : S_K]; double t0 = br_now();
    if (sw) br_sw_run(g, P, &j->scen[j->bat[r * 3 + 2]], tA, tD, 1 << 30);
    else br_run(g, P, &j->scen[r % P->S], tA, j->traj, 1 << 30);
    sec += br_now() - t0; steps += g[sw ? SWH_K : S_K] - k0;
  }
  j->sec[me] = sec; j->steps[me] = steps;
  free(tA); free(tD);
}
int br_tail_run(BrTail* t, float* state, const BrParams* P, const int* idx, int n, const float* w, const float* wA, const float* wD,
                const int* bat, const BrScen* scen, const float* traj) {
  int nt = t->threads < n ? t->threads : n; if (nt < 1) nt = 1;
  double* acc = (double*)calloc((size_t)nt * 2, sizeof(double));
  if (!acc) { fprintf(stderr, "CPU: out of memory\n"); return -1; }
  TailJob j = { state, P, idx, n, w, wA, wD, bat, scen, traj, 0, 0, acc, acc + nt };
  br_run_threads(nt, tail_worker, &j);
  int ok = j.next >= n; double sec = 0, steps = 0;
  for (int i = 0; i < nt; i++) { sec += acc[i]; steps += acc[nt + i]; }
  free(acc);
  if (!ok) { fprintf(stderr, "CPU: out of memory\n"); return -1; }
  if (sec > 2e-4 && steps > 0) { double ms = sec * 1000 / steps; t->cpuMs = t->cpuMs > 0 ? 0.5 * (t->cpuMs + ms) : ms; }
  return 0;
}
void br_tail_begin(BrTail* t, const float* state, const BrParams* P, const float* w, const float* wA, const float* wD, const int* bat,
                   const BrScen* scen, const float* traj) {
  unsigned key = tail_key(P);
  if (!t->on || P->nRoll < 1 || !br_cpu_fits(P) || (t->key == key && t->cpuMs > 0)) return;
  t->key = key; t->cpuMs = 0;
  // fly a copy of the first item for a few steps on this thread: a few steps to warm up, then the timed ones
  int sw = P->mode == BR_MODE_SWARM, aA[BR_MAXL + 1], aD[BR_MAXL + 1], haveA = -1, haveD = -1, i0 = 0;
  size_t nA = sw ? (size_t)P->attNw : (size_t)P->nw, nD = sw ? (size_t)P->defNw : 0;
  float* g = (float*)malloc(sizeof(float) * (size_t)P->stride);
  float* tA = (float*)malloc(sizeof(float) * (nA ? nA : 1)); float* tD = (float*)malloc(sizeof(float) * (nD ? nD : 1));
  if (g && tA && tD) {
    TailJob j = { NULL, P, &i0, 1, w, wA, wD, bat, scen, traj, 0, 0, NULL, NULL };
    side_arch(P, P->attNin, P->attNout, aA); side_arch(P, P->defNin, P->defNout, aD);
    tail_weights(&j, 0, tA, tD, &haveA, &haveD, aA, aD);
    memcpy(g, state, sizeof(float) * (size_t)P->stride);
    const BrScen* sc = sw ? &scen[bat[2]] : &scen[0]; int warm = sw ? 4 : 16, n = sw ? 16 : 64;
    if (sw) br_sw_run(g, P, sc, tA, tD, warm); else br_run(g, P, sc, tA, traj, warm);
    float k0 = g[sw ? SWH_K : S_K]; double t0 = br_now();
    if (sw) br_sw_run(g, P, sc, tA, tD, n); else br_run(g, P, sc, tA, traj, n);
    double sec = br_now() - t0, steps = g[sw ? SWH_K : S_K] - k0;
    if (steps > 0) t->cpuMs = sec * 1000 / steps;
  }
  free(g); free(tA); free(tD);
}
int br_tail_due(const BrTail* t, const BrParams* P, int live, int nActive, double ms, int chunk) {
  if (!t->on || live <= 0 || live > t->cap || nActive <= 0 || chunk <= 0 || !br_cpu_fits(P)) return 0;
  if (t->cpuMs <= 0 || t->key != tail_key(P)) return live <= t->threads;   // not measured: only when each gets its own core
  double gpu = ms / chunk * live / nActive;   // GPU time per step, as if it shrank with the thread count (favours the GPU)
  double cpu = t->cpuMs * (live > t->threads ? (double)live / t->threads : 1.0);
  return cpu < gpu;
}

// ---- helpers shared by the GPU hosts
// Dispatch length. A call's first dispatch aims at the target with the GPU time per item-step that the last call of this
// shape and layout measured; before any, it guesses from the work per item-step (network weights, balls, ball pairs),
// scaled so 65,536 flights of the default network get 64 steps. A huge batch or network thus starts at a step or two
// instead of a dispatch of seconds (GPU watchdogs: ~2 s on Windows). Never above 64: a time measured on another batch
// size is only a hint, and later dispatches grow up to 4× each. Never below one decision period (ctrl steps): a
// dispatch without a network pass is far quicker and would mislead the next estimate.
static unsigned step_key(const BrParams* P, int layout) { return tail_key(P) ^ (unsigned)layout * 2654435761u; }
static int min_chunk(const BrParams* P) { return P->ctrl > 1 ? P->ctrl : 1; }
int br_chunk_first(const Backend* b, const BrParams* P, int layout, double target) {
  double n = P->nActive > 0 ? P->nActive : 1, c; int lo = min_chunk(P);
  if (b->stepMs > 0 && b->stepKey == step_key(P, layout)) c = target / (b->stepMs * n);
  else {
    double u = 64.0 + P->nw;   // a flight: physics and a pass of its network
    if (P->mode == BR_MODE_SWARM) { int nB = P->attN + P->defN;
      u = 64.0 * nB + 32.0 * nB * nB + (P->attAI ? (P->attCmd ? 1 : P->attN) * (double)P->attNw : 0) + (P->defAI ? (P->defCmd ? 1 : P->defN) * (double)P->defNw : 0);
      if (layout > nB) u *= (double)layout / nB; }   // a work-group per battle: its idle threads cost too
    c = 64.0 * 65536 * 1123 / (n * u);
  }
  return c < lo ? lo : c > 64 ? 64 : (int)c;
}
void br_chunk_seen(Backend* b, const BrParams* P, int layout, int chunk, double ms) {
  b->stepMs = (ms > 0.5 ? ms : 0.5) / ((double)chunk * (P->nActive > 0 ? P->nActive : 1)); b->stepKey = step_key(P, layout);
}
// Later dispatches: toward the target time (long enough to amortise overhead, short enough to stay responsive), at most
// 4× longer or shorter than the last. A dispatch of under 4 steps spends much of its time starting up, so a huge batch
// keeps 4 unless those would take over 250 ms (far below GPU watchdogs); then down to one decision period.
int br_chunk_next(const BrParams* P, int chunk, double ms, double target) {
  double scale = target / (ms > 0.5 ? ms : 0.5), step = ms / chunk;
  int next = (int)(chunk * (scale > 4 ? 4 : scale < 0.25 ? 0.25 : scale)), lo = step * 4 <= 250 ? 4 : (int)(250 / step);
  if (lo < min_chunk(P)) lo = min_chunk(P);
  return next < lo ? lo : next > 100000 ? 100000 : next;
}
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
// work-group with a thread per ball (its battle kept in on-chip memory, commander layers split across the threads).
int br_sw_group_layout(const Backend* b, const BrParams* P, int* threads) {
  int nB = P->attN + P->defN, t = (nB + 31) / 32 * 32; *threads = t > 64 ? 64 : t;
  if (b->swLayout) return b->swLayout == 2;
  return nB >= 12;
}
void br_sw_init_states(float* state, const BrParams* P, const BrScen* scen, const int* bat, int n, const float* start) {
  for (int r = 0; r < n; r++) br_sw_init(state + (size_t)r * P->stride, P, &scen[bat[r * 3 + 2]], start);
}
void br_sw_collect(const float* state, const BrParams* P, int n, float* out) {
  for (int r = 0; r < n; r++) br_sw_result(state + (size_t)r * P->stride, P, out + (size_t)r * BR_SWOUT);
}
