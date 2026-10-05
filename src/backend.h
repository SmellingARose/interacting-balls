// Backend interface: evaluate a batch of flights (genome × scenario) and return per flight
// BR_OUT floats: [closest approach, hit, flight time, physics steps, fraction of mid-flight below 5 m].
// eval / evalBattles return 0, or nonzero on any failure (build, launch, memory, device lost) with a message on stderr.
#ifndef BR_BACKEND_H
#define BR_BACKEND_H
#include <stddef.h>
#include "sim_core.h"

// GPU backends: a dispatch costs about the same however few threads it runs, so once the CPU would move the items still
// running faster than the GPU just did, the CPU finishes them with the same core code. Usually that is the last few of
// a batch; a batch within the cap that the CPU flies faster than the GPU (a small swarm batch, say) goes nearly whole
// after the first dispatch. The hand-off point depends on timing and the CPU's tanh differs slightly from the GPU's, so
// with the tail on GPU results vary in the last digits from run to run.
typedef struct {
  int on;            // 1 = hand the rest to the CPU (0: the GPU flies everything, with repeatable results: to check a GPU against the CPU)
  int cap;           // never hand off more than this (about the GPU's parallel width: beyond it the GPU is busy)
  int threads;       // CPU threads for the tail
  double cpuMs;      // measured CPU time per flight-step (battles: battle-step) per thread; 0 = not measured yet
  unsigned key;      // network shape and mode that cpuMs was measured for
} BrTail;

typedef struct Backend {
  const char* name;
  // traj: tag-mode runner paths (6 floats per step, scenarios point into it); NULL / 0 in reach mode
  int  (*eval)(struct Backend* b, const float* weights, int nGenomes, const BrScen* scen, const float* traj, size_t trajFloats, BrParams* P, float* out);
  // Swarm: fly nBattles battles. Battle i pits attacker genome bat[3i] (from attW) against defender genome bat[3i+1]
  // (from defW) on scenario bat[3i+2]; a side the algorithm flies (P->attAI / P->defAI = 0) ignores its index.
  // start: per-scenario start positions (SW_START floats per ball). out: BR_SWOUT floats per battle.
  int  (*evalBattles)(struct Backend* b, const float* attW, const float* defW, const int* bat, int nBattles, const BrScen* scen,
                      const float* start, size_t startFloats, BrParams* P, float* out);
  void (*destroy)(struct Backend* b);
  void* impl;
  char info[256];
  int wg;           // GPU work-group size (0 = backend default)
  double chunkMs;   // GPU time per dispatch
  int maxWg;        // largest work-group size the device (and, once built, the kernel) allows
  double memBytes;  // device memory (0 = unknown)
  int swLayout;     // GPU swarm layout: 0 = automatic, 1 = one thread per battle, 2 = one work-group per battle (a thread per ball)
  BrTail tail;      // GPU backends: CPU hand-off of the items still running
  double stepMs;    // GPU time per item-step, measured on the last call (sizes the next call's first dispatch)
  unsigned stepKey; // network shape, mode and layout stepMs was measured for
} Backend;

Backend* cpu_backend_create(int threads);
#ifdef BR_HAVE_METAL
Backend* metal_backend_create(char* err, int errLen);
#endif
#ifdef BR_HAVE_OPENCL
Backend* opencl_backend_create(int deviceIndex, char* err, int errLen);   // deviceIndex < 0: opencl_best_gpu()
void opencl_list_devices(void);
int  opencl_devices(char names[][160], int isGpu[], int max);   // for the UI
int  opencl_best_gpu(void);   // index (as in "opencl:N") of the fastest-looking GPU, discrete before integrated; -1 if none
#endif

// Shared helpers used by the GPU hosts
int  br_max_width(const BrParams* P);
int  br_sw_max_width(const BrParams* P);   // widest swarm layer for these settings (GPU buffers are compiled to fit)
int  br_cpu_fits(const BrParams* P);       // 1 when the CPU core's fixed buffers (MAXW, SW_MAXW, ball counts) hold this shape
void br_transpose_layers(const float* g, float* t, int nl, const int* arch);
void br_init_states(float* state, const BrParams* P, const BrScen* scen);
void br_collect(const float* state, const BrParams* P, float* out);
void br_sw_init_states(float* state, const BrParams* P, const BrScen* scen, const int* bat, int n, const float* start);
void br_sw_collect(const float* state, const BrParams* P, int n, float* out);
int  br_sw_group_layout(const Backend* b, const BrParams* P, int* threads);   // 1: use the work-group layout (threads per group)
// Dispatch length in physics steps: br_chunk_first for a call's first dispatch over P->nActive items; br_chunk_seen after
// that one and after any the whole batch ran through (none ended: a clean time per item-step); br_chunk_next after each.
// layout: threads per battle's work-group, 0 = a thread per flight or battle.
int  br_chunk_first(const Backend* b, const BrParams* P, int layout, double target);
void br_chunk_seen(Backend* b, const BrParams* P, int layout, int chunk, double ms);
int  br_chunk_next(const BrParams* P, int chunk, double ms, double target);
// CPU tail: br_tail_begin before the first dispatch (state: the batch's start states, weights in the GPU layout;
// measures the CPU once per network shape), br_tail_due after each dispatch (live: still running, nActive: threads that
// dispatch ran, ms: its time), br_tail_run to finish the listed items (finished ones are skipped). wA/wD/bat: swarm only.
void br_tail_begin(BrTail* t, const float* state, const BrParams* P, const float* w, const float* wA, const float* wD, const int* bat,
                   const BrScen* scen, const float* traj);
int  br_tail_due(const BrTail* t, const BrParams* P, int live, int nActive, double ms, int chunk);
int  br_tail_run(BrTail* t, float* state, const BrParams* P, const int* idx, int n, const float* w, const float* wA, const float* wD,
                 const int* bat, const BrScen* scen, const float* traj);
extern const char* BR_SRC_SIM_CORE;
extern const char* BR_SRC_METAL;
extern const char* BR_SRC_OPENCL;
#endif
