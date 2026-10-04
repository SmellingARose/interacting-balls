// Backend interface: evaluate a batch of flights (genome × scenario) and return per flight
// BR_OUT floats: [closest approach, hit, flight time, physics steps, fraction of mid-flight below 5 m].
#ifndef BR_BACKEND_H
#define BR_BACKEND_H
#include <stddef.h>
#include "sim_core.h"

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
  int maxWg;        // largest work-group size the device allows
  double memBytes;  // device memory (0 = unknown)
} Backend;

Backend* cpu_backend_create(int threads);
#ifdef BR_HAVE_METAL
Backend* metal_backend_create(char* err, int errLen);
#endif
#ifdef BR_HAVE_OPENCL
Backend* opencl_backend_create(int deviceIndex, char* err, int errLen);
void opencl_list_devices(void);
int  opencl_devices(char names[][160], int isGpu[], int max);   // for the UI
#endif

// Shared helpers used by the GPU hosts
int  br_max_width(const BrParams* P);
void br_transpose_layers(const float* g, float* t, int nl, const int* arch);
void br_init_states(float* state, const BrParams* P, const BrScen* scen);
void br_collect(const float* state, const BrParams* P, float* out);
extern const char* BR_SRC_SIM_CORE;
extern const char* BR_SRC_METAL;
extern const char* BR_SRC_OPENCL;
#endif
