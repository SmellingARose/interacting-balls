// Training service used by the UI server: settings, background training, auto-tune and flight playback.
#ifndef BR_TRAINER_H
#define BR_TRAINER_H
#include <stddef.h>
#include "backend.h"

typedef struct {
  char backend[32];            // "auto", "cpu", "metal" or "opencl:N"
  int threads, wg; double chunkMs;
  int layers, width, K, mem;
  int pop, scen, reuse, islands, nIsl, migrate;
  int cma, decayOn; double decay, lrMin;
  double dt, tw, speedW; int everyStep;
  int valEvery, valScen;
  // tag mode: chase a runner ball that launches `range`–(`range`+1.5 km) out and flies at a defended point
  int mode, fair;                     // mode 0 = reach goal, 1 = tag; fair = keep only tags the algorithm can reach
  double range, evade, noise, delayMs, detect, twRunner;
} TrainCfg;

void cfg_defaults(TrainCfg* c);
void cfg_from_json(TrainCfg* c, const char* json);
void cfg_to_json(const TrainCfg* c, char* out, int len);
void setup_params(BrParams* P, const TrainCfg* c, int S);
Backend* make_backend(const char* id, const TrainCfg* c, char* err, int errLen);
const char* default_backend_id(void);

// Small growable string for JSON replies
typedef struct { char* s; size_t n, cap; } Sb;
void sb_printf(Sb* b, const char* fmt, ...);
void sb_jstr(Sb* b, const char* s);   // appends a quoted, escaped JSON string

void trainer_init(void);
void trainer_start(const TrainCfg* c);   // starts, or applies new settings to a running session
void trainer_pause(void);
void trainer_reset(void);
void trainer_status_json(Sb* out, int since);
void trainer_star_json(Sb* out);
int  trainer_load_json(const char* json, char* msg, int len);   // refuses a network saved in the other mode
void trainer_set_mode(int mode);   // 0 reach, 1 tag: saves the current star and loads that mode's own
void hardware_json(Sb* out);
void autotune_start(const TrainCfg* c);
void autotune_status_json(Sb* out);
void fly_json(const char* req, Sb* out);

// headless helpers (command-line testing)
int  cli_main(int argc, char** argv);
#endif
