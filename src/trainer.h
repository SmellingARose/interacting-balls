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
  int altOn;                          // small reward for staying above 5 m through mid-flight
  int valEvery, valScen;
  // tag mode: chase a runner ball that launches `range`–(`range`+1.5 km) out and flies at a defended point
  int mode;                           // BR_MODE_REACH, BR_MODE_TAG or BR_MODE_SWARM
  double blast;      // tag: blast radius (m); the runner is caught within max(2 m touch, this)
  double reachMax;   // reach: goals 1.5 km to this far (m)
  double range, evade, noise, delayMs, detect, twRunner;
  // swarm: attN attackers vs defN defenders; each side flown by a network (AI) or the guidance algorithm; a network side
  // uses a commander brain (one network steers the whole side) or nearest-K (one small network per ball, sees swK nearest)
  int attN, defN, attAI, defAI, attCmd, defCmd, swK;
  double rangeMax;   // swarm: attackers launch `range`–`rangeMax` from the defended point, from any direction
  int noEarly;       // swarm: never end a battle early (for checking the early-end shortcut)
  int autoDiff; double diffAt;   // automatic difficulty: start easy, step up when validation reaches diffAt
  int imitate;    // a new network first copies the guidance algorithm (behaviour cloning)
  int normIn;     // standardise sensors with running statistics (reach and intercept)
  int restarts;   // restart the optimizer around the best network when validation stalls (BIPOP-style)
  int optKind;    // CMA family: 0 automatic (LM-MA-ES above 2,000 weights), 1 sep-CMA-ES, 2 LM-MA-ES
  int swLayout;   // GPU swarm layout (0 = automatic, 1 = thread per battle, 2 = work-group per battle), from auto-tune
} TrainCfg;

// Small growable string for JSON replies
typedef struct { char* s; size_t n, cap; } Sb;
void sb_printf(Sb* b, const char* fmt, ...);
void sb_jstr(Sb* b, const char* s);   // appends a quoted, escaped JSON string

void cfg_defaults(TrainCfg* c);
void cfg_from_json(TrainCfg* c, const char* json);   // reads the keys present and clamps every setting to its range
void cfg_to_json(const TrainCfg* c, Sb* out);        // the same keys, for the status
void setup_params(BrParams* P, const TrainCfg* c, int S);
Backend* make_backend(const char* id, const TrainCfg* c, char* err, int errLen);
const char* default_backend_id(char* buf, int len);   // what "auto" picks on this computer

void trainer_init(void);
void trainer_start(const TrainCfg* c);   // starts, or applies new settings to a running session
void trainer_pause(void);
void trainer_reset(const TrainCfg* c);   // forgets the network of c's mode (swarm: c's matchup); its files are kept as *.bak
int  trainer_busy(void);                 // training or auto-tuning
void trainer_status_json(Sb* out, int since);   // includes "cfg": the settings training uses
void trainer_star_json(Sb* out);   // the kept (best validated) network, else the latest star
int  trainer_load_json(const char* json, char* msg, int len);   // reach / intercept only; refuses a network saved in another mode
void trainer_set_mode(int mode);   // saves the current star and loads that mode's own
void hardware_json(Sb* out);
void autotune_start(const TrainCfg* c);
void autotune_status_json(Sb* out);
void fly_json(const char* req, Sb* out);
void battle_json(const char* req, Sb* out);   // swarm: one battle, every ball's path and the catches and leaks

// headless helpers (command-line testing)
int  cli_main(int argc, char** argv);
#endif
