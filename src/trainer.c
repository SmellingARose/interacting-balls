// Training service: background training with islands (CMA-ES or GA), hardware auto-tune, star persistence and
// flight playback for the UI. All simulation runs here, natively; the UI only displays.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include "trainer.h"
#include "threads.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void br_transpose_genome(const float* g, float* t, const BrParams* P);   // backend_cpu.c
void br_transpose_layers(const float* g, float* t, int nl, const int* arch);

// ---------------- string builder
void sb_printf(Sb* b, const char* fmt, ...) {
  for (;;) {
    size_t room = b->cap - b->n; va_list ap; va_start(ap, fmt);
    int k = b->cap ? vsnprintf(b->s + b->n, room, fmt, ap) : -1; va_end(ap);
    if (k >= 0 && (size_t)k < room) { b->n += (size_t)k; return; }
    size_t need = b->n + (k > 0 ? (size_t)k : 256) + 1, cap = b->cap ? b->cap * 2 : 1024; while (cap < need) cap *= 2;
    b->s = (char*)realloc(b->s, cap); b->cap = cap;
  }
}

void sb_jstr(Sb* b, const char* s) {
  sb_printf(b, "\"");
  for (; *s; s++) { unsigned char ch = (unsigned char)*s;
    if (ch == 0x22 || ch == 0x5C) sb_printf(b, "\\%c", ch); else if (ch < 0x20) sb_printf(b, "\\u%04x", ch); else sb_printf(b, "%c", ch); }
  sb_printf(b, "\"");
}

// ---------------- tiny JSON reading (flat objects of numbers / strings / booleans)
static const char* jfind(const char* j, const char* key) {
  char pat[64]; snprintf(pat, sizeof pat, "\"%s\"", key);
  const char* p = j ? strstr(j, pat) : NULL; if (!p) return NULL;
  p += strlen(pat); while (*p == ' ' || *p == ':') p++; return p;
}
static double jnum(const char* j, const char* key, double def) {
  const char* p = jfind(j, key); if (!p) return def;
  if (!strncmp(p, "true", 4)) return 1;
  if (!strncmp(p, "false", 5)) return 0;
  char* e; double v = strtod(p, &e); return e == p ? def : v;
}
static void jstr(const char* j, const char* key, char* out, int len) {
  const char* p = jfind(j, key); if (!p || *p != '"') return; p++;
  int n = 0; while (*p && *p != '"' && n < len - 1) out[n++] = *p++; out[n] = 0;
}

// ---------------- random numbers (one generator per thread of use)
typedef struct { unsigned s[4]; int have; double spare; } Rng;
static unsigned rotl(unsigned x, int k) { return (x << k) | (x >> (32 - k)); }
static double urand(Rng* r) { unsigned* s = r->s; unsigned v = s[0] + s[3], t = s[1] << 9; s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 11); return (v >> 8) * (1.0 / 16777216.0); }
static double gauss(Rng* r) { if (r->have) { r->have = 0; return r->spare; } double m = sqrt(-2 * log(urand(r) + 1e-12)), a = 2 * M_PI * urand(r); r->spare = m * sin(a); r->have = 1; return m * cos(a); }
static void rng_seed(Rng* r, unsigned s) { for (int i = 0; i < 4; i++) { s += 0x9E3779B9u; unsigned z = s; z = (z ^ (z >> 16)) * 0x85EBCA6Bu; z = (z ^ (z >> 13)) * 0xC2B2AE35u; r->s[i] = z ^ (z >> 16); } r->have = 0; }

// ---------------- settings
static const char* mode_name(int m) { return m == BR_MODE_SWARM ? "swarm" : m == BR_MODE_TAG ? "tag" : "reach"; }
static const char* mode_label(int m) { return m == BR_MODE_SWARM ? "swarm" : m == BR_MODE_TAG ? "intercept" : "reach"; }
// Tag mode's catch distance: the balls touch (2 m) or the runner is inside the blast radius.
static double tag_r(const TrainCfg* c) { return fmax(2, c->blast); }
void cfg_defaults(TrainCfg* c) {
  memset(c, 0, sizeof *c);
  strcpy(c->backend, "auto"); c->wg = 64; c->chunkMs = 40;
  c->layers = 1; c->width = 32; c->pop = 256; c->scen = 16; c->reuse = 5; c->nIsl = 4; c->migrate = 10;
  c->cma = 1; c->restarts = 1; c->normIn = 1; c->imitate = 1; c->autoDiff = 0; c->diffAt = 0.8; c->decayOn = 1; c->decay = 0.995; c->lrMin = 1e-4;
  c->dt = 0.02; c->tw = 2.5; c->valEvery = 10; c->valScen = 64;
  c->mode = 0; c->range = 7000; c->twRunner = 2.5; c->reachMax = 7000; c->blast = 0;
  c->rangeMax = 8500; c->attN = 4; c->defN = 4; c->attAI = 0; c->defAI = 1; c->attCmd = 0; c->defCmd = 0; c->swK = 2;
}
static int clampi(int v, int a, int b) { return v < a ? a : v > b ? b : v; }
void cfg_from_json(TrainCfg* c, const char* j) {
  jstr(j, "backend", c->backend, sizeof c->backend);
  c->threads = (int)jnum(j, "threads", c->threads); c->wg = clampi((int)jnum(j, "wg", c->wg), 8, 1024); c->chunkMs = jnum(j, "chunkMs", c->chunkMs);
  c->layers = clampi((int)jnum(j, "layers", c->layers), 1, 8); c->width = clampi((int)jnum(j, "width", c->width), 4, 128);
  c->K = clampi((int)jnum(j, "K", c->K), 0, 4); c->mem = jnum(j, "mem", c->mem) != 0;
  c->pop = clampi((int)jnum(j, "pop", c->pop), 8, 262144); c->scen = clampi((int)jnum(j, "scen", c->scen), 1, 256);
  c->reuse = clampi((int)jnum(j, "reuse", c->reuse), 1, 1000); c->islands = jnum(j, "islands", c->islands) != 0;
  c->nIsl = clampi((int)jnum(j, "nIsl", c->nIsl), 2, 256); c->migrate = clampi((int)jnum(j, "migrate", c->migrate), 1, 1000);
  c->cma = jnum(j, "cma", c->cma) != 0; c->optKind = clampi((int)jnum(j, "optKind", c->optKind), 0, 2); c->restarts = jnum(j, "restarts", c->restarts) != 0; c->normIn = jnum(j, "normIn", c->normIn) != 0; c->imitate = jnum(j, "imitate", c->imitate) != 0; c->autoDiff = jnum(j, "autoDiff", c->autoDiff) != 0; c->diffAt = fmin(1, fmax(0.05, jnum(j, "diffAt", c->diffAt))); c->decayOn = jnum(j, "decayOn", c->decayOn) != 0; c->decay = jnum(j, "decay", c->decay); c->lrMin = jnum(j, "lrMin", c->lrMin);
  c->dt = jnum(j, "dt", c->dt); if (c->dt < 0.005) c->dt = 0.005; if (c->dt > 0.04) c->dt = 0.04;
  c->tw = jnum(j, "tw", c->tw); c->speedW = jnum(j, "speedW", c->speedW); c->everyStep = jnum(j, "everyStep", c->everyStep) != 0;
  c->altOn = jnum(j, "altOn", c->altOn) != 0;
  c->valEvery = clampi((int)jnum(j, "valEvery", c->valEvery), 1, 1000); c->valScen = clampi((int)jnum(j, "valScen", c->valScen), 8, 512);
  c->mode = clampi((int)jnum(j, "mode", c->mode), BR_MODE_REACH, BR_MODE_SWARM);
  c->reachMax = fmin(100000, fmax(2000, jnum(j, "reachMax", c->reachMax))); c->blast = fmin(10, fmax(0, jnum(j, "blast", c->blast)));
  c->range = fmin(100000, fmax(4000, jnum(j, "range", c->range))); c->evade = fmin(1, fmax(0, jnum(j, "evade", c->evade)));
  c->noise = fmin(200, fmax(0, jnum(j, "noise", c->noise))); c->delayMs = fmin(2000, fmax(0, jnum(j, "delayMs", c->delayMs)));
  c->detect = fmin(100000, fmax(0, jnum(j, "detect", c->detect))); c->twRunner = fmin(5, fmax(1.2, jnum(j, "twRunner", c->twRunner)));
  c->rangeMax = fmin(100000, fmax(c->range, jnum(j, "rangeMax", c->rangeMax)));
  c->attN = clampi((int)jnum(j, "attN", c->attN), 1, BR_SW_MAXA); c->defN = clampi((int)jnum(j, "defN", c->defN), 1, BR_SW_MAXD);
  c->attAI = jnum(j, "attAI", c->attAI) != 0; c->defAI = jnum(j, "defAI", c->defAI) != 0;
  c->attCmd = jnum(j, "attCmd", c->attCmd) != 0; c->defCmd = jnum(j, "defCmd", c->defCmd) != 0;
  c->swK = clampi((int)jnum(j, "swK", c->swK), 1, BR_SW_MAXK); c->swLayout = clampi((int)jnum(j, "swLayout", c->swLayout), 0, 2);
}
void cfg_to_json(const TrainCfg* c, char* o, int len) {
  snprintf(o, len, "{\"backend\":\"%s\",\"wg\":%d,\"chunkMs\":%g,\"layers\":%d,\"width\":%d,\"K\":%d,\"mem\":%d,\"pop\":%d,\"scen\":%d,\"reuse\":%d,"
    "\"islands\":%d,\"nIsl\":%d,\"migrate\":%d,\"cma\":%d,\"optKind\":%d,\"restarts\":%d,\"normIn\":%d,\"imitate\":%d,\"autoDiff\":%d,\"diffAt\":%g,\"decayOn\":%d,\"decay\":%g,\"lrMin\":%g,\"dt\":%g,\"tw\":%g,\"speedW\":%g,\"everyStep\":%d,\"altOn\":%d,"
    "\"mode\":%d,\"blast\":%g,\"reachMax\":%g,\"range\":%g,\"evade\":%g,\"noise\":%g,\"delayMs\":%g,\"detect\":%g,\"twRunner\":%g,"
    "\"rangeMax\":%g,\"attN\":%d,\"defN\":%d,\"attAI\":%d,\"defAI\":%d,\"attCmd\":%d,\"defCmd\":%d,\"swK\":%d}",
    c->backend, c->wg, c->chunkMs, c->layers, c->width, c->K, c->mem, c->pop, c->scen, c->reuse, c->islands, c->nIsl, c->migrate,
    c->cma, c->optKind, c->restarts, c->normIn, c->imitate, c->autoDiff, c->diffAt, c->decayOn, c->decay, c->lrMin, c->dt, c->tw, c->speedW, c->everyStep, c->altOn,
    c->mode, c->blast, c->reachMax, c->range, c->evade, c->noise, c->delayMs, c->detect, c->twRunner,
    c->rangeMax, c->attN, c->defN, c->attAI, c->defAI, c->attCmd, c->defCmd, c->swK);
}

// Weights of a network with `nin` inputs, `nout` outputs and P's hidden layers.
static int sw_nw(const BrParams* P, int nin, int nout) {
  int nw = 0; for (int l = 0; l < P->nl; l++) { int ni = l ? P->arch[l] : nin, no = l == P->nl - 1 ? nout : P->arch[l + 1]; nw += no * ni + no; }
  return nw;
}
// Swarm networks have no past frames or memory: every ball already sees its neighbours (or the whole battle) each step.
static void sw_params(BrParams* P, const TrainCfg* c) {
  P->K = 0; P->rec = 0; P->memN = 0;
  P->attN = c->attN; P->defN = c->defN; P->attAI = c->attAI; P->defAI = c->defAI; P->attCmd = c->attCmd; P->defCmd = c->defCmd; P->swK = c->swK;
  P->attNin = c->attCmd ? br_sw_nin_cmd(c->attN, c->defN) : br_sw_nin_nk(c->swK); P->attNout = c->attCmd ? 3 * c->attN : 3;
  P->defNin = c->defCmd ? br_sw_nin_cmd(c->defN, c->attN) : br_sw_nin_nk(c->swK); P->defNout = c->defCmd ? 3 * c->defN : 3;
  P->attNw = sw_nw(P, P->attNin, P->attNout); P->defNw = sw_nw(P, P->defNin, P->defNout);
  P->thrustAtk = (float)(c->twRunner * P->mass * BR_G); P->evade = (float)c->evade; P->leakR = 30;
  P->delay = 0;   // swarm sensors report the present (plus noise)
  P->swEarly = !c->noEarly;
  P->stride = br_sw_stride(P);
}
void setup_params(BrParams* P, const TrainCfg* c, int S) {
  memset(P, 0, sizeof *P);
  double mass = 105;
  P->mass = (float)mass; P->thrust = (float)(c->tw * mass * 9.81); P->Cd = 0.32f; P->A = 0.03f; P->CNa = 4.5f; P->SM = 0.35f; P->len = 3; P->I = 40;
  P->dt = (float)c->dt; P->maxT = 120; P->hitR = 30;
  P->mode = c->mode; P->ballD = (float)tag_r(c); P->noise = (float)c->noise; P->detect = (float)c->detect;
  P->delay = (int)lround(c->delayMs / (c->dt * 1000));
  if (c->mode != BR_MODE_REACH) P->maxT = (float)fmax(120, ceil(((c->mode == BR_MODE_SWARM ? fmax(c->rangeMax, c->range) : c->range + 1500)) / 150) + 60);   // long runs need time to arrive
  else P->maxT = (float)fmax(120, ceil(c->reachMax / 150) + 60);
  P->ctrl = c->everyStep ? 1 : 2; P->K = c->K; P->rec = c->mem; P->memN = c->mem ? c->width : 0;
  P->nin = BR_NI * (c->K + 1) + P->memN;
  P->nl = c->layers + 1; P->arch[0] = P->nin;
  for (int l = 1; l <= c->layers; l++) P->arch[l] = c->width;
  P->arch[c->layers + 1] = 3;
  int nw = 0; for (int l = 0; l < P->nl; l++) nw += P->arch[l + 1] * P->arch[l] + P->arch[l + 1];
  P->nw = nw; P->S = S;
  P->stride = (S_HIST + BR_NI * c->K + P->memN + 3) & ~3;
  if (c->mode == BR_MODE_SWARM) sw_params(P, c);
}

const char* default_backend_id(void) {
#ifdef BR_HAVE_METAL
  return "metal";
#else
  #ifdef BR_HAVE_OPENCL
  { char n[32][160]; int g[32]; int k = opencl_devices(n, g, 32); for (int i = 0; i < k; i++) if (g[i]) { static char id[16]; snprintf(id, sizeof id, "opencl:%d", i); return id; } }
  #endif
  return "cpu";
#endif
}

Backend* make_backend(const char* id, const TrainCfg* c, char* err, int errLen) {
  Backend* b = NULL;
  if (!strcmp(id, "auto")) id = default_backend_id();
  if (!strcmp(id, "cpu")) b = cpu_backend_create(c->threads);
#ifdef BR_HAVE_METAL
  else if (!strcmp(id, "metal")) b = metal_backend_create(err, errLen);
#endif
#ifdef BR_HAVE_OPENCL
  else if (!strncmp(id, "opencl", 6)) b = opencl_backend_create(id[6] == ':' ? atoi(id + 7) : -1, err, errLen);
#endif
  else snprintf(err, errLen, "compute option '%s' is not available in this build", id);
  if (b) { b->wg = c->wg; b->chunkMs = c->chunkMs; }
  return b;
}

// ---------------- scenarios and scoring (reach-goal mode: ball starts 10 m up, pointing straight up)
static void gen_scen(BrScen* s, Rng* r, double dist, double maxD) {
  double a = urand(r) * 2 * M_PI, d = dist > 0 ? dist : 1500 + urand(r) * (maxD - 1500), wa = urand(r) * 2 * M_PI, w = urand(r) * 15;
  memset(s, 0, sizeof *s);
  s->tx = (float)(cos(a) * d); s->tz = (float)(sin(a) * d);
  s->wx = (float)(cos(wa) * w); s->wz = (float)(sin(wa) * w);
  s->sy = 10.0f; s->q0 = 0.0f;
}
// Runner: the guidance algorithm, with its own engine (twRunner) and the evade weave, flies
// from `start` to the defended point `def`. Records px,py,pz,vx,vy,vz for the start and after every physics step into
// `traj` (6 floats per step, at most maxSteps steps). Returns the number of steps recorded; *atkHit = reached the goal.
#define RUNNER_CTRL 2
static int br_make_attack(const float def[3], const float wind[3], const float start[3], const BrParams* P0, float thrustAtk, double evade,
                          Rng* r, float* traj, int maxSteps, int* atkHit) {
  BrParams P = *P0; P.mode = BR_MODE_REACH; P.detect = 0; P.thrust = thrustAtk;   // the runner chases a fixed goal with its own engine
  P.stride = S_HIST;
  BrScen sc; memset(&sc, 0, sizeof sc);
  sc.tx = def[0]; sc.ty = def[1]; sc.tz = def[2]; sc.wx = wind[0]; sc.wy = wind[1]; sc.wz = wind[2];
  sc.sx = start[0]; sc.sy = start[1]; sc.sz = start[2];
  sc.q0 = start[1] > 5 ? 0.0f : (float)(urand(r) * 0.02);   // ground launch: tiny random tilt
  double w1 = 0.6 + urand(r) * 1.2, w2 = 0.6 + urand(r) * 1.2, f1 = urand(r) * 6.28, f2 = urand(r) * 6.28;
  float g[S_HIST]; br_init(g, &P, &sc);
  BrState s; br_load(&s, g);
  BrTgt tg; memset(&tg, 0, sizeof tg);
  tg.x = tg.px = tg.sx = def[0]; tg.y = tg.py = tg.sy = def[1]; tg.z = tg.pz = tg.sz = def[2];
  traj[0] = s.px; traj[1] = s.py; traj[2] = s.pz; traj[3] = traj[4] = traj[5] = 0;
  int n = 1;
  while (s.alive && n < maxSteps) {
    if (s.k++ % RUNNER_CTRL == 0) {
      float u[3]; br_evader_control(&s, &P, &sc, &tg, (float)evade, (float)w1, (float)f1, (float)w2, (float)f2, u);
      s.u0 = u[0]; s.u1 = u[1]; s.u2 = u[2];
    }
    br_step(&s, &P, &sc, &tg);
    float* o = traj + (size_t)n * 6; o[0] = s.px; o[1] = s.py; o[2] = s.pz; o[3] = s.vx; o[4] = s.vy; o[5] = s.vz; n++;
  }
  *atkHit = s.hit;
  return n;
}

// ---------------- tag scenarios: runner launches range–(range+1.5 km) from the defended
// point and flies at it; the chaser launches 0.5–3 km from it, at least 0.7·range from the runner's launch site.
static void ring_pt(const float c[3], double a, double b, Rng* r, float o[3]) {
  double t = urand(r) * 2 * M_PI, d = a + urand(r) * (b - a);
  o[0] = c[0] + (float)(cos(t) * d); o[1] = 1.6f; o[2] = c[2] + (float)(sin(t) * d);
}
// The algorithm chasing a recorded runner (tag rules, P's detect, no noise/delay). Returns the closest approach.
static float algo_tag_rollout(const BrParams* P0, const BrScen* sc, const float* traj) {
  BrParams P = *P0; P.mode = BR_MODE_TAG; P.noise = 0; P.delay = 0; P.stride = S_HIST;
  float g[S_HIST]; br_init(g, &P, sc);
  BrState s; br_load(&s, g);
  BrTgt tg; memset(&tg, 0, sizeof tg);
  while (s.alive) {
    if (!br_target(&s, &P, sc, traj, &tg)) break;
    if (s.k % P.ctrl == 0) { float u[3]; br_algo_control(&s, &P, sc, &tg, u); s.u0 = u[0]; s.u1 = u[1]; s.u2 = u[2]; }
    s.k++;
    br_step(&s, &P, sc, &tg);
  }
  return s.minD;
}
// Max physics steps one runner path can take (sizes the path buffer: scenarios × this × 6 floats).
static int tag_max_steps(const BrParams* P) { return (int)(P->maxT / P->dt) + 2; }
// Fills one tag scenario; its runner path goes to traj + off·6 (room for tag_max_steps). Returns the steps used.
static int gen_tag_scen(BrScen* out, Rng* r, const TrainCfg* c, const BrParams* P, float* traj, int off) {
  double R = c->range; int maxS = tag_max_steps(P);
  float wind[3]; { double wa = urand(r) * 2 * M_PI, w = urand(r) * 15; wind[0] = (float)(cos(wa) * w); wind[1] = 0; wind[2] = (float)(sin(wa) * w); }
  float def[3]; { double a = urand(r) * 2 * M_PI, d = urand(r) * 3000; def[0] = (float)(cos(a) * d); def[1] = 0; def[2] = (float)(sin(a) * d); }
  float thrA = (float)(c->twRunner * P->mass * BR_G);
  float* dst = traj + (size_t)off * 6;
  float rs[3], st[3]; ring_pt(def, R, R + 1500, r, rs);
  do ring_pt(def, 500, 3000, r, st); while (hypot(st[0] - rs[0], st[2] - rs[2]) < R * 0.7);
  int hit, len = br_make_attack(def, wind, rs, P, thrA, c->evade, r, dst, maxS, &hit);
  BrScen sc; memset(&sc, 0, sizeof sc);
  sc.tx = def[0]; sc.ty = def[1]; sc.tz = def[2]; sc.wx = wind[0]; sc.wy = wind[1]; sc.wz = wind[2];
  sc.sx = st[0]; sc.sy = st[1]; sc.sz = st[2]; sc.q0 = (float)(urand(r) * 0.02);
  sc.trajOff = (float)off; sc.trajLen = (float)len; sc.seed = (float)(int)(urand(r) * 16777216); sc.atkHit = (float)hit;
  *out = sc;
  return len;
}
// A whole generation's tag scenarios, built on all CPU cores (flying every runner is the
// costly part at long range). Each scenario draws from its own generator seeded from `r`, so the result does not
// depend on the thread count. Paths are recorded into fixed slots, then packed contiguously (offsets in steps).
// Returns the floats used.
typedef struct { BrScen* sc; int S, maxS; unsigned* seeds; const TrainCfg* c; const BrParams* P; float* slots; int* len; br_atomic_int next; } TagJob;
static void tag_worker(void* arg) {
  TagJob* j = (TagJob*)arg;
  for (;;) { int i = br_atomic_fetch_add(&j->next, 1); if (i >= j->S) break;
    Rng r; rng_seed(&r, j->seeds[i]);
    j->len[i] = gen_tag_scen(&j->sc[i], &r, j->c, j->P, j->slots, i * j->maxS); }
}
static size_t gen_tag_batch(BrScen* sc, int S, Rng* r, const TrainCfg* c, const BrParams* P, float* traj) {
  int maxS = tag_max_steps(P), nt = br_cpu_count(); if (nt > S) nt = S;
  unsigned* seeds = (unsigned*)malloc(sizeof(unsigned) * S); int* len = (int*)malloc(sizeof(int) * S);
  for (int i = 0; i < S; i++) seeds[i] = (unsigned)(urand(r) * 4294967295.0);
  TagJob job = { sc, S, maxS, seeds, c, P, traj, len, 0 };
  br_run_threads(nt, tag_worker, &job);
  int off = 0;   // pack: slot i starts at i·maxS ≥ off, so moving forward never overwrites an unread slot
  for (int i = 0; i < S; i++) {
    if (off != i * maxS) memmove(traj + (size_t)off * 6, traj + (size_t)i * maxS * 6, sizeof(float) * 6 * (size_t)len[i]);
    sc[i].trajOff = (float)off; off += len[i];
  }
  free(seeds); free(len);
  return (size_t)off * 6;
}

// Score of one flight o = [closest approach, hit, time, steps]: log distance down to the hit
// size (2 m tagging, 30 m reach), +5 for a hit plus a speed bonus or a small time penalty; tag mode: −2 when the
// runner reached its defended point (atkHit comes from the scenario, which the host already knows).
// Altitude reward (when on): up to +0.5, from 0 when half of mid-flight is above 5 m to full at 80% or more.
static double fitness(const float* o, double speedW, int mode, float atkHit, int altOn, double tagR) {
  double f = -log(fmax(o[0], mode == BR_MODE_TAG ? tagR : 30.0) / 30.0);
  if (o[1] > 0.5f) f += 5 + (speedW > 0 ? speedW * fmax(0, 1 - o[2] / 60.0) : -0.01 * o[2]);
  else if (mode == BR_MODE_TAG && atkHit > 0.5f) f -= 2;
  if (altOn) f += 0.5 * fmin(1, fmax(0, (1 - o[4] - 0.5) / 0.3));
  return f;
}

// Swarm battle setup: defended point 0–3 km from the origin, wind; attackers on the ground at range–(range+1.5 km)
// from it, from any direction; defenders 0.5–3 km from it. Writes
// SW_START floats per ball (x, y, z, start tilt, weave w1, f1, w2, f2) into `st`; `off` is where they start.
static void sw_gen_scen(BrScen* sc, Rng* r, const TrainCfg* c, float* st, float off) {
  memset(sc, 0, sizeof *sc);
  double a = urand(r) * 2 * M_PI, d = urand(r) * 3000, wa = urand(r) * 2 * M_PI, w = urand(r) * 15;
  sc->tx = (float)(cos(a) * d); sc->ty = 0; sc->tz = (float)(sin(a) * d);
  sc->wx = (float)(cos(wa) * w); sc->wz = (float)(sin(wa) * w);
  sc->trajOff = off; sc->seed = (float)(int)(urand(r) * 16777216);
  for (int b = 0; b < c->attN + c->defN; b++) {
    float* p = st + b * SW_START; double t, rr;
    if (b < c->attN) { t = urand(r) * 2 * M_PI; rr = c->range + urand(r) * fmax(0, c->rangeMax - c->range); }
    else { t = urand(r) * 2 * M_PI; rr = 500 + urand(r) * 2500; }
    p[0] = sc->tx + (float)(cos(t) * rr); p[1] = 1.6f; p[2] = sc->tz + (float)(sin(t) * rr); p[3] = (float)(urand(r) * 0.02);
    p[4] = (float)(0.6 + urand(r) * 1.2); p[5] = (float)(urand(r) * 6.28); p[6] = (float)(0.6 + urand(r) * 1.2); p[7] = (float)(urand(r) * 6.28);
  }
}

// A set of scenarios: reach goals, or tag setups with their runner paths packed into one buffer.
typedef struct { BrScen* sc; int n, cap; float* traj; size_t trajFloats, trajCap; } ScenSet;
static void scen_make(ScenSet* s, int S, Rng* r, const TrainCfg* c, const BrParams* P) {
  if (S > s->cap) { s->sc = (BrScen*)realloc(s->sc, sizeof(BrScen) * S); s->cap = S; }
  s->n = S; s->trajFloats = 0;
  if (c->mode == BR_MODE_REACH) { for (int i = 0; i < S; i++) gen_scen(&s->sc[i], r, 0, c->reachMax); return; }
  if (c->mode == BR_MODE_SWARM) {
    size_t per = (size_t)(c->attN + c->defN) * SW_START;
    if ((size_t)S * per > s->trajCap) { s->traj = (float*)realloc(s->traj, sizeof(float) * S * per); s->trajCap = S * per; }
    for (int i = 0; i < S; i++) sw_gen_scen(&s->sc[i], r, c, s->traj + i * per, (float)(i * per));
    s->trajFloats = S * per; return;
  }
  size_t need = (size_t)S * tag_max_steps(P) * 6;
  if (need > s->trajCap) { s->traj = (float*)realloc(s->traj, sizeof(float) * need); s->trajCap = need; }
  s->trajFloats = gen_tag_batch(s->sc, S, r, c, P, s->traj);
}
static void scen_free(ScenSet* s) { free(s->sc); free(s->traj); memset(s, 0, sizeof *s); }
// Settings that change what a scenario is (when they change, scenarios are rebuilt).
static int scen_same(const TrainCfg* a, const TrainCfg* b) {
  return a->mode == b->mode && (a->mode != BR_MODE_REACH || a->reachMax == b->reachMax) && (a->mode == BR_MODE_REACH || (a->range == b->range && a->evade == b->evade && a->blast == b->blast &&
         a->twRunner == b->twRunner && a->detect == b->detect && a->dt == b->dt && a->everyStep == b->everyStep &&
         (a->mode != BR_MODE_SWARM || (a->attN == b->attN && a->defN == b->defN && a->rangeMax == b->rangeMax))));
}
static int eval_set(Backend* b, const float* g, int n, const ScenSet* s, BrParams* P, float* out) {
  return b->eval(b, g, n, s->sc, s->trajFloats ? s->traj : NULL, s->trajFloats, P, out);
}

static void random_genome(float* g, const BrParams* P, Rng* r) {
  for (int i = 0; i < P->nw; i++) g[i] = (float)((urand(r) * 2 - 1) * 0.5);
  if (P->rec) { int ni = P->arch[0], h1 = P->arch[1], m0 = BR_NI * (P->K + 1);   // memory weights start small
    for (int j = 0; j < h1; j++) for (int i = m0; i < ni; i++) g[j * ni + i] *= 0.2f; }
}

// ---------------- optimizers (GA and sep-CMA-ES)
typedef struct { int n, lam; float* pop; double sigma; } GA;
// sep-CMA-ES (diagonal covariance), or for large networks LM-MA-ES (Loshchilov, Glasmachers & Beyer 2017): a few
// memory vectors learn the main directions of the search at O(n) cost, where a covariance model would be too slow.
// Both draw mirrored pairs (z, −z), which halves the noise of each update at no extra cost.
typedef struct { int n, lam, mu; double* w; double mueff, cs, ds, cc, c1, cmu, chiN, sigma; double *m, *C, *ps, *pc; float *Z, *Y, *X; int g;
  int lm, nm; double *Mv, *cdv, *ccv; } CMA;
#define LM_AUTO_N 2000   // networks with more weights use LM-MA-ES (optKind 0 = automatic)
static int g_optKind = 0;   // 0 automatic, 1 sep-CMA-ES, 2 LM-MA-ES (set from the settings before an optimizer is built)
typedef struct { int cma; GA ga; CMA c; float* bestG; double bestF; } Island;

static const double* g_sortFit;
static int cmp_idx(const void* a, const void* b) { double fa = g_sortFit[*(const int*)a], fb = g_sortFit[*(const int*)b]; return (fa < fb) - (fa > fb); }
static int* rank(const double* f, int n) { int* o = (int*)malloc(sizeof(int) * n); for (int i = 0; i < n; i++) o[i] = i; g_sortFit = f; qsort(o, n, sizeof(int), cmp_idx); return o; }

static void ga_init(GA* ga, int n, int lam, const float* seed, double s0, Rng* r) {
  ga->n = n; ga->lam = lam; ga->sigma = s0; ga->pop = (float*)malloc(sizeof(float) * (size_t)n * lam);
  for (int i = 0; i < lam; i++) for (int j = 0; j < n; j++) ga->pop[(size_t)i * n + j] = seed[j] + (i ? (float)(gauss(r) * s0) : 0);
}
static void ga_tell(GA* ga, const double* f, Rng* r) {
  int n = ga->n, lam = ga->lam, *ord = rank(f, lam);
  int elite = lam / 8 > 1 ? lam / 8 : 1, pool = lam / 3 > 2 ? lam / 3 : 2;
  float* next = (float*)malloc(sizeof(float) * (size_t)n * lam);
  for (int i = 0; i < elite; i++) memcpy(next + (size_t)i * n, ga->pop + (size_t)ord[i] * n, sizeof(float) * n);
  for (int i = elite; i < lam; i++) {
    const float* a = ga->pop + (size_t)ord[(int)(urand(r) * urand(r) * pool)] * n, *b = ga->pop + (size_t)ord[(int)(urand(r) * urand(r) * pool)] * n;
    float* c = next + (size_t)i * n; double sg = ga->sigma * (0.3 + urand(r));
    for (int j = 0; j < n; j++) c[j] = (urand(r) < 0.5 ? a[j] : b[j]) + (urand(r) < 0.3 ? (float)(gauss(r) * sg) : 0);
  }
  free(ga->pop); ga->pop = next; free(ord);
}
static void ga_free(GA* ga) { free(ga->pop); }

static void cma_init(CMA* c, int n, int lam, const float* seed, double sigma0) {
  memset(c, 0, sizeof *c);
  c->n = n; c->lam = lam; c->mu = lam / 2 > 1 ? lam / 2 : 1;
  c->w = (double*)malloc(sizeof(double) * c->mu); double sw = 0, s2 = 0;
  for (int i = 0; i < c->mu; i++) { c->w[i] = log(c->mu + 0.5) - log(i + 1.0); sw += c->w[i]; }
  for (int i = 0; i < c->mu; i++) { c->w[i] /= sw; s2 += c->w[i] * c->w[i]; }
  double me = c->mueff = 1 / s2;
  c->cs = (me + 2) / (n + me + 5); c->ds = 1 + 2 * fmax(0, sqrt((me - 1) / (n + 1)) - 1) + c->cs;
  c->cc = (4 + me / n) / (n + 4 + 2 * me / n);
  double c1 = 2 / ((n + 1.3) * (n + 1.3) + me), cmu = fmin(1 - c1, 2 * (me - 2 + 1 / me) / ((n + 2.0) * (n + 2.0) + me)), sc = (n + 2) / 3.0;
  c1 *= sc; cmu *= sc; if (c1 + cmu > 0.9) { double k = 0.9 / (c1 + cmu); c1 *= k; cmu *= k; } c->c1 = c1; c->cmu = cmu;
  c->chiN = sqrt((double)n) * (1 - 1.0 / (4 * n) + 1.0 / (21.0 * n * n));
  c->m = (double*)malloc(sizeof(double) * n); c->C = (double*)malloc(sizeof(double) * n);
  c->ps = (double*)calloc(n, sizeof(double)); c->pc = (double*)calloc(n, sizeof(double));
  for (int i = 0; i < n; i++) { c->m[i] = seed[i]; c->C[i] = 1; }
  c->sigma = sigma0;
  c->Z = (float*)malloc(sizeof(float) * (size_t)n * lam); c->Y = (float*)malloc(sizeof(float) * (size_t)n * lam); c->X = (float*)malloc(sizeof(float) * (size_t)n * lam);
  c->lm = g_optKind == 2 || (g_optKind == 0 && n > LM_AUTO_N);
  if (c->lm) {   // LM-MA-ES settings from the paper
    c->nm = 4 + (int)floor(3 * log((double)n)); c->Mv = (double*)calloc((size_t)c->nm * n, sizeof(double));
    c->cdv = (double*)malloc(sizeof(double) * c->nm); c->ccv = (double*)malloc(sizeof(double) * c->nm);
    for (int j = 0; j < c->nm; j++) { c->cdv[j] = 1 / (pow(1.5, j) * n); c->ccv[j] = fmin(1, lam / (pow(4, j) * n)); }
    c->cs = fmin(0.5, 2.0 * lam / n);
  }
}
static void cma_ask(CMA* c, Rng* r) {
  int n = c->n;
  for (int k = 0; k < c->lam; k++) {
    float* Z = c->Z + (size_t)k * n; float* Y = c->Y + (size_t)k * n; float* X = c->X + (size_t)k * n;
    if (k & 1) { const float* Zp = Z - n; for (int i = 0; i < n; i++) Z[i] = -Zp[i]; }   // mirrored: the opposite of the previous one
    else for (int i = 0; i < n; i++) Z[i] = (float)gauss(r);
    if (c->lm) {   // d = z, then pulled toward each memory direction in turn
      double* d = (double*)malloc(sizeof(double) * n); for (int i = 0; i < n; i++) d[i] = Z[i];
      for (int j = 0; j < c->nm; j++) { const double* M = c->Mv + (size_t)j * n; double dot = 0; for (int i = 0; i < n; i++) dot += M[i] * d[i];
        double a = 1 - c->cdv[j], b = c->cdv[j] * dot; for (int i = 0; i < n; i++) d[i] = a * d[i] + b * M[i]; }
      for (int i = 0; i < n; i++) { Y[i] = (float)d[i]; X[i] = (float)(c->m[i] + c->sigma * d[i]); }
      free(d);
    } else for (int i = 0; i < n; i++) { double y = sqrt(c->C[i]) * Z[i]; Y[i] = (float)y; X[i] = (float)(c->m[i] + c->sigma * y); }
  }
}
static void cma_tell(CMA* c, const double* f) {
  int n = c->n, mu = c->mu, *ord = rank(f, c->lam);
  double* ym = (double*)calloc(n, sizeof(double)); double* zm = (double*)calloc(n, sizeof(double));
  for (int k = 0; k < mu; k++) { const float* y = c->Y + (size_t)ord[k] * n; const float* z = c->Z + (size_t)ord[k] * n; double wk = c->w[k];
    for (int i = 0; i < n; i++) { ym[i] += wk * y[i]; zm[i] += wk * z[i]; } }
  if (c->lm) {   // LM-MA-ES update: mean, step-size path, memory directions, step size
    double cs = c->cs, a = sqrt(cs * (2 - cs) * c->mueff), psn2 = 0;
    for (int i = 0; i < n; i++) { c->m[i] += c->sigma * ym[i]; c->ps[i] = (1 - cs) * c->ps[i] + a * zm[i]; psn2 += c->ps[i] * c->ps[i]; }
    for (int j = 0; j < c->nm; j++) { double* M = c->Mv + (size_t)j * n; double cj = c->ccv[j], b = sqrt(c->mueff * cj * (2 - cj));
      for (int i = 0; i < n; i++) M[i] = (1 - cj) * M[i] + b * zm[i]; }
    c->g++; c->sigma = fmin(2, c->sigma * exp(cs / 2 * (psn2 / n - 1)));
    free(ym); free(zm); free(ord); return;
  }
  double cs = c->cs, cc = c->cc, a = sqrt(cs * (2 - cs) * c->mueff), b = sqrt(cc * (2 - cc) * c->mueff), psn = 0;
  for (int i = 0; i < n; i++) { c->m[i] += c->sigma * ym[i]; c->ps[i] = (1 - cs) * c->ps[i] + a * zm[i]; psn += c->ps[i] * c->ps[i]; }
  psn = sqrt(psn); c->g++;
  int hs = psn / sqrt(1 - pow(1 - cs, 2.0 * c->g)) / c->chiN < 1.4 + 2.0 / (n + 1);
  for (int i = 0; i < n; i++) {
    c->pc[i] = (1 - cc) * c->pc[i] + hs * b * ym[i];
    double rk = 0; for (int k = 0; k < mu; k++) { double y = c->Y[(size_t)ord[k] * n + i]; rk += c->w[k] * y * y; }
    c->C[i] = fmax(1e-8, (1 - c->c1 - c->cmu) * c->C[i] + c->c1 * (c->pc[i] * c->pc[i] + (1 - hs) * cc * (2 - cc) * c->C[i]) + c->cmu * rk);
  }
  c->sigma = fmin(2, c->sigma * exp((cs / c->ds) * (psn / c->chiN - 1)));
  free(ym); free(zm); free(ord);
}
static void cma_free(CMA* c) { free(c->w); free(c->m); free(c->C); free(c->ps); free(c->pc); free(c->Z); free(c->Y); free(c->X); free(c->Mv); free(c->cdv); free(c->ccv); }

static const float* isl_genomes(Island* I) { return I->cma ? I->c.X : I->ga.pop; }
static int isl_size(Island* I) { return I->cma ? I->c.lam : I->ga.lam; }
static double* isl_sigma(Island* I) { return I->cma ? &I->c.sigma : &I->ga.sigma; }

// ---------------- the training service
#define HIST_MAX 2000
#define ISL_SHOW 16
typedef struct { int gen; float best, mean; float isl[ISL_SHOW]; } Hist;

static struct {
  br_mutex mx; br_thread th; volatile int running, stopReq, threadLive;
  TrainCfg cfg, pending; volatile int hasPending;
  Backend* be; char beId[32]; int beWg; double beChunk; int beThreads;
  BrParams P; Island* isl; int nIsl; int structKey[8];
  Rng rng; ScenSet scen; int scenAge; TrainCfg scenCfg;
  ScenSet val; TrainCfg valCfg; int valReady;   // fixed validation set (same seed every time, rebuilt when settings change)
  int starMode;   // mode the current star was trained in (BR_MODE_*); each mode keeps its own save file
  float* star; int starNw; int starArch[BR_MAXL]; int starNl; int starK, starMem; double starDt, starTw; int starEvery; int starVersion; double starFit;
  int gen; double best, mean, sigma, genTime, flightsPerS, stepsPerS, starHits, starMiss, valHit; int valGen;
  Hist* hist; int histN;
  char msg[256]; char beInfo[256];
  // restarts (BIPOP-style): population multiplier and starting step size of the current run, stalled validations
  int restarts, stall; double popMul, restartSigma, bestVal, fitSum, bestFitAvg; int fitN;
  // sensor normalisation: the optimizer works on networks that see standardised sensors; mean / spread per sensor
  int normOn; double normMu[BR_NI], normSd[BR_NI]; float* eff; size_t effCap;
  // automatic difficulty: level 0 (easy) .. 10 (the settings); the settings training actually uses this generation
  int level, levelHits; TrainCfg run;
  // auto-tune
  br_thread tuneTh; volatile int tuning, tuneLive; TrainCfg tuneCfg; Sb tuneLog; Sb tuneResult;
} T;

static void set_msg(const char* fmt, ...) { va_list ap; va_start(ap, fmt); br_lock(&T.mx); vsnprintf(T.msg, sizeof T.msg, fmt, ap); br_unlock(&T.mx); va_end(ap); }

// One saved star per mode: BallArena-star-reach.json and BallArena-star-tag.json. Older single-mode saves
// (mode -1: BallArena-star.json, -2: BallisticRange-champion.json) are read as the reach star if no reach file exists.
static const char* save_path_mode(int mode) {
  static char p[1024]; const char* h = getenv("HOME");
  const char* name = mode == -2 ? "BallisticRange-champion.json" : mode < 0 ? "BallArena-star.json" : mode == BR_MODE_SWARM ? "BallArena-star-swarm.json" : mode == BR_MODE_TAG ? "BallArena-star-tag.json" : "BallArena-star-reach.json";
#ifdef _WIN32
  if (!h) h = getenv("USERPROFILE");
  snprintf(p, sizeof p, "%s\\%s", h ? h : ".", name);
#else
  snprintf(p, sizeof p, "%s/%s", h ? h : ".", name);
#endif
  return p;
}
static const char* save_path(void) { return save_path_mode(T.starMode); }

static void star_json_locked(Sb* o) {
  sb_printf(o, "{\"format\":\"ball-arena-network\",\"mode\":\"%s\",\"arch\":[", mode_name(T.starMode));
  for (int l = 0; l <= T.starNl; l++) sb_printf(o, "%s%d", l ? "," : "", T.starArch[l]);
  sb_printf(o, "],\"K\":%d,\"memory\":%s,\"dt\":%g,\"ctrlEvery\":%d,\"thrustToWeight\":%g,\"generation\":%d,\"fitness\":%.4f,\"validationHitRate\":%.4f,\"version\":%d,\"weights\":[",
    T.starK, T.starMem ? "true" : "false", T.starDt, T.starEvery ? 1 : 2, T.starTw, T.gen, T.starFit, T.valHit, T.starVersion);
  for (int i = 0; i < T.starNw; i++) sb_printf(o, "%s%.7g", i ? "," : "", T.star[i]);
  sb_printf(o, "]}");
}
static void autosave(void) {
  Sb o = {0}; br_lock(&T.mx); if (T.star) star_json_locked(&o); br_unlock(&T.mx);
  if (o.n) { FILE* f = fopen(save_path(), "w"); if (f) { fwrite(o.s, 1, o.n, f); fclose(f); } }
  free(o.s);
}

// Load a network (JSON with "arch", "K", "memory", "weights"). Called with the lock NOT held.
static int load_json_mode(const char* json, char* msg, int len, int wantMode) {
  const char* a = strstr(json, "\"arch\""); const char* w = strstr(json, "\"weights\"");
  if (!a || !w) { snprintf(msg, len, "not a Ball Arena network file"); return -1; }
  int arch[BR_MAXL], nl = -1; const char* p = strchr(a, '['); if (!p) return -1; p++;
  while (*p && *p != ']' && nl < BR_MAXL - 1) { arch[++nl] = (int)strtol(p, (char**)&p, 10); while (*p == ',' || *p == ' ') p++; }
  if (nl < 1) { snprintf(msg, len, "bad arch"); return -1; }
  // mode: "tag", or "reach" ("mission" and a missing field are older reach saves). A star from the other mode is refused.
  int fmode = BR_MODE_REACH; { const char* m = strstr(json, "\"mode\""); const char* t = m ? strstr(m, "\"tag\"") : NULL; const char* w = m ? strstr(m, "\"swarm\"") : NULL;
    if (t && t - m < 16) fmode = BR_MODE_TAG; else if (w && w - m < 16) fmode = BR_MODE_SWARM; }
  if (wantMode >= 0 && fmode != wantMode) { snprintf(msg, len, "this network was trained in %s mode; switch to that mode to load it", mode_label(fmode)); return -1; }
  long nwl = 0; for (int l = 0; l <= nl; l++) if (arch[l] < 1 || arch[l] > 4096) { snprintf(msg, len, "bad layer size"); return -1; }
  for (int l = 0; l < nl; l++) nwl += (long)arch[l + 1] * arch[l] + arch[l + 1];
  if (nwl > (4L << 20)) { snprintf(msg, len, "network too large"); return -1; }
  int nw = (int)nwl;
  float* g = (float*)malloc(sizeof(float) * nw); p = strchr(w, '['); if (!p) { free(g); return -1; } p++;
  int n = 0; while (n < nw && *p && *p != ']') { g[n++] = strtof(p, (char**)&p); while (*p == ',' || *p == ' ') p++; }
  if (n != nw) { free(g); snprintf(msg, len, "expected %d weights, found %d", nw, n); return -1; }
  br_lock(&T.mx);
  free(T.star); T.star = g; T.starNw = nw; T.starMode = fmode; T.starNl = nl; memcpy(T.starArch, arch, sizeof(int) * (nl + 1));
  T.starK = (int)jnum(json, "K", 0); T.starMem = jnum(json, "memory", 0) != 0; T.starVersion++; T.starFit = jnum(json, "fitness", 0);
  // physics it was trained with (files without them: the current settings)
  T.starDt = jnum(json, "dt", T.cfg.dt); T.starTw = jnum(json, "thrustToWeight", T.cfg.tw); T.starEvery = jnum(json, "ctrlEvery", T.cfg.everyStep ? 1 : 2) == 1;
  T.valHit = jnum(json, "validationHitRate", 0);
  br_unlock(&T.mx);
  snprintf(msg, len, "loaded network %d", arch[0]); for (int l = 1; l <= nl; l++) { char t[16]; snprintf(t, sizeof t, "-%d", arch[l]); strncat(msg, t, len - strlen(msg) - 1); }
  return 0;
}

static void free_islands(void);
// Loading a network from the interface or --load: it must belong to the mode currently selected.
int trainer_load_json(const char* json, char* msg, int len) { return load_json_mode(json, msg, len, T.cfg.mode); }

// Make `mode`'s saved star the current one (none if that mode has never been trained). Lock NOT held.
static void load_mode_star(int mode) {
  br_lock(&T.mx); free(T.star); T.star = NULL; T.starNw = 0; T.starMode = mode; T.starVersion++; T.gen = 0; T.histN = 0;
  T.best = T.mean = T.starHits = T.starMiss = T.valHit = 0; T.valGen = 0;
  T.restarts = 0; T.stall = 0; T.bestVal = 0; T.popMul = 1; T.restartSigma = 0; T.bestFitAvg = -1e30; T.fitSum = 0; T.fitN = 0; T.level = 10; T.levelHits = 0;
  br_unlock(&T.mx);
  if (mode == BR_MODE_SWARM) { set_msg("swarm: each side's star is kept per matchup and loaded when training starts"); return; }
  const char* p = save_path_mode(mode); FILE* f = fopen(p, "rb");
  for (int old = -1; !f && mode == BR_MODE_REACH && old >= -2; old--) { p = save_path_mode(old); f = fopen(p, "rb"); }
  if (f) { fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET); char* b = (char*)malloc(len + 1); len = (long)fread(b, 1, len, f); b[len] = 0; fclose(f);
    char m[200]; if (!load_json_mode(b, m, sizeof m, mode)) set_msg("restored your last %s star (%s)", mode_label(mode), p); free(b); }
  else set_msg("no saved %s star yet", mode_label(mode));
}
// Select a mode: saves the current star, then makes that mode's saved star current. Called with training stopped
// (the interface's mode switch) or from the training thread between generations.
void trainer_set_mode(int mode) {
  mode = clampi(mode, BR_MODE_REACH, BR_MODE_SWARM);
  if (mode == T.starMode && T.cfg.mode == mode) return;
  autosave(); T.cfg.mode = mode; load_mode_star(mode);
  free_islands();
}

void trainer_init(void) {
  br_mutex_init(&T.mx); rng_seed(&T.rng, (unsigned)(br_now() * 1000));
  cfg_defaults(&T.cfg); T.hist = (Hist*)calloc(HIST_MAX, sizeof(Hist)); T.bestFitAvg = -1e30; T.popMul = 1;
  load_mode_star(T.cfg.mode);
}

static int star_matches(const BrParams* P) {
  if (!T.star || T.starMode != P->mode || T.starNl != P->nl) return 0;
  for (int l = 0; l <= P->nl; l++) if (T.starArch[l] != P->arch[l]) return 0;
  return T.starK == P->K && T.starMem == P->rec;
}

static void free_islands(void) {
  for (int i = 0; i < T.nIsl; i++) { if (T.isl[i].cma) cma_free(&T.isl[i].c); else ga_free(&T.isl[i].ga); free(T.isl[i].bestG); }
  free(T.isl); T.isl = NULL; T.nIsl = 0;
}
// ---------------- sensor normalisation (reach and intercept)
// The optimizer searches over networks whose first layer sees standardised sensors, (x − μ) / σ per sensor (past frames
// use their sensor's statistics; memory inputs are left alone). Before a network is flown, the standardisation is folded
// into its first layer: W' = W / σ, b' = b − W μ / σ, so the simulation and every backend are unchanged, and the star
// (and its saved file) is an ordinary network. Unfolding is the inverse.
static void norm_fold(const float* g, float* e, const BrParams* P, int unfold) {
  int ni = P->arch[0], no = P->arch[1], nn = BR_NI * (P->K + 1);
  memcpy(e, g, sizeof(float) * P->nw);
  const float* W = g; const float* B = g + no * ni; float* We = e; float* Be = e + no * ni;
  for (int j = 0; j < no; j++) {
    double b = B[j];
    for (int i = 0; i < nn; i++) { int f = i % BR_NI; double mu = T.normMu[f], sd = T.normSd[f];
      if (!unfold) { double w = W[j * ni + i] / sd; We[j * ni + i] = (float)w; b -= w * mu; }
      else { double w = W[j * ni + i]; We[j * ni + i] = (float)(w * sd); b += w * mu; } }
    Be[j] = (float)b;
  }
}
// Sensor statistics from flying network `w` (an ordinary, folded network) on this generation's scenarios on the CPU.
static int norm_measure(const float* w, double* mu, double* sd) {
  BrParams P = T.P; P.S = 1; P.nRoll = 1; int stride = P.stride;
  float* st = (float*)malloc(sizeof(float) * stride); float* wt = (float*)malloc(sizeof(float) * P.nw); br_transpose_genome(w, wt, &P);
  double s1[BR_NI] = {0}, s2[BR_NI] = {0}; long n = 0;
  for (int si = 0; si < T.scen.n && si < 16; si++) {
    const BrScen* sc = &T.scen.sc[si]; br_init(st, &P, sc);
    BrState s; br_load(&s, st); BrTgt tg; memset(&tg, 0, sizeof tg);
    float* hist = st + S_HIST; float* mem = st + S_HIST + BR_NI * P.K;
    for (int k = 0; k < 200000 && s.alive; k++) {
      if (!br_target(&s, &P, sc, T.scen.traj, &tg)) break;
      if (s.k % P.ctrl == 0) { float x[MAXW]; br_inputs(&s, &tg, x); for (int f = 0; f < BR_NI; f++) { s1[f] += x[f]; s2[f] += (double)x[f] * x[f]; } n++;
        br_control(&s, &P, &tg, wt, hist, mem); }
      s.k++; br_step(&s, &P, sc, &tg);
    }
  }
  free(st); free(wt);
  if (n < 50) return -1;
  for (int f = 0; f < BR_NI; f++) { mu[f] = s1[f] / n; double v = s2[f] / n - mu[f] * mu[f]; sd[f] = sqrt(v > 1e-6 ? v : 1e-6); if (sd[f] < 0.05) sd[f] = 0.05; }
  return 0;
}
// New statistics (blended half-way with the old ones), keeping every network the optimizer holds behaviourally the same:
// each island's centre (and GA population) is re-expressed for the new statistics.
static void norm_update(const float* star) {
  double mu[BR_NI], sd[BR_NI]; if (norm_measure(star, mu, sd)) return;
  int nw = T.P.nw; float* e = (float*)malloc(sizeof(float) * nw); float* g = (float*)malloc(sizeof(float) * nw);
  double om[BR_NI], os[BR_NI]; memcpy(om, T.normMu, sizeof om); memcpy(os, T.normSd, sizeof os);
  double nm[BR_NI], ns[BR_NI]; for (int f = 0; f < BR_NI; f++) { nm[f] = T.normOn ? 0.5 * (om[f] + mu[f]) : mu[f]; ns[f] = T.normOn ? 0.5 * (os[f] + sd[f]) : sd[f]; }
  for (int i = 0; i < T.nIsl; i++) { Island* I = &T.isl[i];
    int cnt = I->cma ? 1 : I->ga.lam;
    for (int q = 0; q < cnt; q++) {
      if (I->cma) for (int j = 0; j < nw; j++) g[j] = (float)I->c.m[j]; else memcpy(g, I->ga.pop + (size_t)q * nw, sizeof(float) * nw);
      memcpy(T.normMu, om, sizeof om); memcpy(T.normSd, os, sizeof os); norm_fold(g, e, &T.P, 0);   // the network it is now
      memcpy(T.normMu, nm, sizeof nm); memcpy(T.normSd, ns, sizeof ns); norm_fold(e, g, &T.P, 1);   // the same network, new statistics
      if (I->cma) { for (int j = 0; j < nw; j++) I->c.m[j] = g[j];
        int ni = T.P.arch[0], no = T.P.arch[1], nn = BR_NI * (T.P.K + 1);   // search widths follow the rescaled weights
        if (!I->c.lm) for (int jj = 0; jj < no; jj++) for (int ii = 0; ii < nn; ii++) { double k = ns[ii % BR_NI] / os[ii % BR_NI]; I->c.C[jj * ni + ii] *= k * k; } }
      else memcpy(I->ga.pop + (size_t)q * nw, g, sizeof(float) * nw);
    }
  }
  memcpy(T.normMu, nm, sizeof nm); memcpy(T.normSd, ns, sizeof ns); T.normOn = 1;
  free(e); free(g);
}

// ---------------- automatic difficulty (curriculum)
// Level 10 is exactly the settings. Lower levels interpolate toward an easy version: short reach goals; close, slower,
// non-weaving attackers or runners, no sensor noise or delay, launch at once, a bigger catch radius; one attacker in a
// nearest-K swarm. The network's shape never changes with the level.
static TrainCfg curriculum(const TrainCfg* c, int level) {
  TrainCfg e = *c; if (!c->autoDiff || level >= 10) return e;
  double t = level / 10.0;
#define LERP(a, b) ((a) + ((b) - (a)) * t)
  if (c->mode == BR_MODE_REACH) e.reachMax = LERP(fmin(3000, c->reachMax), c->reachMax);
  else {
    e.range = LERP(fmin(4000, c->range), c->range);
    if (c->mode == BR_MODE_SWARM) e.rangeMax = fmax(e.range, LERP(fmin(6000, c->rangeMax), c->rangeMax));
    e.evade = c->evade * t; e.noise = c->noise * t; e.delayMs = c->delayMs * t;
    e.detect = c->detect <= 0 || t <= 0 ? 0 : c->detect + (100000 - c->detect) * (1 - t);   // radar sees from further away
    e.blast = LERP(fmax(10, c->blast), c->blast); e.twRunner = LERP(fmin(1.5, c->twRunner), c->twRunner);
    if (c->mode == BR_MODE_SWARM && !(c->attAI && c->attCmd) && !(c->defAI && c->defCmd)) e.attN = 1 + (int)lround((c->attN - 1) * t);
  }
#undef LERP
  return e;
}
static double validate(const float* g);
// ---------------- head start by imitation (behaviour cloning)
// A network that would start from random weights first learns to copy the guidance algorithm: the algorithm flies on the
// CPU while every decision is recorded as (what the network would sense, what the algorithm did), then the network is
// fitted to those pairs by mini-batch gradient descent (Adam, mean squared error) for a few seconds. Evolution then
// continues from a network that already flies, instead of one that crashes for hundreds of generations.
#define BC_MAXN 60000      // recorded decisions
#define BC_SECONDS 3.0     // fitting budget per rnd
#define BC_ROUNDS 3        // DAgger: rnd 0 the algorithm flies; later rounds the network flies and the algorithm labels
typedef struct { float* X; float* Y; int n, nin, nout, cap; } BcData;
static void bc_add(BcData* d, const float* x, const float* u3, int k) {   // k balls' controls (3 each) as network targets
  if (d->n >= d->cap) return;
  memcpy(d->X + (size_t)d->n * d->nin, x, sizeof(float) * d->nin);
  float* y = d->Y + (size_t)d->n * d->nout;
  for (int b = 0; b < k; b++) { y[b * 3] = (float)fmin(0.95, fmax(-0.95, 2 * u3[b * 3] - 1)); y[b * 3 + 1] = (float)fmin(0.95, fmax(-0.95, u3[b * 3 + 1])); y[b * 3 + 2] = (float)fmin(0.95, fmax(-0.95, u3[b * 3 + 2])); }
  d->n++;
}
// Fits a tanh network (genome layout as stored: per layer W out×in row-major, then biases) to the data.
static void bc_fit(float* g, int nl, const int* arch, const BcData* d, Rng* r, int init) {
  int nw = 0, off[BR_MAXL + 1], maxw = 0; for (int l = 0; l < nl; l++) { off[l] = nw; nw += arch[l + 1] * arch[l] + arch[l + 1]; }
  for (int l = 0; l <= nl; l++) if (arch[l] > maxw) maxw = arch[l];
  if (init) for (int l = 0; l < nl; l++) { double sc = 1.0 / sqrt((double)arch[l]); for (int i = 0; i < arch[l + 1] * arch[l]; i++) g[off[l] + i] = (float)(gauss(r) * sc);
    for (int i = 0; i < arch[l + 1]; i++) g[off[l] + arch[l + 1] * arch[l] + i] = 0; }
  double *m = (double*)calloc(nw, sizeof(double)), *v = (double*)calloc(nw, sizeof(double)), *gr = (double*)calloc(nw, sizeof(double));
  float* A = (float*)malloc(sizeof(float) * (size_t)(nl + 1) * maxw); float* D = (float*)malloc(sizeof(float) * 2 * maxw);
  int B = 128, t = 0; double lr = 1e-3, b1 = 0.9, b2 = 0.999, t0 = br_now(), loss = 0;
  int* idx = (int*)malloc(sizeof(int) * d->n); for (int i = 0; i < d->n; i++) idx[i] = i;
  for (int ep = 0; ep < 60 && br_now() - t0 < BC_SECONDS; ep++) {
    for (int i = d->n - 1; i > 0; i--) { int j = (int)(urand(r) * (i + 1)); int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp; }
    loss = 0;
    for (int b0 = 0; b0 + B <= d->n; b0 += B) {
      memset(gr, 0, sizeof(double) * nw);
      for (int q = 0; q < B; q++) {
        const float* x = d->X + (size_t)idx[b0 + q] * d->nin; const float* y = d->Y + (size_t)idx[b0 + q] * d->nout;
        memcpy(A, x, sizeof(float) * arch[0]);
        for (int l = 0; l < nl; l++) { int ni = arch[l], no = arch[l + 1]; const float* W = g + off[l]; const float* bb = W + no * ni; const float* a = A + l * maxw; float* o = A + (l + 1) * maxw;
          for (int j = 0; j < no; j++) { double acc = bb[j]; for (int i = 0; i < ni; i++) acc += W[j * ni + i] * a[i]; o[j] = (float)tanh(acc); } }
        float* dl = D; const float* aL = A + nl * maxw;
        for (int j = 0; j < arch[nl]; j++) { double e = aL[j] - y[j]; loss += e * e; dl[j] = (float)(e * (1 - aL[j] * aL[j])); }
        for (int l = nl - 1; l >= 0; l--) { int ni = arch[l], no = arch[l + 1]; const float* W = g + off[l]; const float* a = A + l * maxw; double* gW = gr + off[l]; double* gb = gW + no * ni;
          float* dp = (dl == D) ? D + maxw : D;
          for (int j = 0; j < no; j++) { gb[j] += dl[j]; for (int i = 0; i < ni; i++) gW[j * ni + i] += dl[j] * a[i]; }
          if (l) { for (int i = 0; i < ni; i++) { double s2 = 0; for (int j = 0; j < no; j++) s2 += W[j * ni + i] * dl[j]; dp[i] = (float)(s2 * (1 - a[i] * a[i])); } dl = dp; } }
      }
      t++; double c1 = 1 - pow(b1, t), c2 = 1 - pow(b2, t);
      for (int k = 0; k < nw; k++) { double gk = gr[k] / B; m[k] = b1 * m[k] + (1 - b1) * gk; v[k] = b2 * v[k] + (1 - b2) * gk * gk; g[k] -= (float)(lr * (m[k] / c1) / (sqrt(v[k] / c2) + 1e-8)); }
    }
  }
  set_msg("head start: copied the guidance algorithm (fit error %.3f per output)", d->n >= B ? loss / ((double)(d->n / B) * B * arch[nl]) : 0);
  free(m); free(v); free(gr); free(A); free(D); free(idx);
}
// Reach / intercept: record the algorithm flying this mode's scenarios.
static int bc_flights(const TrainCfg* c, float* g) {
  BrParams P; setup_params(&P, c, 64); Rng r; rng_seed(&r, 9091); ScenSet sc = {0}; scen_make(&sc, 64, &r, c, &P);
  BcData d = { 0 }; d.nin = P.nin; d.nout = 3; d.cap = BC_MAXN; d.X = (float*)malloc(sizeof(float) * (size_t)d.cap * d.nin); d.Y = (float*)malloc(sizeof(float) * (size_t)d.cap * 3);
  float* st = (float*)malloc(sizeof(float) * P.stride); float hist[BR_NI * 4] = {0}; float x[MAXW];
  float* wt = (float*)malloc(sizeof(float) * P.nw); int ok = 0;
  for (int rnd = 0; rnd < BC_ROUNDS; rnd++) {
    int quota = d.n + (d.cap - d.n) / (BC_ROUNDS - rnd);   // spread the data over the rounds
    if (rnd) br_transpose_genome(g, wt, &P);
    for (int si = 0; si < sc.n && d.n < quota; si++) {
      int sj = (si + rnd * 17) % sc.n;
      br_init(st, &P, &sc.sc[sj]); BrState s; br_load(&s, st); BrTgt tg; memset(&tg, 0, sizeof tg); memset(hist, 0, sizeof hist);
      float* nh = st + S_HIST; float* nm = st + S_HIST + BR_NI * P.K;
      for (int k = 0; k < 400000 && s.alive && d.n < quota; k++) {
        if (!br_target(&s, &P, &sc.sc[sj], sc.traj, &tg)) break;
        if (s.k % P.ctrl == 0) {
          memset(x, 0, sizeof(float) * P.nin); br_inputs(&s, &tg, x);
          for (int i = 0; i < BR_NI * P.K; i++) x[BR_NI + i] = hist[i];
          for (int i = BR_NI * P.K - 1; i >= BR_NI; i--) hist[i] = hist[i - BR_NI]; for (int i = 0; i < BR_NI && P.K; i++) hist[i] = x[i];
          float u[3]; br_algo_control(&s, &P, &sc.sc[sj], &tg, u); bc_add(&d, x, u, 1);   // the algorithm's answer is the label
          if (rnd) br_control(&s, &P, &tg, wt, nh, nm); else { s.u0 = u[0]; s.u1 = u[1]; s.u2 = u[2]; }   // who flies
        }
        s.k++; br_step(&s, &P, &sc.sc[sj], &tg);
      }
    }
    if (d.n > 500) { bc_fit(g, P.nl, P.arch, &d, &r, rnd == 0); ok = 1; }
  }
  free(st); free(wt); scen_free(&sc);
  free(d.X); free(d.Y); return ok ? 0 : -1;
}
// Swarm: record battles flown by the algorithm on both sides, as `side`'s network (nearest-K or commander) would see them.
static int bc_battles(const TrainCfg* c, const BrParams* P0, int side, float* g) {
  BrParams P = *P0; P.attAI = P.defAI = 0; Rng r; rng_seed(&r, 9092 + side); int S = 48;
  ScenSet sc = {0}; TrainCfg cc = *c; scen_make(&sc, S, &r, &cc, &P);
  int cmd = side ? P0->defCmd : P0->attCmd, nin = side ? P0->defNin : P0->attNin, nout = side ? P0->defNout : P0->attNout;
  BcData d = { 0 }; d.nin = nin; d.nout = nout; d.cap = cmd ? BC_MAXN / 4 : BC_MAXN; d.X = (float*)malloc(sizeof(float) * (size_t)d.cap * nin); d.Y = (float*)malloc(sizeof(float) * (size_t)d.cap * nout);
  float* g0 = (float*)malloc(sizeof(float) * br_sw_stride(&P)); float a[SW_MAXW], b[SW_MAXW], u[BR_SW_MAXB * 3];
  BrParams PA = P; PA.thrust = P.thrustAtk; int nA = P.attN, nB = nA + P.defN, o0 = side ? nA : 0, o1 = side ? nB : nA; float nz = P.noise * 1.732f;
  int arch[BR_MAXL + 1]; for (int l = 0; l <= P0->nl; l++) arch[l] = P0->arch[l]; arch[0] = nin; arch[P0->nl] = nout;
  int nwS = side ? P0->defNw : P0->attNw, ok = 0; float* wt = (float*)malloc(sizeof(float) * nwS);
  BrParams PN = *P0; if (side) PN.attAI = 0; else PN.defAI = 0;   // network flies `side` (later rounds), the algorithm the other
  static _Thread_local float BX[BR_SW_MAXA * SW_MAXW], BY[BR_SW_MAXA * SW_MAXW];
  for (int rnd = 0; rnd < BC_ROUNDS; rnd++) {
  int quota = d.n + (d.cap - d.n) / (BC_ROUNDS - rnd);
  if (rnd) br_transpose_layers(g, wt, P0->nl, arch);
  for (int si0 = 0; si0 < S && d.n < quota; si0++) {
    int si = (si0 + rnd * 11) % S;
    const BrScen* s0 = &sc.sc[si]; br_sw_init(g0, &P, s0, sc.traj); BR_U32 seed = (BR_U32)s0->seed;
    for (int it = 0; it < 400000 && g0[SWH_DONE] == 0 && d.n < quota; it++) {
      int k = (int)g0[SWH_K];
      if (k % P.ctrl == 0) {
        br_sw_assign(g0, &P);
        for (int bi = 0; bi < nB; bi++) br_sw_decide_ball(g0, &P, &PA, s0, NULL, NULL, bi, seed, k, nz, a, b);   // the algorithm (labels)
        if (cmd) { br_sw_cmd_inputs(g0, P0, s0, side, seed, (BR_U32)k, nz, a);
          for (int bi = o0; bi < o1; bi++) { const float* q = g0 + SW_H + bi * SW_B; int j = (bi - o0) * 3; u[j] = q[S_ALIVE] ? q[S_U0] : 0.5f; u[j + 1] = q[S_ALIVE] ? q[S_U1] : 0; u[j + 2] = q[S_ALIVE] ? q[S_U2] : 0; }
          bc_add(&d, a, u, o1 - o0); }
        else for (int bi = o0; bi < o1 && d.n < d.cap; bi++) { const float* q = g0 + SW_H + bi * SW_B; if (q[S_ALIVE] == 0) continue;
          BrState s; br_sw_load(&s, q); BrTgt tg;
          tg.x = tg.px = tg.sx = s0->tx; tg.y = tg.py = tg.sy = s0->ty; tg.z = tg.pz = tg.sz = s0->tz; tg.vx = tg.vy = tg.vz = tg.svx = tg.svy = tg.svz = 0.0f;
          int t = side ? (int)q[SB_TGT] : -1;
          if (t >= 0) { float v[6]; br_sw_seen(g0 + SW_H + t * SW_B, seed, (BR_U32)k, t, nz, v); tg.sx = v[0]; tg.sy = v[1]; tg.sz = v[2]; tg.svx = v[3]; tg.svy = v[4]; tg.svz = v[5]; }
          br_sw_nk_inputs(&s, &tg, g0, P0, bi, seed, k, nz, a); float uu[3] = { q[S_U0], q[S_U1], q[S_U2] }; bc_add(&d, a, uu, 1); }
        if (rnd) {   // the network takes over its side's controls
          if (cmd) br_sw_decide_cmd(g0, &PN, s0, side ? NULL : wt, side ? wt : NULL, side, seed, k, nz, a, b);
          else br_sw_decide_nk_side(g0, &PN, s0, wt, side, seed, k, nz, BX, BY); }
      }
      for (int bi = 0; bi < nB; bi++) br_sw_move_ball(g0, &P, &PA, s0, bi);
      for (int dd = nA; dd < nB; dd++) br_sw_scan(g0, &P, dd);
      br_sw_resolve(g0, &P, s0);
    }
  }
  if (d.n > 500) { bc_fit(g, P0->nl, arch, &d, &r, rnd == 0); ok = 1; }
  }
  free(g0); free(wt); scen_free(&sc);
  free(d.X); free(d.Y); return ok ? 0 : -1;
}

static void build_islands(void) {
  free_islands(); g_optKind = T.cfg.optKind;
  T.run = curriculum(&T.cfg, T.level); TrainCfg* c = &T.run; setup_params(&T.P, c, c->scen);
  int k = c->islands ? c->nIsl : 1; if (k > c->pop / 4) k = c->pop / 4 > 1 ? c->pop / 4 : 1;
  if (T.popMul <= 0) T.popMul = 1;
  int per = (int)(c->pop * T.popMul) / k; if (per < 4) per = 4;
  float* seed = (float*)malloc(sizeof(float) * T.P.nw); int seeded = 0, copiedStart = 0;
  br_lock(&T.mx); if (star_matches(&T.P)) { memcpy(seed, T.star, sizeof(float) * T.P.nw); seeded = 1; } br_unlock(&T.mx);
  for (int f = 0; f < BR_NI; f++) { T.normMu[f] = 0; T.normSd[f] = 1; } T.normOn = 0;   // statistics start neutral (no change)
  if (!seeded && c->imitate && c->mode != BR_MODE_SWARM) { set_msg("head start: recording the guidance algorithm…");
    if (!bc_flights(c, seed)) { seeded = 1; copiedStart = 1;
      br_lock(&T.mx); if (T.starNw != T.P.nw) { free(T.star); T.star = (float*)malloc(sizeof(float) * T.P.nw); T.starNw = T.P.nw; }
      memcpy(T.star, seed, sizeof(float) * T.P.nw); T.starNl = T.P.nl; memcpy(T.starArch, T.P.arch, sizeof(int) * (T.P.nl + 1)); T.starK = T.P.K; T.starMem = T.P.rec;
      T.starDt = c->dt; T.starTw = c->tw; T.starEvery = c->everyStep; T.starVersion++; br_unlock(&T.mx);
      if (T.be) { double v = validate(seed); set_msg("head start: the copy of the guidance algorithm scores %.0f%% on validation", v * 100); } } }
  T.isl = (Island*)calloc(k, sizeof(Island)); T.nIsl = k;
  for (int i = 0; i < k; i++) {
    Island* I = &T.isl[i]; I->cma = c->cma; I->bestG = (float*)calloc(T.P.nw, sizeof(float)); I->bestF = -1e30;
    float* s = (float*)malloc(sizeof(float) * T.P.nw);
    if (seeded) memcpy(s, seed, sizeof(float) * T.P.nw); else { random_genome(s, &T.P, &T.rng); if (c->cma) for (int j = 0; j < T.P.nw; j++) s[j] *= 0.2f; }
    if (c->cma) cma_init(&I->c, T.P.nw, per, s, seeded ? (T.restartSigma > 0 ? T.restartSigma : 0.02) : 0.25); else ga_init(&I->ga, T.P.nw, per, s, seeded ? 0.03 : 0.1, &T.rng);
    free(s);
  }
  free(seed);
  if (!seeded || copiedStart) T.level = 0;
  if (!seeded) { br_lock(&T.mx); T.histN = 0; T.gen = 0; br_unlock(&T.mx); T.restarts = 0; T.stall = 0; T.bestVal = 0; T.popMul = 1; T.restartSigma = 0; T.bestFitAvg = -1e30; T.fitSum = 0; T.fitN = 0; }
  T.structKey[0] = c->layers; T.structKey[1] = c->width; T.structKey[2] = c->K; T.structKey[3] = c->mem;
  T.structKey[4] = c->pop; T.structKey[5] = c->islands ? c->nIsl : 0; T.structKey[6] = c->cma; T.structKey[7] = 1;
  T.scenAge = 1 << 30;
}
static int needs_rebuild(const TrainCfg* c) {
  return !T.isl || T.structKey[0] != c->layers || T.structKey[1] != c->width || T.structKey[2] != c->K || T.structKey[3] != c->mem ||
         T.structKey[4] != c->pop || T.structKey[5] != (c->islands ? c->nIsl : 0) || T.structKey[6] != c->cma;
}
static int ensure_backend(void) {
  const char* want = strcmp(T.cfg.backend, "auto") ? T.cfg.backend : default_backend_id();
  if (T.be && !strcmp(T.beId, want) && T.beThreads == T.cfg.threads) { T.be->wg = T.cfg.wg; T.be->chunkMs = T.cfg.chunkMs; T.be->swLayout = T.cfg.swLayout; return 0; }
  if (T.be) { T.be->destroy(T.be); T.be = NULL; }
  char err[256] = "";
  T.be = make_backend(want, &T.cfg, err, sizeof err);
  if (!T.be) { set_msg("cannot start %s: %s", want, err); return -1; }
  snprintf(T.beId, sizeof T.beId, "%s", want); T.beThreads = T.cfg.threads; T.be->swLayout = T.cfg.swLayout;
  br_lock(&T.mx); snprintf(T.beInfo, sizeof T.beInfo, "%s", T.be->info); br_unlock(&T.mx);
  return 0;
}

static double validate(const float* g) {
  BrParams P; setup_params(&P, &T.run, T.run.valScen);
  if (!T.valReady || T.val.n != P.S || !scen_same(&T.valCfg, &T.run)) {
    Rng r; rng_seed(&r, 4242); scen_make(&T.val, P.S, &r, &T.run, &P); T.valCfg = T.run; T.valReady = 1;
  }
  float* out = (float*)malloc(sizeof(float) * BR_OUT * P.S);
  int hits = 0; if (!eval_set(T.be, g, 1, &T.val, &P, out)) for (int i = 0; i < P.S; i++) hits += out[i * BR_OUT + 1] > 0.5f;
  free(out); return (double)hits / P.S;
}

// ---------------- swarm training: one optimizer per AI side; AI vs AI is self-play against a pool of recent stars
#define SW_POOL 8
typedef struct {
  int ai, nin, nout, nw, key[8], live;   // key: the network/optimizer layout this side was built for
  CMA cma;
  float* star; double starFit, best, mean; int starVersion;
  float* pool; int poolN, poolNext;       // this side's recent stars: the other side trains against them
  double val; int valGen;                 // validation vs the algorithm: catch rate (defenders) or leak rate (attackers)
  double bestVal, fitSum, bestFitAvg; int stall, restarts, fitN;   // restarts when validation and the training score stall
} SwSide;
static struct {
  SwSide side[2];                         // 0 = attackers, 1 = defenders
  ScenSet scen, val; int scenAge, valReady; TrainCfg scenCfg, valCfg;
  double catchRate, leakRate, battlesPerS, ballStepsPerS;
} SW;
// After a validation: two in a row at diffAt or better move up a level (and restart the stall counters).
static void curriculum_check(double val) {
  if (!T.cfg.autoDiff || T.level >= 10) return;
  if (val + 1e-9 < T.cfg.diffAt) { T.levelHits = 0; return; }
  if (++T.levelHits < 2) return;
  T.level++; T.levelHits = 0; T.stall = 0; T.bestVal = 0; T.bestFitAvg = -1e30;
  for (int side = 0; side < 2; side++) { SW.side[side].stall = 0; SW.side[side].bestVal = 0; SW.side[side].bestFitAvg = -1e30; }
  set_msg("difficulty: level %d of 10%s", T.level, T.level == 10 ? " (your settings)" : "");
}
static const char* sw_side_name(int side) { return side ? "defend" : "attack"; }
// One saved star per side and matchup: commander brains depend on both counts; nearest-K brains only on K.
static const char* sw_save_path(const TrainCfg* c, int side) {
  static char p[1024]; char name[128]; const char* h = getenv("HOME");
  int cmd = side ? c->defCmd : c->attCmd;
  if (cmd) snprintf(name, sizeof name, "BallArena-star-swarm-cmd-A%d-D%d-%s.json", c->attN, c->defN, sw_side_name(side));
  else snprintf(name, sizeof name, "BallArena-star-swarm-nk-K%d-%s.json", c->swK, sw_side_name(side));
#ifdef _WIN32
  if (!h) h = getenv("USERPROFILE");
  snprintf(p, sizeof p, "%s\\%s", h ? h : ".", name);
#else
  snprintf(p, sizeof p, "%s/%s", h ? h : ".", name);
#endif
  return p;
}
static void sw_arch(const BrParams* P, int side, int* arch) {
  for (int l = 0; l <= P->nl; l++) arch[l] = P->arch[l];
  arch[0] = side ? P->defNin : P->attNin; arch[P->nl] = side ? P->defNout : P->attNout;
}
static void sw_save(const TrainCfg* c, const BrParams* P, int side) {
  SwSide* S = &SW.side[side]; if (!S->star) return;
  int arch[BR_MAXL + 1]; sw_arch(P, side, arch);
  Sb o = {0};
  sb_printf(&o, "{\"format\":\"ball-arena-network\",\"mode\":\"swarm\",\"side\":\"%s\",\"brain\":\"%s\",\"attackers\":%d,\"defenders\":%d,\"K\":%d,\"arch\":[",
    sw_side_name(side), (side ? c->defCmd : c->attCmd) ? "commander" : "nearest-k", c->attN, c->defN, c->swK);
  for (int l = 0; l <= P->nl; l++) sb_printf(&o, "%s%d", l ? "," : "", arch[l]);
  sb_printf(&o, "],\"dt\":%g,\"ctrlEvery\":%d,\"thrustToWeight\":%g,\"runnerThrustToWeight\":%g,\"generation\":%d,\"fitness\":%.4f,\"validation\":%.4f,\"weights\":[",
    c->dt, c->everyStep ? 1 : 2, c->tw, c->twRunner, T.gen, S->starFit, S->val);
  for (int i = 0; i < S->nw; i++) sb_printf(&o, "%s%.7g", i ? "," : "", S->star[i]);
  sb_printf(&o, "]}");
  FILE* f = fopen(sw_save_path(c, side), "w"); if (f) { fwrite(o.s, 1, o.n, f); fclose(f); }
  free(o.s);
}
// Weights of this side's saved star if its layer sizes match the current settings (NULL otherwise).
static float* sw_load(const TrainCfg* c, const BrParams* P, int side) {
  char* j = NULL; { FILE* f = fopen(sw_save_path(c, side), "rb"); if (!f) return NULL; fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    j = (char*)malloc(n + 1); n = (long)fread(j, 1, n, f); j[n] = 0; fclose(f); }
  int arch[BR_MAXL + 1], want[BR_MAXL + 1], nl = -1; sw_arch(P, side, want);
  const char* a = strstr(j, "\"arch\""); const char* w = strstr(j, "\"weights\""); const char* p = a ? strchr(a, '[') : NULL;
  if (!p || !w) { free(j); return NULL; } p++;
  while (*p && *p != ']' && nl < BR_MAXL) { arch[++nl] = (int)strtol(p, (char**)&p, 10); while (*p == ',' || *p == ' ') p++; }
  int ok = nl == P->nl; for (int l = 0; ok && l <= nl; l++) ok = arch[l] == want[l];
  float* g = NULL; int nw = side ? P->defNw : P->attNw;
  if (ok) { g = (float*)malloc(sizeof(float) * nw); p = strchr(w, '['); int n = 0;
    if (p) { p++; while (n < nw && *p && *p != ']') { g[n++] = strtof(p, (char**)&p); while (*p == ',' || *p == ' ') p++; } }
    if (n != nw) { free(g); g = NULL; } }
  free(j); return g;
}
static void sw_side_free(SwSide* S) { if (S->live) cma_free(&S->cma); free(S->star); free(S->pool); memset(S, 0, sizeof *S); }
// (Re)build a side when its network or population layout changed: seeded from its saved star when that fits.
static void sw_build(const TrainCfg* c, const BrParams* P, int side, int lam) {
  SwSide* S = &SW.side[side]; int ai = side ? c->defAI : c->attAI, cmd = side ? c->defCmd : c->attCmd;
  int key[8] = { ai, c->layers, c->width, cmd, cmd ? c->attN : 0, cmd ? c->defN : 0, cmd ? 0 : c->swK, lam };
  if (S->live == ai && !memcmp(S->key, key, sizeof key)) return;
  sw_side_free(S); memcpy(S->key, key, sizeof key); S->ai = ai; g_optKind = c->optKind; S->bestFitAvg = -1e30;
  if (!ai) return;
  S->nin = side ? P->defNin : P->attNin; S->nout = side ? P->defNout : P->attNout; S->nw = side ? P->defNw : P->attNw;
  float* seed = sw_load(c, P, side); int seeded = seed != NULL, copied = 0;
  if (!seeded) T.level = 0;   // a new network starts at the easy end (automatic difficulty)
  if (!seed) { seed = (float*)malloc(sizeof(float) * S->nw); for (int i = 0; i < S->nw; i++) seed[i] = (float)((urand(&T.rng) * 2 - 1) * 0.1);
    if (c->imitate) { set_msg("%s head start: recording the guidance algorithm…", side ? "defenders" : "attackers"); copied = !bc_battles(c, P, side, seed); } }
  cma_init(&S->cma, S->nw, lam, seed, seeded ? 0.02 : copied ? 0.05 : 0.25); S->live = 1;
  S->star = seed; S->starFit = -1e30; S->pool = (float*)malloc(sizeof(float) * S->nw * SW_POOL);
  memcpy(S->pool, seed, sizeof(float) * S->nw); S->poolN = 1; S->poolNext = 1;
  set_msg("%s: %s", side ? "defenders" : "attackers", seeded ? "continuing from the saved star" : copied ? "new network, started by copying the guidance algorithm" : "new network");
}
// Score of one battle for one side (normalised by the counts, so scores compare across swarm sizes).
static double sw_fit(const float* o, int side, const TrainCfg* c) {
  double nA = c->attN, nD = c->defN;
  if (side) return (5 * o[0] - 5 * o[1]) / nA + 0.5 * o[3] / nD;   // defenders: catches, leaks, closeness to attackers
  return (5 * o[1] - 2 * o[0]) / nA + 0.5 * o[2] / nA;             // attackers: leaks, losses, closeness to the point
}
// Validation: each AI side's star against the algorithm on a fixed set of battles.
static void sw_validate(const TrainCfg* c) {
  BrParams P; setup_params(&P, c, c->valScen);
  if (!SW.valReady || SW.val.n != P.S || !scen_same(&SW.valCfg, c)) { Rng r; rng_seed(&r, 4242); scen_make(&SW.val, P.S, &r, c, &P); SW.valCfg = *c; SW.valReady = 1; }
  int* bat = (int*)calloc((size_t)P.S * 3, sizeof(int)); float* out = (float*)malloc(sizeof(float) * BR_SWOUT * P.S);
  for (int i = 0; i < P.S; i++) bat[i * 3 + 2] = i;
  for (int side = 0; side < 2; side++) {
    SwSide* S = &SW.side[side]; if (!S->ai || !S->star) continue;
    BrParams Q = P; if (side) Q.attAI = 0; else Q.defAI = 0;   // the other side: the algorithm
    if (T.be->evalBattles(T.be, side ? NULL : S->star, side ? S->star : NULL, bat, P.S, SW.val.sc, SW.val.traj, SW.val.trajFloats, &Q, out)) continue;
    double n = 0; for (int i = 0; i < P.S; i++) n += out[i * BR_SWOUT + (side ? 0 : 1)];
    br_lock(&T.mx); S->val = n / ((double)P.S * c->attN); S->valGen = T.gen; br_unlock(&T.mx);
    double avg = S->fitN ? S->fitSum / S->fitN : -1e30; S->fitSum = 0; S->fitN = 0;
    int better = S->val > S->bestVal + 1e-9 || avg > S->bestFitAvg + 0.05;
    if (S->val > S->bestVal) S->bestVal = S->val; if (avg > S->bestFitAvg) S->bestFitAvg = avg;
    if (better) S->stall = 0; else S->stall++;
    if (c->restarts && S->stall >= 6 && S->bestVal < 1) {   // restart this side's optimizer around its star (BIPOP-style)
      S->restarts++; S->stall = 0; int lam = S->cma.lam, big = S->restarts & 1;
      int nl = big ? (lam * 2 > 4096 ? 4096 : lam * 2) : (lam / 2 > 8 ? lam / 2 : 8);
      cma_free(&S->cma); g_optKind = c->optKind; cma_init(&S->cma, S->nw, nl, S->star, big ? 0.1 : 0.01);
      set_msg("%s: validation stalled at %.0f%%, restart %d (%s search)", side ? "defenders" : "attackers", S->bestVal * 100, S->restarts, big ? "wide" : "fine");
    }
  }
  free(bat); free(out);
}
// One swarm generation. Returns -1 to stop training.
static int sw_generation(void) {
  T.run = curriculum(&T.cfg, T.level); TrainCfg* c = &T.run;
  if (!c->attAI && !c->defAI) { set_msg("both sides fly the algorithm: set one side to AI to train"); return -1; }
  if (!T.be->evalBattles) { set_msg("%s cannot fly swarm battles yet", T.be->info); return -1; }
  BrParams P; setup_params(&P, c, c->scen);
  int nAI = c->attAI + c->defAI, lam = c->pop / nAI; if (lam < 8) lam = 8;
  for (int side = 0; side < 2; side++) sw_build(c, &P, side, lam);
  if (SW.scen.n != c->scen || SW.scenAge >= c->reuse || !scen_same(&SW.scenCfg, c)) { scen_make(&SW.scen, c->scen, &T.rng, c, &P); SW.scenCfg = *c; SW.scenAge = 0; }
  SW.scenAge++;
  double t0 = br_now(), bsteps = 0, catches = 0, leaks = 0; int battles = 0;
  for (int side = 0; side < 2; side++) {
    SwSide* S = &SW.side[side]; if (!S->ai) continue;
    SwSide* O = &SW.side[side ^ 1];
    cma_ask(&S->cma, &T.rng);
    int L = S->cma.lam;   // this side's population (a restart may have changed it)
    int nb = L * c->scen; int* bat = (int*)malloc(sizeof(int) * 3 * nb); float* out = (float*)malloc(sizeof(float) * BR_SWOUT * nb);
    for (int i = 0; i < L; i++) for (int j = 0; j < c->scen; j++) { int b = i * c->scen + j, opp = O->ai ? (i + j) % O->poolN : 0;
      bat[b * 3] = side ? opp : i; bat[b * 3 + 1] = side ? i : opp; bat[b * 3 + 2] = j; }
    const float* mine = S->cma.X; const float* theirs = O->ai ? O->pool : NULL;
    if (T.be->evalBattles(T.be, side ? theirs : mine, side ? mine : theirs, bat, nb, SW.scen.sc, SW.scen.traj, SW.scen.trajFloats, &P, out)) {
      free(bat); free(out); set_msg("battle evaluation failed on %s", T.be->info); return -1; }
    double* fit = (double*)malloc(sizeof(double) * L); int best = 0; double mean = 0;
    for (int i = 0; i < L; i++) { double f = 0; for (int j = 0; j < c->scen; j++) { const float* o = out + (size_t)(i * c->scen + j) * BR_SWOUT; f += sw_fit(o, side, c); bsteps += o[4]; }
      fit[i] = f / c->scen; mean += fit[i]; if (fit[i] > fit[best]) best = i; }
    for (int j = 0; j < c->scen; j++) { const float* o = out + (size_t)(best * c->scen + j) * BR_SWOUT; catches += o[0]; leaks += o[1]; }
    battles += nb;
    br_lock(&T.mx);
    memcpy(S->star, mine + (size_t)best * S->nw, sizeof(float) * S->nw); S->starFit = fit[best]; S->best = fit[best]; S->fitSum += fit[best]; S->fitN++; S->mean = mean / L; S->starVersion++;
    br_unlock(&T.mx);
    cma_tell(&S->cma, fit); if (c->decayOn) S->cma.sigma *= c->decay; if (S->cma.sigma < c->lrMin) S->cma.sigma = c->lrMin;
    free(fit); free(bat); free(out);
  }
  double dt = br_now() - t0; int gen;
  br_lock(&T.mx);
  gen = ++T.gen; T.genTime = dt; SW.battlesPerS = battles / dt; SW.ballStepsPerS = bsteps / dt; T.flightsPerS = SW.battlesPerS; T.stepsPerS = SW.ballStepsPerS;
  SW.catchRate = catches / ((double)c->scen * nAI * c->attN); SW.leakRate = leaks / ((double)c->scen * nAI * c->attN);
  T.best = c->defAI ? SW.side[1].best : SW.side[0].best; T.mean = c->defAI ? SW.side[1].mean : SW.side[0].mean;
  T.sigma = c->defAI ? SW.side[1].cma.sigma : SW.side[0].cma.sigma; T.starVersion++;
  if (T.histN == HIST_MAX) { memmove(T.hist, T.hist + 1, sizeof(Hist) * (HIST_MAX - 1)); T.histN--; }
  Hist* h = &T.hist[T.histN++]; memset(h, 0, sizeof *h); h->gen = gen; h->best = (float)T.best; h->mean = (float)T.mean;
  h->isl[0] = (float)SW.side[0].best; h->isl[1] = (float)SW.side[1].best;
  br_unlock(&T.mx);
  // self-play: every few generations each side's star joins its pool (the other side's opponents)
  if (gen % 5 == 0) for (int side = 0; side < 2; side++) { SwSide* S = &SW.side[side]; if (!S->ai) continue;
    memcpy(S->pool + (size_t)S->poolNext * S->nw, S->star, sizeof(float) * S->nw); S->poolNext = (S->poolNext + 1) % SW_POOL; if (S->poolN < SW_POOL) S->poolN++; }
  if (gen % c->valEvery == 0) { sw_validate(c); for (int side = 0; side < 2; side++) if (SW.side[side].ai) sw_save(c, &P, side);
    double lo = 1; for (int side = 0; side < 2; side++) if (SW.side[side].ai && SW.side[side].val < lo) lo = SW.side[side].val;
    curriculum_check(lo); }   // every AI side must reach the target
  return 0;
}
static void sw_autosave(void) {
  if (T.cfg.mode != BR_MODE_SWARM) return;
  BrParams P; setup_params(&P, &T.cfg, T.cfg.scen);
  for (int side = 0; side < 2; side++) if (SW.side[side].ai && SW.side[side].star && SW.side[side].nw == (side ? P.defNw : P.attNw)) sw_save(&T.cfg, &P, side);
}

static void train_thread(void* arg) {
  (void)arg;
  float* flat = NULL; size_t flatCap = 0; float* out = NULL; size_t outCap = 0; double* fit = NULL; int fitCap = 0;
  set_msg("training");
  while (!T.stopReq) {
    if (T.hasPending) { br_lock(&T.mx); TrainCfg nc = T.pending; T.hasPending = 0; br_unlock(&T.mx);
      if (nc.mode != T.starMode) trainer_set_mode(nc.mode);
      T.cfg = nc; }
    if (ensure_backend()) break;
    T.run = curriculum(&T.cfg, T.level);
    if (T.cfg.mode == BR_MODE_SWARM) { if (sw_generation()) break; continue; }
    if (needs_rebuild(&T.cfg)) build_islands();
    TrainCfg* c = &T.run;
    setup_params(&T.P, c, c->scen);   // physics settings may have changed
    double tScen = br_now();   // tag scenarios fly the runner on the CPU: count that time
    if (T.scen.n != c->scen || T.scenAge >= c->reuse || !scen_same(&T.scenCfg, c)) {
      scen_make(&T.scen, c->scen, &T.rng, c, &T.P); T.scenCfg = *c;
      T.scenAge = 0;
    }
    tScen = br_now() - tScen;
    T.scenAge++;
    int n = 0; for (int i = 0; i < T.nIsl; i++) { if (T.isl[i].cma) cma_ask(&T.isl[i].c, &T.rng); n += isl_size(&T.isl[i]); }
    size_t need = (size_t)n * T.P.nw; if (need > flatCap) { flat = (float*)realloc(flat, sizeof(float) * need); flatCap = need; }
    size_t needO = (size_t)n * c->scen * BR_OUT; if (needO > outCap) { out = (float*)realloc(out, sizeof(float) * needO); outCap = needO; }
    if (n > fitCap) { fit = (double*)realloc(fit, sizeof(double) * n); fitCap = n; }
    { size_t o = 0; for (int i = 0; i < T.nIsl; i++) { size_t k = (size_t)isl_size(&T.isl[i]) * T.P.nw; memcpy(flat + o, isl_genomes(&T.isl[i]), sizeof(float) * k); o += k; } }
    const float* evalW = flat;   // the networks as flown: with sensor standardisation folded into the first layer
    if (T.normOn) { if (need > T.effCap) { T.eff = (float*)realloc(T.eff, sizeof(float) * need); T.effCap = need; }
      for (int i = 0; i < n; i++) norm_fold(flat + (size_t)i * T.P.nw, T.eff + (size_t)i * T.P.nw, &T.P, 0); evalW = T.eff; }
    double t0 = br_now();
    if (eval_set(T.be, evalW, n, &T.scen, &T.P, out)) { set_msg("evaluation failed on %s", T.be->info); break; }
    double dt = br_now() - t0 + tScen, steps = 0, mean = 0; int star = 0;
    for (int i = 0; i < n; i++) { double f = 0; for (int j = 0; j < c->scen; j++) { const float* o = out + ((size_t)i * c->scen + j) * BR_OUT; f += fitness(o, c->speedW, c->mode, T.scen.sc[j].atkHit, c->altOn, tag_r(c)); steps += o[3]; }
      fit[i] = f / c->scen; mean += fit[i]; if (fit[i] > fit[star]) star = i; }
    mean /= n;
    int hits = 0; double miss = 0; for (int j = 0; j < c->scen; j++) { const float* o = out + ((size_t)star * c->scen + j) * BR_OUT; hits += o[1] > 0.5f; miss += o[0]; }
    // tell each island its slice; decay + floor; island bests for the chart
    float islBest[ISL_SHOW] = {0}; double sig = 0; int off = 0;
    for (int i = 0; i < T.nIsl; i++) {
      Island* I = &T.isl[i]; int k = isl_size(I); int bi = 0;
      for (int q = 1; q < k; q++) if (fit[off + q] > fit[off + bi]) bi = q;
      I->bestF = fit[off + bi]; memcpy(I->bestG, flat + (size_t)(off + bi) * T.P.nw, sizeof(float) * T.P.nw);
      if (i < ISL_SHOW) islBest[i] = (float)I->bestF;
      if (I->cma) cma_tell(&I->c, fit + off); else ga_tell(&I->ga, fit + off, &T.rng);
      double* s = isl_sigma(I); if (c->decayOn) *s *= c->decay; if (*s < c->lrMin) *s = c->lrMin; sig += *s;
      off += k;
    }
    int gen;
    br_lock(&T.mx);
    gen = ++T.gen;
    if (T.starNw != T.P.nw) { free(T.star); T.star = (float*)malloc(sizeof(float) * T.P.nw); T.starNw = T.P.nw; }
    memcpy(T.star, evalW + (size_t)star * T.P.nw, sizeof(float) * T.P.nw);
    T.starNl = T.P.nl; memcpy(T.starArch, T.P.arch, sizeof(int) * (T.P.nl + 1)); T.starK = T.P.K; T.starMem = T.P.rec; T.starDt = c->dt; T.starTw = c->tw; T.starEvery = c->everyStep; T.starVersion++; T.starFit = fit[star];
    T.best = fit[star]; T.mean = mean; T.fitSum += fit[star]; T.fitN++; T.sigma = sig / T.nIsl; T.genTime = dt; T.flightsPerS = n * c->scen / dt; T.stepsPerS = steps / dt;
    T.starHits = (double)hits / c->scen; T.starMiss = miss / c->scen;
    if (T.histN == HIST_MAX) { memmove(T.hist, T.hist + 1, sizeof(Hist) * (HIST_MAX - 1)); T.histN--; }
    Hist* h = &T.hist[T.histN++]; h->gen = gen; h->best = (float)T.best; h->mean = (float)mean; memcpy(h->isl, islBest, sizeof islBest);
    br_unlock(&T.mx);
    if (c->normIn && c->mode != BR_MODE_SWARM && (gen == 5 || gen % 50 == 0)) { float* sc = (float*)malloc(sizeof(float) * T.P.nw);
      br_lock(&T.mx); memcpy(sc, T.star, sizeof(float) * T.P.nw); br_unlock(&T.mx); norm_update(sc); free(sc); }
    // islands: pass each island's best to the next one in a ring
    if (T.nIsl > 1 && gen % c->migrate == 0)
      for (int i = 0; i < T.nIsl; i++) { Island* to = &T.isl[(i + 1) % T.nIsl]; const float* g = T.isl[i].bestG;
        if (to->cma) for (int j = 0; j < T.P.nw; j++) to->c.m[j] = 0.5 * (to->c.m[j] + g[j]);
        else memcpy(to->ga.pop + (size_t)(to->ga.lam - 1) * T.P.nw, g, sizeof(float) * T.P.nw); }
    if (gen % c->valEvery == 0) {
      float* g = (float*)malloc(sizeof(float) * T.P.nw); memcpy(g, evalW + (size_t)star * T.P.nw, sizeof(float) * T.P.nw);
      double v = validate(g); free(g);
      br_lock(&T.mx); T.valHit = v; T.valGen = gen; br_unlock(&T.mx);
      // restarts: validation not better for 6 checks in a row (and not already perfect) restarts the optimizer around the
      // star, alternating a large population with a wide search and a small one with a fine search (BIPOP-CMA-ES)
      double avg = T.fitN ? T.fitSum / T.fitN : -1e30; T.fitSum = 0; T.fitN = 0;   // mean star fitness since the last check
      int better = v > T.bestVal + 1e-9 || avg > T.bestFitAvg + 0.05;
      if (v > T.bestVal) T.bestVal = v; if (avg > T.bestFitAvg) T.bestFitAvg = avg;
      if (better) T.stall = 0; else T.stall++;   // stuck = neither validation nor the training score improved
      curriculum_check(v);
      if (c->restarts && c->cma && T.stall >= 6 && T.bestVal < 1) {
        T.restarts++; T.stall = 0;
        if (T.restarts & 1) { T.popMul = fmin(8, (T.popMul > 1 ? T.popMul : 1) * 2); T.restartSigma = 0.1; }
        else { T.popMul = 0.5; T.restartSigma = 0.01; }
        set_msg("validation stalled at %.0f%%: restart %d with a %s search (population ×%g)", T.bestVal * 100, T.restarts, T.restarts & 1 ? "wide" : "fine", T.popMul);
        build_islands();   // seeded from the star, which is kept
      }
      autosave();
    }
  }
  free(flat); free(out); free(fit);
  T.running = 0;
  if (T.stopReq) set_msg("paused");
}

void trainer_start(const TrainCfg* c) {
  br_lock(&T.mx); T.pending = *c; T.hasPending = 1; br_unlock(&T.mx);
  if (T.running || T.tuning) return;
  if (T.threadLive) { br_thread_join(&T.th); T.threadLive = 0; }
  T.stopReq = 0; T.running = 1; T.threadLive = 1;
  br_thread_start(&T.th, train_thread, NULL);
}
int trainer_busy(void) { return T.running || T.tuning; }
void trainer_pause(void) {
  T.stopReq = 1;
  if (T.threadLive) { br_thread_join(&T.th); T.threadLive = 0; }
  T.running = 0; autosave(); sw_autosave();
}
void trainer_reset(void) {
  trainer_pause();
  free_islands();
  br_lock(&T.mx);
  free(T.star); T.star = NULL; T.starNw = 0; T.starVersion++; T.gen = 0; T.histN = 0;
  T.best = T.mean = T.sigma = T.genTime = T.flightsPerS = T.stepsPerS = T.starHits = T.starMiss = T.valHit = 0; T.valGen = 0; T.level = 0; T.levelHits = 0;
  snprintf(T.msg, sizeof T.msg, "reset · training stopped");
  br_unlock(&T.mx);
  if (T.cfg.mode == BR_MODE_SWARM) {   // swarm: forget both sides and delete this matchup's saved stars
    for (int side = 0; side < 2; side++) { if ((side ? T.cfg.defAI : T.cfg.attAI)) remove(sw_save_path(&T.cfg, side)); sw_side_free(&SW.side[side]); }
    return; }
  remove(save_path());
}

void trainer_status_json(Sb* o, int since) {
  br_lock(&T.mx);
  sb_printf(o, "{\"mode\":%d,\"running\":%s,\"tuning\":%s,\"gen\":%d,\"best\":%.4f,\"mean\":%.4f,\"starHits\":%.4f,\"starMiss\":%.2f,\"flightsPerS\":%.0f,\"stepsPerS\":%.0f,"
    "\"genTime\":%.4f,\"sigma\":%.3g,\"valHit\":%.4f,\"valGen\":%d,\"starVersion\":%d,\"hasStar\":%s,\"islands\":%d,\"restarts\":%d,\"level\":%d,\"autoDiff\":%d,\"backend\":",
    T.starMode, T.running ? "true" : "false", T.tuning ? "true" : "false", T.gen, T.best, T.mean, T.starHits, T.starMiss, T.flightsPerS, T.stepsPerS,
    T.genTime, T.sigma, T.valHit, T.valGen, T.starVersion, T.star ? "true" : "false", T.nIsl, T.restarts, T.cfg.autoDiff ? T.level : 10, T.cfg.autoDiff);
  sb_jstr(o, T.beInfo); sb_printf(o, ",\"message\":"); sb_jstr(o, T.msg);
  if (T.starMode == BR_MODE_SWARM)
    sb_printf(o, ",\"swarm\":{\"attAI\":%d,\"defAI\":%d,\"attBest\":%.4f,\"defBest\":%.4f,\"catchRate\":%.4f,\"leakRate\":%.4f,\"valLeak\":%.4f,\"valCatch\":%.4f,\"attStar\":%s,\"defStar\":%s,\"battlesPerS\":%.0f}",
      SW.side[0].ai, SW.side[1].ai, SW.side[0].best, SW.side[1].best, SW.catchRate, SW.leakRate, SW.side[0].val, SW.side[1].val,
      SW.side[0].star ? "true" : "false", SW.side[1].star ? "true" : "false", SW.battlesPerS);
  sb_printf(o, ",\"hist\":[");
  int first = 1;
  for (int i = 0; i < T.histN; i++) { Hist* h = &T.hist[i]; if (h->gen <= since) continue;
    sb_printf(o, "%s[%d,%.4f,%.4f", first ? "" : ",", h->gen, h->best, h->mean); first = 0;
    int k = T.starMode == BR_MODE_SWARM ? 2 : T.nIsl > 1 ? (T.nIsl < ISL_SHOW ? T.nIsl : ISL_SHOW) : 0;   // swarm: attackers' and defenders' best
    for (int q = 0; q < k; q++) sb_printf(o, ",%.3f", h->isl[q]);
    sb_printf(o, "]"); }
  sb_printf(o, "]}");
  br_unlock(&T.mx);
}
void trainer_star_json(Sb* o) { br_lock(&T.mx); if (T.star) star_json_locked(o); else sb_printf(o, "{}"); br_unlock(&T.mx); }

void hardware_json(Sb* o) {
  sb_printf(o, "{\"cpuThreads\":%d,\"options\":[{\"id\":\"cpu\",\"label\":\"CPU (%d threads)\"}", br_cpu_count(), br_cpu_count());
#ifdef BR_HAVE_METAL
  { char e[128]; Backend* b = metal_backend_create(e, sizeof e); if (b) { sb_printf(o, ",{\"id\":\"metal\",\"label\":"); sb_jstr(o, b->info); sb_printf(o, "}"); b->destroy(b); } }
#endif
#ifdef BR_HAVE_OPENCL
  { char n[32][160]; int g[32]; int k = opencl_devices(n, g, 32);
    for (int i = 0; i < k; i++) if (g[i]) { char l[200]; snprintf(l, sizeof l, "OpenCL, %s", n[i]); sb_printf(o, ",{\"id\":\"opencl:%d\",\"label\":", i); sb_jstr(o, l); sb_printf(o, "}"); } }
#endif
  sb_printf(o, "],\"default\":\"%s\",\"savePath\":", default_backend_id()); sb_jstr(o, save_path()); sb_printf(o, "}");
}

// ---------------- auto-tune: measure every compute option on this machine and keep the fastest
static void tlog(const char* fmt, ...) {
  char line[300]; va_list ap; va_start(ap, fmt); vsnprintf(line, sizeof line, fmt, ap); va_end(ap);
  br_lock(&T.mx); if (T.tuneLog.n) sb_printf(&T.tuneLog, ","); sb_jstr(&T.tuneLog, line); br_unlock(&T.mx);
}
typedef struct { char id[32], label[160]; int n, wg; double chunk, sps, sec; } Meas;

// Throughput of one batch. genSec: CPU time to build a generation's scenarios (tag mode flies every runner), spread over the generations that reuse them, counted as part of each generation.
static double g_tuneGenSec;
static double measure(Backend* b, const float* G, int n, const ScenSet* sc, BrParams* P, float* out, double* secOut) {
  eval_set(b, G, n < 32 ? n : 32, sc, P, out);   // warm-up
  double best = 1e9, steps = 0;
  for (int r = 0; r < 2; r++) { double t0 = br_now(); if (eval_set(b, G, n, sc, P, out)) return 0; double t = br_now() - t0; if (t < best) best = t; }
  best += g_tuneGenSec;
  for (int k = 0; k < n * P->S; k++) steps += out[k * BR_OUT + 3];
  if (secOut) *secOut = best;
  return steps / best;
}

// Swarm auto-tune: one generation's battles (saved stars with small variations when they fit, else random brains)
// on every compute option and GPU work-group size; the fastest in battles per second wins.
static void sw_tune(const TrainCfg* cin) {
  TrainCfg c = *cin; Rng r; rng_seed(&r, 777);
  BrParams P; setup_params(&P, &c, c.scen);
  int nAI = c.attAI + c.defAI, lam = nAI ? c.pop / nAI : c.pop; if (lam < 8) lam = 8;
  int G = lam < 64 ? lam : 64, nb = lam * c.scen * (nAI ? nAI : 1);
  ScenSet scs = {0}; scen_make(&scs, c.scen, &r, &c, &P);
  float* w[2] = { NULL, NULL };
  for (int side = 0; side < 2; side++) { if (!(side ? c.defAI : c.attAI)) continue;
    int nw = side ? P.defNw : P.attNw; w[side] = (float*)malloc(sizeof(float) * (size_t)G * nw); float* st = sw_load(&c, &P, side);
    for (int k = 0; k < G; k++) for (int i = 0; i < nw; i++) w[side][(size_t)k * nw + i] = st ? st[i] + (float)(gauss(&r) * 0.02) : (float)((urand(&r) * 2 - 1) * 0.1);
    tlog("%s: %s", side ? "Defenders" : "Attackers", st ? "your saved star with small variations" : "random brains (no saved star for this matchup)"); free(st); }
  int* bat = (int*)malloc(sizeof(int) * 3 * nb); for (int i = 0; i < nb; i++) { bat[i * 3] = (i / c.scen) % G; bat[i * 3 + 1] = (i / c.scen + 3) % G; bat[i * 3 + 2] = i % c.scen; }
  float* out = (float*)malloc(sizeof(float) * BR_SWOUT * nb);
  tlog("Measuring one generation: %d battles of %d vs %d", nb, c.attN, c.defN);
  char ids[40][32]; int nOpt = 0; strcpy(ids[nOpt++], "cpu");
#ifdef BR_HAVE_METAL
  strcpy(ids[nOpt++], "metal");
#endif
#ifdef BR_HAVE_OPENCL
  { char nm[32][160]; int g[32]; int k = opencl_devices(nm, g, 32); for (int i = 0; i < k && nOpt < 40; i++) if (g[i]) snprintf(ids[nOpt++], 32, "opencl:%d", i); }
#endif
  Meas rows[200]; int nRows = 0, best = -1;
  for (int oi = 0; oi < nOpt; oi++) {
    int gpu = strcmp(ids[oi], "cpu") != 0;
    int wgs[5] = { 0, 32, 64, 128, -1 }, nwg = gpu ? 5 : 1;   // -1: one work-group per battle (a thread per ball)
    for (int wi = gpu ? 1 : 0; wi < nwg; wi++) {
      char err[256] = ""; TrainCfg cc = c; Backend* be = make_backend(ids[oi], &cc, err, sizeof err); if (!be) { tlog("%s: not available (%s)", ids[oi], err); break; }
      if (!be->evalBattles) { be->destroy(be); break; }
      be->wg = wgs[wi] > 0 ? wgs[wi] : 64; be->swLayout = wgs[wi] < 0 ? 2 : gpu ? 1 : 0; be->chunkMs = 40;
      double sec = 1e9, bs = 0;
      for (int k = 0; k < 2; k++) { double t0 = br_now(); if (be->evalBattles(be, w[0], w[1], bat, nb, scs.sc, scs.traj, scs.trajFloats, &P, out)) { sec = 1e9; break; } double t = br_now() - t0; if (t < sec) sec = t; }
      for (int i = 0; i < nb; i++) bs += out[i * BR_SWOUT + 4];
      Meas* m = &rows[nRows++]; snprintf(m->id, sizeof m->id, "%s", ids[oi]); snprintf(m->label, sizeof m->label, "%s", be->info);
      m->n = lam; m->wg = wgs[wi]; m->chunk = 40; m->sps = bs / sec; m->sec = sec;   // wg −1: work-group per battle
      char wl[48] = ""; if (wgs[wi] > 0) snprintf(wl, sizeof wl, " · thread per battle, work-group %d", wgs[wi]); else if (wgs[wi] < 0) snprintf(wl, sizeof wl, " · work-group per battle");
      tlog("%s%s: %.0f battles/s (%.2f s per generation)", be->info, wl, nb / sec, sec);
      if (best < 0 || sec < rows[best].sec) best = nRows - 1;
      be->destroy(be);
    }
  }
  Sb res = {0};
  if (best < 0) sb_printf(&res, "{\"ok\":false}");
  else { Meas* m = &rows[best];
    sb_printf(&res, "{\"ok\":true,\"best\":{\"backend\":\"%s\",\"label\":", m->id); sb_jstr(&res, m->label);
    sb_printf(&res, ",\"wg\":%d,\"swLayout\":%d,\"chunkMs\":40,\"stepsPerS\":%.0f,\"secPerGen\":%.4f,\"batchNetworks\":%d,\"pop\":%d,\"scen\":%d,\"islands\":0,\"nIsl\":%d},\"note\":",
      m->wg > 0 ? m->wg : 64, m->wg < 0 ? 2 : strcmp(m->id, "cpu") ? 1 : 0, m->sps, m->sec, m->n, c.pop, c.scen, c.nIsl);
    sb_jstr(&res, "Swarm: measured in battles per generation with your current matchup; population and scenarios are unchanged.");
    sb_printf(&res, ",\"options\":[");
    for (int i = 0, first = 1; i < nRows; i++) { int top = 1; for (int j = 0; j < nRows; j++) if (!strcmp(rows[j].id, rows[i].id) && rows[j].sec < rows[i].sec) top = 0;
      if (!top) continue; sb_printf(&res, "%s{\"backend\":\"%s\",\"label\":", first ? "" : ",", rows[i].id); sb_jstr(&res, rows[i].label); first = 0;
      sb_printf(&res, ",\"stepsPerS\":%.0f,\"secPerGen\":%.4f,\"batchNetworks\":%d,\"wg\":%d,\"chunkMs\":40}", rows[i].sps, rows[i].sec, rows[i].n, rows[i].wg); }
    sb_printf(&res, "],\"rows\":[");
    for (int i = 0; i < nRows; i++) { sb_printf(&res, "%s{\"backend\":\"%s\",\"label\":", i ? "," : "", rows[i].id); sb_jstr(&res, rows[i].label);
      sb_printf(&res, ",\"networks\":%d,\"flights\":%d,\"wg\":%d,\"chunkMs\":40,\"stepsPerS\":%.0f,\"sec\":%.3f}", rows[i].n, nb, rows[i].wg, rows[i].sps, rows[i].sec); }
    sb_printf(&res, "]}");
    tlog("Done: %s flies the most battles per second (%.2f s per generation).", m->label, m->sec);
  }
  br_lock(&T.mx); free(T.tuneResult.s); T.tuneResult = res; br_unlock(&T.mx);
  free(w[0]); free(w[1]); free(bat); free(out); scen_free(&scs);
}
static void tune_thread(void* arg) {
  (void)arg; TrainCfg c = T.tuneCfg; Rng r; rng_seed(&r, 777);
  if (c.mode == BR_MODE_SWARM) { sw_tune(&c); T.tuning = 0; return; }
  BrParams P; setup_params(&P, &c, c.scen);
  // A realistic population: your star ball (if it matches the current network) plus small variations. Random networks
  // crash within seconds and would mislead the measurement, so without a star a quick CPU pre-training runs first.
  float* base = (float*)malloc(sizeof(float) * P.nw); int have = 0;
  br_lock(&T.mx); if (star_matches(&P)) { memcpy(base, T.star, sizeof(float) * P.nw); have = 1; } br_unlock(&T.mx);
  if (!have) {
    tlog("No trained network for this shape yet: pre-training a few seconds on the CPU so the test flights are realistic…");
    Backend* cpu = cpu_backend_create(0); BrParams Q; setup_params(&Q, &c, 8);
    CMA cm; float* s = (float*)malloc(sizeof(float) * P.nw); random_genome(s, &P, &r); for (int j = 0; j < P.nw; j++) s[j] *= 0.2f;
    cma_init(&cm, P.nw, 96, s, 0.25); free(s);
    ScenSet sc8 = {0}; float* o = (float*)malloc(sizeof(float) * BR_OUT * 96 * 8); double* f = (double*)malloc(sizeof(double) * 96);
    double t0 = br_now(); int bi = 0;
    for (int gnr = 0; gnr < 60 && br_now() - t0 < 8; gnr++) {
      if (gnr % 5 == 0) scen_make(&sc8, 8, &r, &c, &Q);
      cma_ask(&cm, &r); eval_set(cpu, cm.X, 96, &sc8, &Q, o);
      for (int i = 0; i < 96; i++) { double s2 = 0; for (int j = 0; j < 8; j++) s2 += fitness(o + ((size_t)i * 8 + j) * BR_OUT, 0, c.mode, sc8.sc[j].atkHit, c.altOn, tag_r(&c)); f[i] = s2 / 8; }
      bi = 0; for (int i = 1; i < 96; i++) if (f[i] > f[bi]) bi = i;
      memcpy(base, cm.X + (size_t)bi * P.nw, sizeof(float) * P.nw);
      cma_tell(&cm, f);
    }
    cma_free(&cm); free(o); free(f); scen_free(&sc8); cpu->destroy(cpu);
  }
  // candidate compute options
  char ids[40][32], labels[40][160]; int nOpt = 0;
  strcpy(ids[nOpt], "cpu"); snprintf(labels[nOpt++], 160, "CPU (%d threads)", br_cpu_count());
#ifdef BR_HAVE_METAL
  strcpy(ids[nOpt], "metal"); snprintf(labels[nOpt++], 160, "Metal");
#endif
#ifdef BR_HAVE_OPENCL
  { char nm[32][160]; int g[32]; int k = opencl_devices(nm, g, 32); for (int i = 0; i < k && nOpt < 40; i++) if (g[i]) { snprintf(ids[nOpt], 32, "opencl:%d", i); snprintf(labels[nOpt++], 160, "OpenCL, %s", nm[i]); } }
#endif
  int useN = c.cma ? 512 : 256;   // one CMA-ES gains little beyond ~512 networks per generation (GA: ~256)
  ScenSet scs = {0}; double tg0 = br_now(); scen_make(&scs, P.S, &r, &c, &P);
  g_tuneGenSec = (br_now() - tg0) / (c.reuse > 1 ? c.reuse : 1);
  if (c.mode == BR_MODE_TAG) tlog("Building %d tag scenarios takes %.0f ms on the CPU (counted in every timing below)", P.S, (br_now() - tg0) * 1000);
  const ScenSet* sc = &scs;
  Meas rows[200]; int nRows = 0; Meas bestOpt[40]; int nBest = 0;
  for (int oi = 0; oi < nOpt; oi++) {
    char err[256] = ""; TrainCfg cc = c; Backend* b = make_backend(ids[oi], &cc, err, sizeof err);
    if (!b) { tlog("%s: not usable (%s)", labels[oi], err); continue; }
    snprintf(labels[oi], 160, "%s", b->info);
    int gpu = strcmp(ids[oi], "cpu") != 0;
    double mem = b->memBytes > 0 ? b->memBytes : 2e9;
    int nMaxMem = (int)fmin(262144.0, 0.3 * mem / (4.0 * (P.nw + (double)P.S * P.stride)));
    int n0 = gpu ? (8192 / P.S > 64 ? 8192 / P.S : 64) : 64;   // GPUs: start near 8k flights per batch
    if (n0 > nMaxMem) n0 = nMaxMem;
    float* G = (float*)malloc(sizeof(float) * (size_t)nMaxMem * P.nw);
    if (!G) { tlog("%s: not enough memory to test", b->info); b->destroy(b); continue; }
    int filled = 0;
    #define FILL(N) for (; filled < (N); filled++) for (int j = 0; j < P.nw; j++) G[(size_t)filled * P.nw + j] = base[j] + (filled ? (float)(gauss(&r) * 0.02) : 0)
    float* out = (float*)malloc(sizeof(float) * BR_OUT * (size_t)nMaxMem * P.S);
    FILL(n0);
    int bestWg = b->wg; double bestChunk = b->chunkMs, sec = 0, sps;
    if (gpu) {
      // 1) work-group size
      double top = 0; int wgs[5] = { 32, 64, 128, 256, 512 };
      for (int w = 0; w < 5; w++) { if (wgs[w] > b->maxWg) continue; b->wg = wgs[w]; b->chunkMs = 40;
        tlog("%s: testing work-group size %d…", b->info, wgs[w]);
        sps = measure(b, G, n0, sc, &P, out, &sec);
        if (nRows < 190) { rows[nRows++] = (Meas){ "", "", n0, wgs[w], 40, sps, sec }; strcpy(rows[nRows - 1].id, ids[oi]); strcpy(rows[nRows - 1].label, b->info); }
        if (sps > top * 1.02) { top = sps; bestWg = wgs[w]; } }
      b->wg = bestWg;
      // 2) dispatch length
      double chunks[3] = { 20, 40, 80 }; top = 0;
      for (int q = 0; q < 3; q++) { b->chunkMs = chunks[q]; tlog("%s: testing %g ms per dispatch…", b->info, chunks[q]);
        sps = measure(b, G, n0, sc, &P, out, &sec);
        if (nRows < 190) { rows[nRows++] = (Meas){ "", "", n0, bestWg, chunks[q], sps, sec }; strcpy(rows[nRows - 1].id, ids[oi]); strcpy(rows[nRows - 1].label, b->info); }
        if (sps > top * 1.02) { top = sps; bestChunk = chunks[q]; } }
      b->chunkMs = bestChunk;
    }
    // 3) batch size: always reach the useful population, then grow until throughput stops improving, a batch takes
    //    > 2.5 s, or memory runs out
    double topS = 0, useSec = 0; int flat = 0; Meas sweep[24]; int ns = 0;
    for (int n = gpu ? 64 : 32; n <= nMaxMem && ns < 24; n *= 2) {
      FILL(n);
      tlog("%s: testing %d networks × %d scenarios = %d flights per batch…", b->info, n, P.S, n * P.S);
      sps = measure(b, G, n, sc, &P, out, &sec);
      sweep[ns] = (Meas){ "", "", n, b->wg, b->chunkMs, sps, sec }; strcpy(sweep[ns].id, ids[oi]); strcpy(sweep[ns].label, b->info); ns++;
      if (nRows < 190) rows[nRows++] = sweep[ns - 1];
      if (n == useN) useSec = sec;
      if (sps > topS * 1.03) { topS = sps; flat = 0; } else flat++;
      if (n >= useN && (flat >= 2 || sec > 2.5)) break;
    }
    if (useSec <= 0) { tlog("%s: could not run %d networks per batch", b->info, useN); free(G); free(out); b->destroy(b); continue; }
    // headroom: the biggest batch that costs at most 25% more time than the useful population (GPUs often run
    // several times more flights in nearly the same time; CPUs scale linearly and get none)
    int fillN = useN; for (int i = 0; i < ns; i++) if (sweep[i].n > fillN && sweep[i].sec <= 1.25 * useSec) fillN = sweep[i].n;
    bestOpt[nBest] = (Meas){ "", "", fillN, b->wg, b->chunkMs, topS, useSec }; strcpy(bestOpt[nBest].id, ids[oi]); strcpy(bestOpt[nBest].label, b->info); nBest++;
    tlog("%s: %.3f s per generation at %d networks (%.1f gens/s); peak %.1f M steps/s", b->info, useSec, useN, 1 / useSec, topS / 1e6);
    free(G); free(out); b->destroy(b);
    #undef FILL
  }
  // choose the option with the most generations per second at the useful population, then spend its free headroom
  // on more scenarios (steadier scores) or, with islands on, on more islands. Raw steps/s is not the goal: a GPU that
  // is slightly faster only at 8000+ networks per batch trains slower than a CPU running many small generations.
  Sb res = {0};
  if (!nBest) sb_printf(&res, "{\"ok\":false}");
  else {
    int w = 0; for (int i = 1; i < nBest; i++) if (bestOpt[i].sec < bestOpt[w].sec) w = i;
    Meas* m = &bestOpt[w];
    int pop = useN, scen = c.scen, isl = c.islands, nIsl = c.nIsl; double extra = (double)m->n / useN; char note[400] = "";
    if (c.islands) {
      int per = c.cma ? 48 : 32;   // CMA-ES groups of ~48, GA groups of ~32: big enough to learn, small enough to keep many going
      pop = m->n; nIsl = (int)lround((double)pop / per); if (nIsl < 2) nIsl = 2; if (nIsl > 256) nIsl = 256;
      pop = nIsl * (pop / nIsl > 4 ? pop / nIsl : 4);
      snprintf(note, sizeof note, "Islands on: %d islands of about %d networks fill the spare capacity.", nIsl, pop / nIsl);
    } else {
      scen = (int)fmin(256, lround(c.scen * extra));
      snprintf(note, sizeof note, "Picked for generations per second: %d networks at %.1f generations/s%s.", pop, 1 / m->sec,
        extra > 1 ? ", with spare capacity used for more scenarios per network (steadier scores)" : "");
    }
    sb_printf(&res, "{\"ok\":true,\"best\":{\"backend\":\"%s\",\"label\":", m->id); sb_jstr(&res, m->label);
    sb_printf(&res, ",\"wg\":%d,\"chunkMs\":%g,\"stepsPerS\":%.0f,\"secPerGen\":%.4f,\"batchNetworks\":%d,\"pop\":%d,\"scen\":%d,\"islands\":%d,\"nIsl\":%d},\"note\":",
      m->wg, m->chunk, m->sps, m->sec, m->n, pop, scen, isl, nIsl);
    sb_jstr(&res, note); sb_printf(&res, ",\"options\":[");
    for (int i = 0; i < nBest; i++) { sb_printf(&res, "%s{\"backend\":\"%s\",\"label\":", i ? "," : "", bestOpt[i].id); sb_jstr(&res, bestOpt[i].label);
      sb_printf(&res, ",\"stepsPerS\":%.0f,\"secPerGen\":%.4f,\"batchNetworks\":%d,\"wg\":%d,\"chunkMs\":%g}", bestOpt[i].sps, bestOpt[i].sec, bestOpt[i].n, bestOpt[i].wg, bestOpt[i].chunk); }
    sb_printf(&res, "],\"rows\":[");
    for (int i = 0; i < nRows; i++) { sb_printf(&res, "%s{\"backend\":\"%s\",\"label\":", i ? "," : "", rows[i].id); sb_jstr(&res, rows[i].label);
      sb_printf(&res, ",\"networks\":%d,\"flights\":%d,\"wg\":%d,\"chunkMs\":%g,\"stepsPerS\":%.0f,\"sec\":%.3f}", rows[i].n, rows[i].n * P.S, rows[i].wg, rows[i].chunk, rows[i].sps, rows[i].sec); }
    sb_printf(&res, "]}");
    tlog("Done: %s runs the most generations per second (%.1f).", m->label, 1 / m->sec);
  }
  br_lock(&T.mx); free(T.tuneResult.s); T.tuneResult = res; br_unlock(&T.mx);
  free(base); scen_free(&scs);
  T.tuning = 0;
}

void autotune_start(const TrainCfg* c) {
  if (T.tuning) return;
  trainer_pause();
  br_lock(&T.mx); free(T.tuneLog.s); T.tuneLog = (Sb){0}; free(T.tuneResult.s); T.tuneResult = (Sb){0}; br_unlock(&T.mx);
  T.tuneCfg = *c; T.tuning = 1;
  if (T.tuneLive) br_thread_join(&T.tuneTh);
  T.tuneLive = 1;
  br_thread_start(&T.tuneTh, tune_thread, NULL);
}
void autotune_status_json(Sb* o) {
  br_lock(&T.mx);
  sb_printf(o, "{\"running\":%s,\"log\":[%s],\"result\":%s}", T.tuning ? "true" : "false", T.tuneLog.s ? T.tuneLog.s : "", T.tuneResult.n ? T.tuneResult.s : "null");
  br_unlock(&T.mx);
}

// ---------------- flight playback: fly the star ball natively and return the paths for the 3D view
void fly_json(const char* req, Sb* o) {
  int count = (int)jnum(req, "count", 3); if (count < 1) count = 1; if (count > 12) count = 12;
  double dist = jnum(req, "dist", 0);
  br_lock(&T.mx);
  TrainCfg c; cfg_defaults(&c); cfg_from_json(&c, req);   // the UI sends its current settings
  if (!T.star || T.starMode != c.mode) { br_unlock(&T.mx); sb_printf(o, "{\"error\":\"no trained network for this mode yet\"}"); return; }
  c.dt = T.starDt; c.tw = T.starTw; c.everyStep = T.starEvery;   // fly it with the physics it was trained with
  BrParams P; setup_params(&P, &c, 1);
  int ok = P.nw == T.starNw && T.starNl == P.nl && T.starK == P.K && T.starMem == P.rec;
  if (ok) for (int l = 0; l <= P.nl; l++) if (T.starArch[l] != P.arch[l]) ok = 0;
  if (!ok) {   // the saved star has another shape: fly it with its own shape
    c.layers = T.starNl - 1; c.width = T.starArch[1]; c.K = T.starK; c.mem = T.starMem; setup_params(&P, &c, 1);
  }
  float* g = (float*)malloc(sizeof(float) * P.nw); memcpy(g, T.star, sizeof(float) * P.nw);
  br_unlock(&T.mx);
  float* wt = (float*)malloc(sizeof(float) * P.nw); br_transpose_genome(g, wt, &P);
  float* st = (float*)malloc(sizeof(float) * P.stride);
  static unsigned flyCount = 0;   // (unsigned) of the raw clock overflows to a constant: wrap it, and count presses
  Rng r; rng_seed(&r, (unsigned)fmod(br_now() * 1e6, 4294967296.0) ^ (++flyCount * 2654435761u));
  int every = (int)fmax(1, lround(0.05 / c.dt));   // ~20 samples per second
  sb_printf(o, "{\"dt\":%g,\"mode\":\"%s\",\"radar\":%g,\"flights\":[", every * c.dt, mode_name(c.mode), c.mode == BR_MODE_TAG ? c.detect : 0);
  if (c.mode == BR_MODE_TAG) {
    // Tag demo: one runner path (as in training), the star chasing it. Each sample is taken at the same physics step
    // for both balls: runner [x,y,z], chaser [x,y,z,throttle]. The runner keeps flying after the chaser ends
    // (unless it was tagged), so a miss shows it getting through.
    int maxS = tag_max_steps(&P); float* tr = (float*)malloc(sizeof(float) * 6 * (size_t)maxS);
    for (int f = 0; f < count; f++) {
      BrScen sc; gen_tag_scen(&sc, &r, &c, &P, tr, 0);
      int L = (int)sc.trajLen;
      br_init(st, &P, &sc);
      const float* rs = tr;
      sb_printf(o, "%s{\"target\":[%.1f,%.1f,%.1f],\"wind\":[%.2f,0,%.2f],\"runnerStart\":[%.1f,%.1f,%.1f],\"start\":[%.1f,%.1f,%.1f],\"points\":[[%.2f,%.2f,%.2f,0]",
        f ? "," : "", sc.tx, sc.ty, sc.tz, sc.wx, sc.wz, rs[0], rs[1], rs[2], sc.sx, sc.sy, sc.sz, sc.sx, sc.sy, sc.sz);
      int k = 0;
      while (st[S_ALIVE] != 0) {
        br_run(st, &P, &sc, wt, tr, every); k = (int)st[S_K];
        sb_printf(o, ",[%.2f,%.2f,%.2f,%.2f]", st[S_PX], st[S_PY], st[S_PZ], st[S_U0]);
      }
      int hit = st[S_HIT] != 0, escaped = !hit && k + 1 >= L;
      // runner samples: the same steps as the chaser's, then on to the end of its path unless it was tagged
      int endK = hit ? k : L - 1;
      sb_printf(o, "],\"runner\":[");
      for (int j = 0, n = 0; ; j += every, n++) { if (j > endK) j = endK; const float* a = tr + (size_t)j * 6;
        sb_printf(o, "%s[%.2f,%.2f,%.2f]", n ? "," : "", a[0], a[1], a[2]); if (j == endK) break; }
      sb_printf(o, "],\"hit\":%s,\"escaped\":%s,\"runnerHit\":%s,\"t\":%.2f,\"closest\":%.1f}",
        hit ? "true" : "false", escaped ? "true" : "false", sc.atkHit > 0.5f && !hit ? "true" : "false", st[S_T], st[S_MIND]);
    }
    free(tr); sb_printf(o, "]}"); free(g); free(wt); free(st); return;
  }
  // each press launches from a new spot on the range (the ball only sees the goal relative to itself)
  double oa = urand(&r) * 2 * M_PI, od = urand(&r) * 15000; float ox = (float)(cos(oa) * od), oz = (float)(sin(oa) * od);
  for (int f = 0; f < count; f++) {
    BrScen sc; gen_scen(&sc, &r, dist, c.reachMax); sc.sx += ox; sc.sz += oz; sc.tx += ox; sc.tz += oz;
    br_init(st, &P, &sc);
    sb_printf(o, "%s{\"target\":[%.1f,%.1f,%.1f],\"wind\":[%.2f,0,%.2f],\"points\":[[%.2f,%.2f,%.2f,0]", f ? "," : "", sc.tx, sc.ty, sc.tz, sc.wx, sc.wz, sc.sx, sc.sy, sc.sz);
    while (st[S_ALIVE] != 0) {
      br_run(st, &P, &sc, wt, NULL, every);   // reach demo; the tag demo is added with the UI
      sb_printf(o, ",[%.2f,%.2f,%.2f,%.2f]", st[S_PX], st[S_PY], st[S_PZ], st[S_U0]);
    }
    sb_printf(o, "],\"hit\":%s,\"t\":%.2f,\"closest\":%.1f}", st[S_HIT] != 0 ? "true" : "false", st[S_T], st[S_MIND]);
  }
  sb_printf(o, "]}");
  free(g); free(wt); free(st);
}

// ---------------- swarm playback: fly one battle natively and return every ball's path and the catches and leaks
// A side flown by a network uses its saved star (or the one training right now); physics come from the defenders'
// star when they are AI, else the attackers', else the panel.
static float* sw_star_for(const TrainCfg* c, const BrParams* P, int side) {
  SwSide* S = &SW.side[side]; float* g = NULL;
  br_lock(&T.mx);
  if (S->ai && S->star && S->nw == (side ? P->defNw : P->attNw) && S->nin == (side ? P->defNin : P->attNin)) { g = (float*)malloc(sizeof(float) * S->nw); memcpy(g, S->star, sizeof(float) * S->nw); }
  br_unlock(&T.mx);
  return g ? g : sw_load(c, P, side);
}
static void sw_star_physics(const TrainCfg* c, int side, TrainCfg* out) {
  FILE* f = fopen(sw_save_path(c, side), "rb"); if (!f) return;
  char b[512]; size_t n = fread(b, 1, sizeof b - 1, f); b[n] = 0; fclose(f);   // the physics fields come before the weights
  out->dt = jnum(b, "dt", out->dt); out->tw = jnum(b, "thrustToWeight", out->tw); out->everyStep = jnum(b, "ctrlEvery", out->everyStep ? 1 : 2) == 1;
  out->twRunner = jnum(b, "runnerThrustToWeight", out->twRunner);
}
void battle_json(const char* req, Sb* o) {
  TrainCfg c; cfg_defaults(&c); cfg_from_json(&c, req); c.mode = BR_MODE_SWARM;
  if (c.defAI) sw_star_physics(&c, 1, &c); else if (c.attAI) sw_star_physics(&c, 0, &c);
  BrParams P; setup_params(&P, &c, 1);
  float *wA = NULL, *wD = NULL;
  if (c.attAI && !(wA = sw_star_for(&c, &P, 0))) { sb_printf(o, "{\"error\":\"no trained attackers for this matchup yet\"}"); return; }
  if (c.defAI && !(wD = sw_star_for(&c, &P, 1))) { free(wA); sb_printf(o, "{\"error\":\"no trained defenders for this matchup yet\"}"); return; }
  int aA[BR_MAXL + 1], aD[BR_MAXL + 1]; sw_arch(&P, 0, aA); sw_arch(&P, 1, aD);
  float* tA = NULL; float* tD = NULL;
  if (wA) { tA = (float*)malloc(sizeof(float) * P.attNw); br_transpose_layers(wA, tA, P.nl, aA); }
  if (wD) { tD = (float*)malloc(sizeof(float) * P.defNw); br_transpose_layers(wD, tD, P.nl, aD); }
  static unsigned count = 0; Rng r; rng_seed(&r, (unsigned)fmod(br_now() * 1e6, 4294967296.0) ^ (++count * 2654435761u));
  int nA = c.attN, nB = nA + c.defN;
  float* st = (float*)malloc(sizeof(float) * SW_START * nB); BrScen sc; sw_gen_scen(&sc, &r, &c, st, 0);
  float* g = (float*)malloc(sizeof(float) * br_sw_stride(&P)); br_sw_init(g, &P, &sc, st);
  int every = (int)fmax(1, lround(0.05 / c.dt));
  // per ball: sampled points [x, y, z, throttle] while it flies; events as they happen
  Sb* pts = (Sb*)calloc(nB, sizeof(Sb)); Sb ev = {0}; int* fate = (int*)calloc(nB, sizeof(int)); int nev = 0;
  for (int b = 0; b < nB; b++) { const float* q = g + SW_H + b * SW_B; sb_printf(&pts[b], "[%.1f,%.1f,%.1f,0]", q[S_PX], q[S_PY], q[S_PZ]); }
  for (int guard = 0; g[SWH_DONE] == 0 && guard < 1000000; guard++) {
    br_sw_run(g, &P, &sc, tA, tD, every);
    float t = g[SWH_K] * P.dt;
    for (int b = 0; b < nB; b++) {
      const float* q = g + SW_H + b * SW_B; int f = (int)q[SB_FATE];
      if (q[S_ALIVE] != 0 || f != fate[b]) sb_printf(&pts[b], ",[%.1f,%.1f,%.1f,%.2f]", q[S_PX], q[S_PY], q[S_PZ], q[S_U0]);
      if (f != fate[b]) {
        const char* type = b < nA ? (f == 1 ? "leak" : f == 2 ? "caught" : "crash") : (f == 1 ? "catch" : "crash");
        if (!(b < nA && f == 2))   // a catch is reported once, by the defender
          sb_printf(&ev, "%s{\"t\":%.2f,\"type\":\"%s\",\"ball\":%d,\"pos\":[%.1f,%.1f,%.1f]}", nev++ ? "," : "", t, type, b, q[S_PX], q[S_PY], q[S_PZ]);
        fate[b] = f;
      }
    }
  }
  sb_printf(o, "{\"dt\":%g,\"mode\":\"swarm\",\"attN\":%d,\"defN\":%d,\"target\":[%.1f,%.1f,%.1f],\"radar\":%g,\"catchR\":%g,\"catches\":%d,\"leaks\":%d,\"t\":%.2f,\"balls\":[",
    every * c.dt, nA, c.defN, sc.tx, sc.ty, sc.tz, c.detect, P.ballD, (int)g[SWH_CATCH], (int)g[SWH_LEAK], g[SWH_K] * P.dt);
  for (int b = 0; b < nB; b++) { sb_printf(o, "%s{\"side\":\"%s\",\"fate\":%d,\"points\":[%s]}", b ? "," : "", b < nA ? "att" : "def", fate[b], pts[b].s ? pts[b].s : ""); free(pts[b].s); }
  sb_printf(o, "],\"events\":[%s]}", ev.s ? ev.s : "");
  free(ev.s); free(pts); free(fate); free(st); free(g); free(wA); free(wD); free(tA); free(tD);
}

// ---------------- developer command line (testing only; the app itself is used through the interface)
static void cli_cfg(TrainCfg* c, int argc, char** argv, int* mode, int* gens, char** load) {
  for (int i = 1; i < argc; i++) {
    const char* a = argv[i]; const char* v = i + 1 < argc ? argv[i + 1] : "";
    if (!strcmp(a, "--bench")) *mode = 1; else if (!strcmp(a, "--compare")) *mode = 2; else if (!strcmp(a, "--train")) *mode = 3;
    else if (!strcmp(a, "--list-devices")) *mode = 4; else if (!strcmp(a, "--help")) *mode = 5;
    else if (!strcmp(a, "--every-step")) c->everyStep = 1;
    else if (!strcmp(a, "--alt-reward")) c->altOn = 1;
    else if (!strcmp(a, "--islands")) { c->islands = 1; c->nIsl = atoi(v); i++; }
    else if (!strcmp(a, "--backend")) { snprintf(c->backend, sizeof c->backend, "%s", v); i++; }
    else if (!strcmp(a, "--pop")) { c->pop = atoi(v); i++; } else if (!strcmp(a, "--scen")) { c->scen = atoi(v); i++; }
    else if (!strcmp(a, "--layers")) { c->layers = atoi(v); i++; } else if (!strcmp(a, "--width")) { c->width = atoi(v); i++; }
    else if (!strcmp(a, "--K")) { c->K = atoi(v); i++; } else if (!strcmp(a, "--mem")) { c->mem = atoi(v); i++; }
    else if (!strcmp(a, "--opt")) { c->cma = strcmp(v, "ga") != 0; c->optKind = !strcmp(v, "sep") ? 1 : !strcmp(v, "lm") ? 2 : 0; i++; } else if (!strcmp(a, "--dt")) { c->dt = atof(v); i++; }
    else if (!strcmp(a, "--gens")) { *gens = atoi(v); i++; } else if (!strcmp(a, "--load")) { *load = (char*)v; i++; }
    else if (!strcmp(a, "--wg")) { c->wg = atoi(v); i++; } else if (!strcmp(a, "--threads")) { c->threads = atoi(v); i++; }
    // intercept (tag) mode
    else if (!strcmp(a, "--mode")) { c->mode = !strcmp(v, "tag") || !strcmp(v, "intercept") ? BR_MODE_TAG : !strcmp(v, "swarm") ? BR_MODE_SWARM : BR_MODE_REACH; i++; }
    else if (!strcmp(a, "--ceiling")) *mode = 6;
    else if (!strcmp(a, "--tw")) { c->tw = atof(v); i++; }
    else if (!strcmp(a, "--attackers")) { c->attN = clampi(atoi(v), 1, BR_SW_MAXA); i++; } else if (!strcmp(a, "--defenders")) { c->defN = clampi(atoi(v), 1, BR_SW_MAXD); i++; }
    else if (!strcmp(a, "--att")) { c->attAI = !strcmp(v, "ai"); i++; } else if (!strcmp(a, "--def")) { c->defAI = !strcmp(v, "ai"); i++; }
    else if (!strcmp(a, "--att-brain")) { c->attCmd = !strcmp(v, "cmd"); i++; } else if (!strcmp(a, "--def-brain")) { c->defCmd = !strcmp(v, "cmd"); i++; }
    else if (!strcmp(a, "--k")) { c->swK = clampi(atoi(v), 1, BR_SW_MAXK); i++; }
    else if (!strcmp(a, "--blast")) { c->blast = fmin(10, fmax(0, atof(v))); i++; }
    else if (!strcmp(a, "--reach-max")) { c->reachMax = fmin(100000, fmax(2000, atof(v))); i++; }
    else if (!strcmp(a, "--val-every")) { c->valEvery = atoi(v) > 0 ? atoi(v) : 10; i++; }
    else if (!strcmp(a, "--no-early")) c->noEarly = 1; else if (!strcmp(a, "--no-restarts")) c->restarts = 0; else if (!strcmp(a, "--no-norm")) c->normIn = 0; else if (!strcmp(a, "--no-imitate")) c->imitate = 0; else if (!strcmp(a, "--auto-diff")) c->autoDiff = 1;
    else if (!strcmp(a, "--atk-range-max")) { c->rangeMax = fmin(100000, fmax(4000, atof(v))); i++; }
    else if (!strcmp(a, "--atk-range")) { c->range = fmin(100000, fmax(4000, atof(v))); i++; }
    else if (!strcmp(a, "--evade")) { c->evade = fmin(1, fmax(0, atof(v))); i++; }
    else if (!strcmp(a, "--noise")) { c->noise = fmax(0, atof(v)); i++; }
    else if (!strcmp(a, "--delay")) { c->delayMs = fmax(0, atof(v)); i++; }
    else if (!strcmp(a, "--detect")) { c->detect = fmin(100000, fmax(0, atof(v))); i++; }
    else if (!strcmp(a, "--tw-runner")) { c->twRunner = fmin(5, fmax(1.2, atof(v))); i++; }
  }
}
static char* read_file(const char* p) { FILE* f = fopen(p, "rb"); if (!f) return NULL; fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  char* b = (char*)malloc(n + 1); n = (long)fread(b, 1, n, f); b[n] = 0; fclose(f); return b; }

int cli_main(int argc, char** argv) {
  TrainCfg c; cfg_defaults(&c); int mode = 0, gens = 100; char* load = NULL;
  cli_cfg(&c, argc, argv, &mode, &gens, &load);
  if (mode == 5) { printf("Double-click the app to open the interface. Developer options: --bench, --compare, --train [--gens N], --list-devices,\n"
                          "with --backend cpu|metal|opencl:N --pop --scen --layers --width --K --mem --opt cma|ga --dt --every-step --alt-reward --islands N --wg --load FILE\n"
                          "Intercept (tag) mode: --mode reach|tag|swarm  --atk-range M (runner launch distance, 4000-100000 m)  --evade 0-1 (runner weave)\n"
                          "  --noise M (sensor noise sigma)  --delay MS (sensor delay)  --detect M (radar range: launch when the runner comes this close, 0 = launch at once)\n"
                          "  --tw-runner X (runner thrust-to-weight)  --blast M (catch radius, 0-10 m)\n"
                          "Swarm mode: --mode swarm  --attackers N --defenders M (1-32)  --atk-range MIN --atk-range-max MAX (launch distance)  --att ai|algo --def ai|algo  --att-brain nk|cmd --def-brain nk|cmd  --k K (nearest-K, 1-8)\n"
                          "  --ceiling --mode swarm: algorithm vs algorithm battles;  --compare --mode swarm: the same battles on every backend and GPU layout\n"); return 0; }
  if (mode == 4) {
#ifdef BR_HAVE_OPENCL
    opencl_list_devices();
#endif
    Sb h = {0}; hardware_json(&h); printf("%s\n", h.s); free(h.s); return 0; }
  trainer_init();
  trainer_set_mode(c.mode);   // that mode's saved star (each mode keeps its own)
  if (load) { char* j = read_file(load); char m[200]; if (!j || trainer_load_json(j, m, sizeof m)) { fprintf(stderr, "cannot load %s\n", load); return 1; } free(j); }
  if (mode == 3) {   // headless training through the same service the interface uses
    trainer_start(&c); int last = 0;
    while (T.running || T.hasPending) {
      br_sleep_ms(200);
      br_lock(&T.mx); int g = T.gen; double best = T.best, hits = T.starHits, val = T.valHit, sps = T.stepsPerS, gt = T.genTime; int vg = T.valGen; br_unlock(&T.mx);
      if (g != last && c.mode == BR_MODE_SWARM) { br_lock(&T.mx);
        printf("gen %5d  attackers %7.3f  defenders %7.3f  caught %3.0f%%  leaked %3.0f%%  validation vs algorithm: leak %3.0f%% catch %3.0f%%  %6.0f battles/s  %.3f s/gen  [%s]\n",
          g, SW.side[0].best, SW.side[1].best, SW.catchRate * 100, SW.leakRate * 100, SW.side[0].val * 100, SW.side[1].val * 100, SW.battlesPerS, gt, T.beInfo);
        br_unlock(&T.mx); last = g; fflush(stdout); }
      { static char lastMsg[256] = ""; br_lock(&T.mx); if (strcmp(lastMsg, T.msg) && strncmp(T.msg, "training", 8)) { snprintf(lastMsg, sizeof lastMsg, "%s", T.msg); printf("  [%s]\n", T.msg); } br_unlock(&T.mx); }
      if (g != last) { printf("gen %5d  best %7.3f  star hits %3.0f%%  validation %3.0f%% (gen %d)  %6.1f M steps/s  %.3f s/gen  [%s]\n", g, best, hits * 100, val * 100, vg, sps / 1e6, gt, T.beInfo); last = g; fflush(stdout); }
      if (g >= gens) break;
    }
    trainer_pause(); return 0;
  }
  if (mode == 6 && c.mode == BR_MODE_SWARM) {   // --ceiling --mode swarm: algorithm vs algorithm battles (a baseline)
    c.attAI = c.defAI = 0; int N = 400;
    BrParams P; setup_params(&P, &c, N); Rng r; rng_seed(&r, 777); ScenSet scs = {0}; scen_make(&scs, N, &r, &c, &P);
    int* bat = (int*)calloc((size_t)N * 3, sizeof(int)); for (int i = 0; i < N; i++) bat[i * 3 + 2] = i;
    float* out = (float*)malloc(sizeof(float) * BR_SWOUT * N);
    char err[256] = ""; Backend* be = make_backend(strcmp(c.backend, "auto") ? c.backend : "cpu", &c, err, sizeof err); if (!be) { fprintf(stderr, "%s\n", err); return 1; }
    double t0 = br_now(); be->evalBattles(be, NULL, NULL, bat, N, scs.sc, scs.traj, scs.trajFloats, &P, out); double t = br_now() - t0;
    double ca = 0, le = 0, bs = 0, ac = 0, dc = 0; for (int i = 0; i < N; i++) { ca += out[i * BR_SWOUT]; le += out[i * BR_SWOUT + 1]; bs += out[i * BR_SWOUT + 4]; ac += out[i * BR_SWOUT + 6]; dc += out[i * BR_SWOUT + 7]; }
    printf("swarm %d vs %d  range %.0f m  radar %.0f  blast %.1f  evade %.1f  noise %.0f  [%s]:  caught %.1f%%  leaked %.1f%%  attackers crashed %.1f%%  defenders crashed %.1f%%   %.0f battles/s  %.1f M ball-steps/s\n",
      c.attN, c.defN, c.range, c.detect, c.blast, c.evade, c.noise, be->info, 100 * ca / (N * c.attN), 100 * le / (N * c.attN), 100 * ac / (N * c.attN), 100 * dc / (N * c.defN), N / t, bs / t / 1e6);
    be->destroy(be); free(bat); free(out); scen_free(&scs); return 0;
  }
  if (mode == 6) {   // --ceiling: the guidance algorithm (perfect sensors) as chaser on tag setups: how often it could tag
    BrParams P; setup_params(&P, &c, 1); Rng r; rng_seed(&r, 777); int N = 400, maxS = tag_max_steps(&P);
    float* tr = (float*)malloc(sizeof(float) * 6 * (size_t)maxS); int b2 = 0, b5 = 0, b20 = 0, through = 0;
    for (int i = 0; i < N; i++) { BrScen sc; gen_tag_scen(&sc, &r, &c, &P, tr, 0); sc.trajOff = 0;
      float d = algo_tag_rollout(&P, &sc, tr); b2 += d <= 2; b5 += d <= 5; b20 += d <= 20; through += sc.atkHit > 0.5f; }
    printf("range %.0f m  evade %.1f  radar %.0f  tw %.1f  dt %g:  tag<=2m %.1f%%  <=5m %.1f%%  <=20m %.1f%%  (runner arrives unopposed %.0f%%)\n",
      c.range, c.evade, c.detect, c.tw, c.dt, 100.0*b2/N, 100.0*b5/N, 100.0*b20/N, 100.0*through/N);
    free(tr); return 0;
  }
  if ((mode == 1 || mode == 2) && c.mode == BR_MODE_SWARM) {   // --bench / --compare --mode swarm: the same battles on every compute option
    int S = 32, G = 32, N = S * G;   // G genomes per AI side, each on every scenario
    BrParams P; setup_params(&P, &c, S); Rng r; rng_seed(&r, 12345); ScenSet scs = {0}; scen_make(&scs, S, &r, &c, &P);
    float* wA = (float*)malloc(sizeof(float) * (size_t)G * (P.attNw ? P.attNw : 1)); float* wD = (float*)malloc(sizeof(float) * (size_t)G * (P.defNw ? P.defNw : 1));
    for (size_t i = 0; i < (size_t)G * P.attNw; i++) wA[i] = (float)((urand(&r) * 2 - 1) * 0.5);
    for (size_t i = 0; i < (size_t)G * P.defNw; i++) wD[i] = (float)((urand(&r) * 2 - 1) * 0.5);
    for (int side = 0; side < 2; side++) { float* st = (side ? c.defAI : c.attAI) ? sw_load(&c, &P, side) : NULL; if (!st) continue;   // trained brains make real catches
      int nw = side ? P.defNw : P.attNw; float* w = side ? wD : wA; for (int k = 0; k < G; k++) for (int i = 0; i < nw; i++) w[(size_t)k * nw + i] = st[i] + (k ? (float)(gauss(&r) * 0.01) : 0);
      printf("  (%s: the saved star, with small variations)\n", side ? "defenders" : "attackers"); free(st); }
    int* bat = (int*)malloc(sizeof(int) * 3 * N); for (int i = 0; i < N; i++) { bat[i * 3] = i / S; bat[i * 3 + 1] = (i / S + 7) % G; bat[i * 3 + 2] = i % S; }
    float* ref = (float*)malloc(sizeof(float) * BR_SWOUT * N); float* out = (float*)malloc(sizeof(float) * BR_SWOUT * N);
    const char* ids[5] = { "cpu", "metal", "metal", "opencl:0", "opencl:0" }; const int lay[5] = { 0, 1, 2, 1, 2 };
    for (int bi = 0; bi < 5; bi++) {
      char err[256] = ""; Backend* be = make_backend(ids[bi], &c, err, sizeof err); if (!be) { printf("  %-9s not available (%s)\n", ids[bi], err); continue; }
      be->swLayout = lay[bi]; if (lay[bi]) { char t[300]; snprintf(t, sizeof t, "%s, %s", be->info, lay[bi] == 2 ? "group per battle" : "thread per battle"); snprintf(be->info, sizeof be->info, "%s", t); }
      if (!be->evalBattles) { printf("  %-9s %-36s cannot fly battles\n", ids[bi], be->info); be->destroy(be); continue; }
      float* dst = bi == 0 ? ref : out; double best = 1e9;
      for (int k = 0; k < 2; k++) { double t0 = br_now(); if (be->evalBattles(be, wA, wD, bat, N, scs.sc, scs.traj, scs.trajFloats, &P, dst)) { best = -1; break; } double t = br_now() - t0; if (t < best) best = t; }
      if (best < 0) { printf("  %-9s %-36s failed\n", ids[bi], be->info); be->destroy(be); continue; }
      double ca = 0, le = 0, bs = 0; for (int i = 0; i < N; i++) { ca += dst[i * BR_SWOUT]; le += dst[i * BR_SWOUT + 1]; bs += dst[i * BR_SWOUT + 4]; }
      double fp = 0; for (int i = 0; i < N; i++) fp += dst[i * BR_SWOUT + 2] + dst[i * BR_SWOUT + 3] + dst[i * BR_SWOUT + 4];   // path fingerprint
      printf("  %-9s %-36s %7.0f battles/s  %6.1f M ball-steps/s  caught %4.0f  leaked %4.0f  paths %.4f", ids[bi], be->info, N / best, bs / best / 1e6, ca, le, fp);
      if (mode == 2 && bi > 0) { int same = 0; for (int i = 0; i < N; i++) same += dst[i * BR_SWOUT] == ref[i * BR_SWOUT] && dst[i * BR_SWOUT + 1] == ref[i * BR_SWOUT + 1];
        printf("  same catches/leaks as CPU: %d/%d", same, N); }
      printf("\n"); be->destroy(be);
    }
    free(wA); free(wD); free(bat); free(ref); free(out); scen_free(&scs); return 0;
  }
  // --bench / --compare: one batch on every compute option
  BrParams P; setup_params(&P, &c, c.scen); Rng r; rng_seed(&r, 12345);
  ScenSet scs = {0}; scen_make(&scs, c.scen, &r, &c, &P); const ScenSet* sc = &scs;
  float* G = (float*)malloc(sizeof(float) * (size_t)c.pop * P.nw);
  for (int i = 0; i < c.pop; i++) random_genome(G + (size_t)i * P.nw, &P, &r);
  if (T.star && star_matches(&P)) for (int i = 0; i < c.pop; i++) for (int j = 0; j < P.nw; j++) G[(size_t)i * P.nw + j] = T.star[j] + (i ? (float)(gauss(&r) * 0.02) : 0);
  size_t n = (size_t)c.pop * c.scen; float* ref = (float*)malloc(sizeof(float) * BR_OUT * n); float* out = (float*)malloc(sizeof(float) * BR_OUT * n);
  const char* ids[8] = { "cpu", "metal", "opencl:0" }; int nIds = 3;
  for (int b = 0; b < nIds; b++) {
    char err[256] = ""; Backend* be = make_backend(ids[b], &c, err, sizeof err); if (!be) { printf("  %-9s not available (%s)\n", ids[b], err); continue; }
    float* dst = b == 0 ? ref : out;
    eval_set(be, G, c.pop < 32 ? c.pop : 32, sc, &P, dst);
    double best = 1e9; for (int k = 0; k < 2; k++) { double t0 = br_now(); eval_set(be, G, c.pop, sc, &P, dst); double t = br_now() - t0; if (t < best) best = t; }
    double steps = 0, hits = 0; for (size_t k = 0; k < n; k++) { steps += dst[k * BR_OUT + 3]; hits += dst[k * BR_OUT + 1]; }
    double sumD = 0; for (size_t k = 0; k < n; k++) sumD += dst[k * BR_OUT + 0];   // regression fingerprint
    printf("  %-9s %-36s %7.1f M steps/s  %5.0f hits  steps %.0f  sum closest %.1f", ids[b], be->info, steps / best / 1e6, hits, steps, sumD);
    if (mode == 2 && b > 0) { int same = 0; for (size_t k = 0; k < n; k++) same += dst[k * BR_OUT + 1] == ref[k * BR_OUT + 1]; printf("  same hit/miss as CPU: %d/%lu", same, (unsigned long)n); }
    printf("\n"); be->destroy(be);
  }
  return 0;
}
