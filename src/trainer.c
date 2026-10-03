// Training service: background training with islands (CMA-ES or GA), hardware auto-tune, champion persistence and
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
  if (!strncmp(p, "true", 4)) return 1; if (!strncmp(p, "false", 5)) return 0;
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
void cfg_defaults(TrainCfg* c) {
  memset(c, 0, sizeof *c);
  strcpy(c->backend, "auto"); c->wg = 64; c->chunkMs = 40;
  c->layers = 1; c->width = 32; c->pop = 256; c->scen = 16; c->reuse = 5; c->nIsl = 4; c->migrate = 10;
  c->cma = 1; c->decayOn = 1; c->decay = 0.995; c->lrMin = 1e-4;
  c->dt = 0.02; c->tw = 2.5; c->valEvery = 10; c->valScen = 64;
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
  c->cma = jnum(j, "cma", c->cma) != 0; c->decayOn = jnum(j, "decayOn", c->decayOn) != 0; c->decay = jnum(j, "decay", c->decay); c->lrMin = jnum(j, "lrMin", c->lrMin);
  c->dt = jnum(j, "dt", c->dt); if (c->dt < 0.005) c->dt = 0.005; if (c->dt > 0.04) c->dt = 0.04;
  c->tw = jnum(j, "tw", c->tw); c->speedW = jnum(j, "speedW", c->speedW); c->everyStep = jnum(j, "everyStep", c->everyStep) != 0;
  c->valEvery = clampi((int)jnum(j, "valEvery", c->valEvery), 1, 1000); c->valScen = clampi((int)jnum(j, "valScen", c->valScen), 8, 1024);
}
void cfg_to_json(const TrainCfg* c, char* o, int len) {
  snprintf(o, len, "{\"backend\":\"%s\",\"wg\":%d,\"chunkMs\":%g,\"layers\":%d,\"width\":%d,\"K\":%d,\"mem\":%d,\"pop\":%d,\"scen\":%d,\"reuse\":%d,"
    "\"islands\":%d,\"nIsl\":%d,\"migrate\":%d,\"cma\":%d,\"decayOn\":%d,\"decay\":%g,\"lrMin\":%g,\"dt\":%g,\"tw\":%g,\"speedW\":%g,\"everyStep\":%d}",
    c->backend, c->wg, c->chunkMs, c->layers, c->width, c->K, c->mem, c->pop, c->scen, c->reuse, c->islands, c->nIsl, c->migrate,
    c->cma, c->decayOn, c->decay, c->lrMin, c->dt, c->tw, c->speedW, c->everyStep);
}

void setup_params(BrParams* P, const TrainCfg* c, int S) {
  memset(P, 0, sizeof *P);
  double mass = 105;
  P->mass = (float)mass; P->thrust = (float)(c->tw * mass * 9.81); P->Cd = 0.32f; P->A = 0.03f; P->CNa = 4.5f; P->SM = 0.35f; P->len = 3; P->I = 40;
  P->dt = (float)c->dt; P->maxT = 120; P->hitR = 30;
  P->ctrl = c->everyStep ? 1 : 2; P->K = c->K; P->rec = c->mem; P->memN = c->mem ? c->width : 0;
  P->nin = BR_NI * (c->K + 1) + P->memN;
  P->nl = c->layers + 1; P->arch[0] = P->nin;
  for (int l = 1; l <= c->layers; l++) P->arch[l] = c->width;
  P->arch[c->layers + 1] = 3;
  int nw = 0; for (int l = 0; l < P->nl; l++) nw += P->arch[l + 1] * P->arch[l] + P->arch[l + 1];
  P->nw = nw; P->S = S;
  P->stride = (S_HIST + BR_NI * c->K + P->memN + 3) & ~3;
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

// ---------------- scenarios and scoring (reach-target mode: ball starts 10 m up, pointing straight up)
static void gen_scen(BrScen* s, Rng* r, double dist) {
  double a = urand(r) * 2 * M_PI, d = dist > 0 ? dist : 1500 + urand(r) * 5500, wa = urand(r) * 2 * M_PI, w = urand(r) * 15;
  memset(s, 0, sizeof *s);
  s->tx = (float)(cos(a) * d); s->tz = (float)(sin(a) * d);
  s->wx = (float)(cos(wa) * w); s->wz = (float)(sin(wa) * w);
  s->sy = 10.0f; s->q0 = 0.0f;
}
static double fitness(const float* o, double speedW) {
  double f = -log(fmax(o[0], 30.0) / 30.0);
  if (o[1] > 0.5f) f += 5 + (speedW > 0 ? speedW * fmax(0, 1 - o[2] / 60.0) : -0.01 * o[2]);
  return f;
}
static void random_genome(float* g, const BrParams* P, Rng* r) {
  for (int i = 0; i < P->nw; i++) g[i] = (float)((urand(r) * 2 - 1) * 0.5);
  if (P->rec) { int ni = P->arch[0], h1 = P->arch[1], m0 = BR_NI * (P->K + 1);   // memory weights start small
    for (int j = 0; j < h1; j++) for (int i = m0; i < ni; i++) g[j * ni + i] *= 0.2f; }
}

// ---------------- optimizers (ports of the browser's GA and sep-CMA-ES)
typedef struct { int n, lam; float* pop; double sigma; } GA;
typedef struct { int n, lam, mu; double* w; double mueff, cs, ds, cc, c1, cmu, chiN, sigma; double *m, *C, *ps, *pc; float *Z, *Y, *X; int g; } CMA;
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
}
static void cma_ask(CMA* c, Rng* r) {
  int n = c->n;
  for (int k = 0; k < c->lam; k++) for (int i = 0; i < n; i++) {
    size_t o = (size_t)k * n + i; double z = gauss(r), y = sqrt(c->C[i]) * z;
    c->Z[o] = (float)z; c->Y[o] = (float)y; c->X[o] = (float)(c->m[i] + c->sigma * y);
  }
}
static void cma_tell(CMA* c, const double* f) {
  int n = c->n, mu = c->mu, *ord = rank(f, c->lam);
  double* ym = (double*)calloc(n, sizeof(double)); double* zm = (double*)calloc(n, sizeof(double));
  for (int k = 0; k < mu; k++) { const float* y = c->Y + (size_t)ord[k] * n; const float* z = c->Z + (size_t)ord[k] * n; double wk = c->w[k];
    for (int i = 0; i < n; i++) { ym[i] += wk * y[i]; zm[i] += wk * z[i]; } }
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
static void cma_free(CMA* c) { free(c->w); free(c->m); free(c->C); free(c->ps); free(c->pc); free(c->Z); free(c->Y); free(c->X); }

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
  Rng rng; BrScen* scen; int scenCount, scenAge;
  float* champ; int champNw; int champArch[BR_MAXL]; int champNl; int champK, champMem; int champVersion; double champFit;
  int gen; double best, mean, sigma, genTime, flightsPerS, stepsPerS, champHits, champMiss, valHit; int valGen;
  Hist* hist; int histN;
  char msg[256]; char beInfo[256];
  // auto-tune
  br_thread tuneTh; volatile int tuning, tuneLive; TrainCfg tuneCfg; Sb tuneLog; Sb tuneResult;
} T;

static void set_msg(const char* fmt, ...) { va_list ap; va_start(ap, fmt); br_lock(&T.mx); vsnprintf(T.msg, sizeof T.msg, fmt, ap); br_unlock(&T.mx); va_end(ap); }

static const char* save_path(void) {
  static char p[1024]; const char* h = getenv("HOME");
#ifdef _WIN32
  if (!h) h = getenv("USERPROFILE");
  snprintf(p, sizeof p, "%s\\BallisticRange-champion.json", h ? h : ".");
#else
  snprintf(p, sizeof p, "%s/BallisticRange-champion.json", h ? h : ".");
#endif
  return p;
}

static void champion_json_locked(Sb* o) {
  sb_printf(o, "{\"format\":\"ballistic-range-network\",\"mode\":\"mission\",\"arch\":[");
  for (int l = 0; l <= T.champNl; l++) sb_printf(o, "%s%d", l ? "," : "", T.champArch[l]);
  sb_printf(o, "],\"K\":%d,\"memory\":%s,\"dt\":%g,\"ctrlEvery\":%d,\"thrustToWeight\":%g,\"generation\":%d,\"fitness\":%.4f,\"validationHitRate\":%.4f,\"version\":%d,\"weights\":[",
    T.champK, T.champMem ? "true" : "false", T.cfg.dt, T.cfg.everyStep ? 1 : 2, T.cfg.tw, T.gen, T.champFit, T.valHit, T.champVersion);
  for (int i = 0; i < T.champNw; i++) sb_printf(o, "%s%.7g", i ? "," : "", T.champ[i]);
  sb_printf(o, "]}");
}
static void autosave(void) {
  Sb o = {0}; br_lock(&T.mx); if (T.champ) champion_json_locked(&o); br_unlock(&T.mx);
  if (o.n) { FILE* f = fopen(save_path(), "w"); if (f) { fwrite(o.s, 1, o.n, f); fclose(f); } }
  free(o.s);
}

// Load a network (JSON with "arch", "K", "memory", "weights"). Called with the lock NOT held.
int trainer_load_json(const char* json, char* msg, int len) {
  const char* a = strstr(json, "\"arch\""); const char* w = strstr(json, "\"weights\"");
  if (!a || !w) { snprintf(msg, len, "not a Ballistic Range network file"); return -1; }
  int arch[BR_MAXL], nl = -1; const char* p = strchr(a, '['); if (!p) return -1; p++;
  while (*p && *p != ']' && nl < BR_MAXL - 1) { arch[++nl] = (int)strtol(p, (char**)&p, 10); while (*p == ',' || *p == ' ') p++; }
  if (nl < 1) { snprintf(msg, len, "bad arch"); return -1; }
  int nw = 0; for (int l = 0; l < nl; l++) nw += arch[l + 1] * arch[l] + arch[l + 1];
  float* g = (float*)malloc(sizeof(float) * nw); p = strchr(w, '['); if (!p) { free(g); return -1; } p++;
  int n = 0; while (n < nw && *p && *p != ']') { g[n++] = strtof(p, (char**)&p); while (*p == ',' || *p == ' ') p++; }
  if (n != nw) { free(g); snprintf(msg, len, "expected %d weights, found %d", nw, n); return -1; }
  br_lock(&T.mx);
  free(T.champ); T.champ = g; T.champNw = nw; T.champNl = nl; memcpy(T.champArch, arch, sizeof(int) * (nl + 1));
  T.champK = (int)jnum(json, "K", 0); T.champMem = jnum(json, "memory", 0) != 0; T.champVersion++; T.champFit = jnum(json, "fitness", 0);
  T.valHit = jnum(json, "validationHitRate", 0);
  br_unlock(&T.mx);
  snprintf(msg, len, "loaded network %d", arch[0]); for (int l = 1; l <= nl; l++) { char t[16]; snprintf(t, sizeof t, "-%d", arch[l]); strncat(msg, t, len - strlen(msg) - 1); }
  return 0;
}

void trainer_init(void) {
  br_mutex_init(&T.mx); rng_seed(&T.rng, (unsigned)(br_now() * 1000));
  cfg_defaults(&T.cfg); T.hist = (Hist*)calloc(HIST_MAX, sizeof(Hist));
  FILE* f = fopen(save_path(), "rb");
  if (f) { fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET); char* b = (char*)malloc(len + 1); len = (long)fread(b, 1, len, f); b[len] = 0; fclose(f);
    char m[128]; if (!trainer_load_json(b, m, sizeof m)) snprintf(T.msg, sizeof T.msg, "restored your last champion (%s)", save_path()); free(b); }
}

static int champ_matches(const BrParams* P) {
  if (!T.champ || T.champNl != P->nl) return 0;
  for (int l = 0; l <= P->nl; l++) if (T.champArch[l] != P->arch[l]) return 0;
  return T.champK == P->K && T.champMem == P->rec;
}

static void free_islands(void) {
  for (int i = 0; i < T.nIsl; i++) { if (T.isl[i].cma) cma_free(&T.isl[i].c); else ga_free(&T.isl[i].ga); free(T.isl[i].bestG); }
  free(T.isl); T.isl = NULL; T.nIsl = 0;
}
static void build_islands(void) {
  free_islands();
  TrainCfg* c = &T.cfg; setup_params(&T.P, c, c->scen);
  int k = c->islands ? c->nIsl : 1; if (k > c->pop / 4) k = c->pop / 4 > 1 ? c->pop / 4 : 1;
  int per = c->pop / k; if (per < 4) per = 4;
  float* seed = (float*)malloc(sizeof(float) * T.P.nw); int seeded = 0;
  br_lock(&T.mx); if (champ_matches(&T.P)) { memcpy(seed, T.champ, sizeof(float) * T.P.nw); seeded = 1; } br_unlock(&T.mx);
  T.isl = (Island*)calloc(k, sizeof(Island)); T.nIsl = k;
  for (int i = 0; i < k; i++) {
    Island* I = &T.isl[i]; I->cma = c->cma; I->bestG = (float*)calloc(T.P.nw, sizeof(float)); I->bestF = -1e30;
    float* s = (float*)malloc(sizeof(float) * T.P.nw);
    if (seeded) memcpy(s, seed, sizeof(float) * T.P.nw); else { random_genome(s, &T.P, &T.rng); if (c->cma) for (int j = 0; j < T.P.nw; j++) s[j] *= 0.2f; }
    if (c->cma) cma_init(&I->c, T.P.nw, per, s, seeded ? 0.02 : 0.25); else ga_init(&I->ga, T.P.nw, per, s, seeded ? 0.03 : 0.1, &T.rng);
    free(s);
  }
  free(seed);
  if (!seeded) { br_lock(&T.mx); T.histN = 0; T.gen = 0; br_unlock(&T.mx); }
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
  if (T.be && !strcmp(T.beId, want) && T.beThreads == T.cfg.threads) { T.be->wg = T.cfg.wg; T.be->chunkMs = T.cfg.chunkMs; return 0; }
  if (T.be) { T.be->destroy(T.be); T.be = NULL; }
  char err[256] = "";
  T.be = make_backend(want, &T.cfg, err, sizeof err);
  if (!T.be) { set_msg("cannot start %s: %s", want, err); return -1; }
  snprintf(T.beId, sizeof T.beId, "%s", want); T.beThreads = T.cfg.threads;
  br_lock(&T.mx); snprintf(T.beInfo, sizeof T.beInfo, "%s", T.be->info); br_unlock(&T.mx);
  return 0;
}

static double validate(const float* g) {
  BrParams P; setup_params(&P, &T.cfg, T.cfg.valScen);
  BrScen* sc = (BrScen*)malloc(sizeof(BrScen) * P.S); for (int i = 0; i < P.S; i++) gen_scen(&sc[i], &T.rng, 0);
  float* out = (float*)malloc(sizeof(float) * 4 * P.S);
  int hits = 0; if (!T.be->eval(T.be, g, 1, sc, &P, out)) for (int i = 0; i < P.S; i++) hits += out[i * 4 + 1] > 0.5f;
  free(sc); free(out); return (double)hits / P.S;
}

static void train_thread(void* arg) {
  (void)arg;
  float* flat = NULL; size_t flatCap = 0; float* out = NULL; size_t outCap = 0; double* fit = NULL; int fitCap = 0;
  set_msg("training");
  while (!T.stopReq) {
    if (T.hasPending) { br_lock(&T.mx); T.cfg = T.pending; T.hasPending = 0; br_unlock(&T.mx); }
    if (ensure_backend()) break;
    if (needs_rebuild(&T.cfg)) build_islands();
    TrainCfg* c = &T.cfg;
    setup_params(&T.P, c, c->scen);   // physics settings may have changed
    if (T.scenCount != c->scen || T.scenAge >= c->reuse) {
      if (T.scenCount != c->scen) { free(T.scen); T.scen = (BrScen*)malloc(sizeof(BrScen) * c->scen); T.scenCount = c->scen; }
      for (int i = 0; i < c->scen; i++) gen_scen(&T.scen[i], &T.rng, 0);
      T.scenAge = 0;
    }
    T.scenAge++;
    int n = 0; for (int i = 0; i < T.nIsl; i++) { if (T.isl[i].cma) cma_ask(&T.isl[i].c, &T.rng); n += isl_size(&T.isl[i]); }
    size_t need = (size_t)n * T.P.nw; if (need > flatCap) { flat = (float*)realloc(flat, sizeof(float) * need); flatCap = need; }
    size_t needO = (size_t)n * c->scen * 4; if (needO > outCap) { out = (float*)realloc(out, sizeof(float) * needO); outCap = needO; }
    if (n > fitCap) { fit = (double*)realloc(fit, sizeof(double) * n); fitCap = n; }
    { size_t o = 0; for (int i = 0; i < T.nIsl; i++) { size_t k = (size_t)isl_size(&T.isl[i]) * T.P.nw; memcpy(flat + o, isl_genomes(&T.isl[i]), sizeof(float) * k); o += k; } }
    double t0 = br_now();
    if (T.be->eval(T.be, flat, n, T.scen, &T.P, out)) { set_msg("evaluation failed on %s", T.be->info); break; }
    double dt = br_now() - t0, steps = 0, mean = 0; int champ = 0;
    for (int i = 0; i < n; i++) { double f = 0; for (int j = 0; j < c->scen; j++) { const float* o = out + ((size_t)i * c->scen + j) * 4; f += fitness(o, c->speedW); steps += o[3]; }
      fit[i] = f / c->scen; mean += fit[i]; if (fit[i] > fit[champ]) champ = i; }
    mean /= n;
    int hits = 0; double miss = 0; for (int j = 0; j < c->scen; j++) { const float* o = out + ((size_t)champ * c->scen + j) * 4; hits += o[1] > 0.5f; miss += o[0]; }
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
    if (T.champNw != T.P.nw) { free(T.champ); T.champ = (float*)malloc(sizeof(float) * T.P.nw); T.champNw = T.P.nw; }
    memcpy(T.champ, flat + (size_t)champ * T.P.nw, sizeof(float) * T.P.nw);
    T.champNl = T.P.nl; memcpy(T.champArch, T.P.arch, sizeof(int) * (T.P.nl + 1)); T.champK = T.P.K; T.champMem = T.P.rec; T.champVersion++; T.champFit = fit[champ];
    T.best = fit[champ]; T.mean = mean; T.sigma = sig / T.nIsl; T.genTime = dt; T.flightsPerS = n * c->scen / dt; T.stepsPerS = steps / dt;
    T.champHits = (double)hits / c->scen; T.champMiss = miss / c->scen;
    if (T.histN == HIST_MAX) { memmove(T.hist, T.hist + 1, sizeof(Hist) * (HIST_MAX - 1)); T.histN--; }
    Hist* h = &T.hist[T.histN++]; h->gen = gen; h->best = (float)T.best; h->mean = (float)mean; memcpy(h->isl, islBest, sizeof islBest);
    br_unlock(&T.mx);
    // islands: pass each island's best to the next one in a ring
    if (T.nIsl > 1 && gen % c->migrate == 0)
      for (int i = 0; i < T.nIsl; i++) { Island* to = &T.isl[(i + 1) % T.nIsl]; const float* g = T.isl[i].bestG;
        if (to->cma) for (int j = 0; j < T.P.nw; j++) to->c.m[j] = 0.5 * (to->c.m[j] + g[j]);
        else memcpy(to->ga.pop + (size_t)(to->ga.lam - 1) * T.P.nw, g, sizeof(float) * T.P.nw); }
    if (gen % c->valEvery == 0) {
      float* g = (float*)malloc(sizeof(float) * T.P.nw); memcpy(g, flat + (size_t)champ * T.P.nw, sizeof(float) * T.P.nw);
      double v = validate(g); free(g);
      br_lock(&T.mx); T.valHit = v; T.valGen = gen; br_unlock(&T.mx);
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
  T.running = 0; autosave();
}
void trainer_reset(void) {
  trainer_pause();
  free_islands();
  br_lock(&T.mx);
  free(T.champ); T.champ = NULL; T.champNw = 0; T.champVersion++; T.gen = 0; T.histN = 0;
  T.best = T.mean = T.sigma = T.genTime = T.flightsPerS = T.stepsPerS = T.champHits = T.champMiss = T.valHit = 0; T.valGen = 0;
  snprintf(T.msg, sizeof T.msg, "reset · training stopped");
  br_unlock(&T.mx);
  remove(save_path());
}

void trainer_status_json(Sb* o, int since) {
  br_lock(&T.mx);
  sb_printf(o, "{\"running\":%s,\"tuning\":%s,\"gen\":%d,\"best\":%.4f,\"mean\":%.4f,\"champHits\":%.4f,\"champMiss\":%.2f,\"flightsPerS\":%.0f,\"stepsPerS\":%.0f,"
    "\"genTime\":%.4f,\"sigma\":%.3g,\"valHit\":%.4f,\"valGen\":%d,\"champVersion\":%d,\"hasChampion\":%s,\"islands\":%d,\"backend\":",
    T.running ? "true" : "false", T.tuning ? "true" : "false", T.gen, T.best, T.mean, T.champHits, T.champMiss, T.flightsPerS, T.stepsPerS,
    T.genTime, T.sigma, T.valHit, T.valGen, T.champVersion, T.champ ? "true" : "false", T.nIsl);
  sb_jstr(o, T.beInfo); sb_printf(o, ",\"message\":"); sb_jstr(o, T.msg); sb_printf(o, ",\"hist\":[");
  int first = 1;
  for (int i = 0; i < T.histN; i++) { Hist* h = &T.hist[i]; if (h->gen <= since) continue;
    sb_printf(o, "%s[%d,%.4f,%.4f", first ? "" : ",", h->gen, h->best, h->mean); first = 0;
    int k = T.nIsl > 1 ? (T.nIsl < ISL_SHOW ? T.nIsl : ISL_SHOW) : 0;
    for (int q = 0; q < k; q++) sb_printf(o, ",%.3f", h->isl[q]);
    sb_printf(o, "]"); }
  sb_printf(o, "]}");
  br_unlock(&T.mx);
}
void trainer_champion_json(Sb* o) { br_lock(&T.mx); if (T.champ) champion_json_locked(o); else sb_printf(o, "{}"); br_unlock(&T.mx); }

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

static double measure(Backend* b, const float* G, int n, const BrScen* sc, BrParams* P, float* out, double* secOut) {
  b->eval(b, G, n < 32 ? n : 32, sc, P, out);   // warm-up
  double best = 1e9, steps = 0;
  for (int r = 0; r < 2; r++) { double t0 = br_now(); if (b->eval(b, G, n, sc, P, out)) return 0; double t = br_now() - t0; if (t < best) best = t; }
  for (int k = 0; k < n * P->S; k++) steps += out[k * 4 + 3];
  if (secOut) *secOut = best;
  return steps / best;
}

static void tune_thread(void* arg) {
  (void)arg; TrainCfg c = T.tuneCfg; Rng r; rng_seed(&r, 777);
  BrParams P; setup_params(&P, &c, c.scen);
  // A realistic population: your champion (if it matches the current network) plus small variations. Random networks
  // crash within seconds and would mislead the measurement, so without a champion a quick CPU pre-training runs first.
  float* base = (float*)malloc(sizeof(float) * P.nw); int have = 0;
  br_lock(&T.mx); if (champ_matches(&P)) { memcpy(base, T.champ, sizeof(float) * P.nw); have = 1; } br_unlock(&T.mx);
  if (!have) {
    tlog("No trained network for this shape yet: pre-training a few seconds on the CPU so the test flights are realistic…");
    Backend* cpu = cpu_backend_create(0); BrParams Q; setup_params(&Q, &c, 8);
    CMA cm; float* s = (float*)malloc(sizeof(float) * P.nw); random_genome(s, &P, &r); for (int j = 0; j < P.nw; j++) s[j] *= 0.2f;
    cma_init(&cm, P.nw, 96, s, 0.25); free(s);
    BrScen sc8[8]; float* o = (float*)malloc(sizeof(float) * 4 * 96 * 8); double* f = (double*)malloc(sizeof(double) * 96);
    double t0 = br_now(); int bi = 0;
    for (int gnr = 0; gnr < 60 && br_now() - t0 < 8; gnr++) {
      if (gnr % 5 == 0) for (int i = 0; i < 8; i++) gen_scen(&sc8[i], &r, 0);
      cma_ask(&cm, &r); cpu->eval(cpu, cm.X, 96, sc8, &Q, o);
      for (int i = 0; i < 96; i++) { double s2 = 0; for (int j = 0; j < 8; j++) s2 += fitness(o + ((size_t)i * 8 + j) * 4, 0); f[i] = s2 / 8; }
      bi = 0; for (int i = 1; i < 96; i++) if (f[i] > f[bi]) bi = i;
      memcpy(base, cm.X + (size_t)bi * P.nw, sizeof(float) * P.nw);
      cma_tell(&cm, f);
    }
    cma_free(&cm); free(o); free(f); cpu->destroy(cpu);
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
  BrScen* sc = (BrScen*)malloc(sizeof(BrScen) * P.S); for (int i = 0; i < P.S; i++) gen_scen(&sc[i], &r, 0);
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
    float* out = (float*)malloc(sizeof(float) * 4 * (size_t)nMaxMem * P.S);
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
  free(base); free(sc);
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

// ---------------- flight playback: fly the champion natively and return the paths for the 3D view
void fly_json(const char* req, Sb* o) {
  int count = (int)jnum(req, "count", 3); if (count < 1) count = 1; if (count > 12) count = 12;
  double dist = jnum(req, "dist", 0);
  br_lock(&T.mx);
  if (!T.champ) { br_unlock(&T.mx); sb_printf(o, "{\"error\":\"no trained network yet\"}"); return; }
  TrainCfg c; cfg_defaults(&c); cfg_from_json(&c, req);   // the UI sends its current settings
  BrParams P; setup_params(&P, &c, 1);
  int ok = P.nw == T.champNw && T.champNl == P.nl && T.champK == P.K && T.champMem == P.rec;
  if (ok) for (int l = 0; l <= P.nl; l++) if (T.champArch[l] != P.arch[l]) ok = 0;
  if (!ok) {   // the saved champion has another shape: fly it with its own shape
    c.layers = T.champNl - 1; c.width = T.champArch[1]; c.K = T.champK; c.mem = T.champMem; setup_params(&P, &c, 1);
  }
  float* g = (float*)malloc(sizeof(float) * P.nw); memcpy(g, T.champ, sizeof(float) * P.nw);
  br_unlock(&T.mx);
  float* wt = (float*)malloc(sizeof(float) * P.nw); br_transpose_genome(g, wt, &P);
  float* st = (float*)malloc(sizeof(float) * P.stride);
  Rng r; rng_seed(&r, (unsigned)(br_now() * 1e6));
  int every = (int)fmax(1, lround(0.05 / c.dt));   // ~20 samples per second
  sb_printf(o, "{\"dt\":%g,\"flights\":[", every * c.dt);
  for (int f = 0; f < count; f++) {
    BrScen sc; gen_scen(&sc, &r, dist);
    br_init(st, &P, &sc);
    sb_printf(o, "%s{\"target\":[%.1f,%.1f,%.1f],\"wind\":[%.2f,0,%.2f],\"points\":[[%.2f,%.2f,%.2f,0]", f ? "," : "", sc.tx, sc.ty, sc.tz, sc.wx, sc.wz, sc.sx, sc.sy, sc.sz);
    while (st[S_ALIVE] != 0) {
      br_run(st, &P, &sc, wt, every);
      sb_printf(o, ",[%.2f,%.2f,%.2f,%.2f]", st[S_PX], st[S_PY], st[S_PZ], st[S_U0]);
    }
    sb_printf(o, "],\"hit\":%s,\"t\":%.2f,\"closest\":%.1f}", st[S_HIT] != 0 ? "true" : "false", st[S_T], st[S_MIND]);
  }
  sb_printf(o, "]}");
  free(g); free(wt); free(st);
}

// ---------------- developer command line (testing only; the app itself is used through the interface)
static void cli_cfg(TrainCfg* c, int argc, char** argv, int* mode, int* gens, char** load) {
  for (int i = 1; i < argc; i++) {
    const char* a = argv[i]; const char* v = i + 1 < argc ? argv[i + 1] : "";
    if (!strcmp(a, "--bench")) *mode = 1; else if (!strcmp(a, "--compare")) *mode = 2; else if (!strcmp(a, "--train")) *mode = 3;
    else if (!strcmp(a, "--list-devices")) *mode = 4; else if (!strcmp(a, "--help")) *mode = 5;
    else if (!strcmp(a, "--every-step")) c->everyStep = 1;
    else if (!strcmp(a, "--islands")) { c->islands = 1; c->nIsl = atoi(v); i++; }
    else if (!strcmp(a, "--backend")) { snprintf(c->backend, sizeof c->backend, "%s", v); i++; }
    else if (!strcmp(a, "--pop")) { c->pop = atoi(v); i++; } else if (!strcmp(a, "--scen")) { c->scen = atoi(v); i++; }
    else if (!strcmp(a, "--layers")) { c->layers = atoi(v); i++; } else if (!strcmp(a, "--width")) { c->width = atoi(v); i++; }
    else if (!strcmp(a, "--K")) { c->K = atoi(v); i++; } else if (!strcmp(a, "--mem")) { c->mem = atoi(v); i++; }
    else if (!strcmp(a, "--opt")) { c->cma = strcmp(v, "ga") != 0; i++; } else if (!strcmp(a, "--dt")) { c->dt = atof(v); i++; }
    else if (!strcmp(a, "--gens")) { *gens = atoi(v); i++; } else if (!strcmp(a, "--load")) { *load = (char*)v; i++; }
    else if (!strcmp(a, "--wg")) { c->wg = atoi(v); i++; } else if (!strcmp(a, "--threads")) { c->threads = atoi(v); i++; }
  }
}
static char* read_file(const char* p) { FILE* f = fopen(p, "rb"); if (!f) return NULL; fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  char* b = (char*)malloc(n + 1); n = (long)fread(b, 1, n, f); b[n] = 0; fclose(f); return b; }

int cli_main(int argc, char** argv) {
  TrainCfg c; cfg_defaults(&c); int mode = 0, gens = 100; char* load = NULL;
  cli_cfg(&c, argc, argv, &mode, &gens, &load);
  if (mode == 5) { printf("Double-click the app to open the interface. Developer options: --bench, --compare, --train [--gens N], --list-devices,\n"
                          "with --backend cpu|metal|opencl:N --pop --scen --layers --width --K --mem --opt cma|ga --dt --every-step --islands N --wg --load FILE\n"); return 0; }
  if (mode == 4) {
#ifdef BR_HAVE_OPENCL
    opencl_list_devices();
#endif
    Sb h = {0}; hardware_json(&h); printf("%s\n", h.s); free(h.s); return 0; }
  trainer_init();
  if (load) { char* j = read_file(load); char m[200]; if (!j || trainer_load_json(j, m, sizeof m)) { fprintf(stderr, "cannot load %s\n", load); return 1; } free(j); }
  if (mode == 3) {   // headless training through the same service the interface uses
    trainer_start(&c); int last = 0;
    while (T.running || T.hasPending) {
      br_sleep_ms(200);
      br_lock(&T.mx); int g = T.gen; double best = T.best, hits = T.champHits, val = T.valHit, sps = T.stepsPerS, gt = T.genTime; int vg = T.valGen; br_unlock(&T.mx);
      if (g != last) { printf("gen %5d  best %7.3f  champ hits %3.0f%%  validation %3.0f%% (gen %d)  %6.1f M steps/s  %.3f s/gen  [%s]\n", g, best, hits * 100, val * 100, vg, sps / 1e6, gt, T.beInfo); last = g; fflush(stdout); }
      if (g >= gens) break;
    }
    trainer_pause(); return 0;
  }
  // --bench / --compare: one batch on every compute option
  BrParams P; setup_params(&P, &c, c.scen); Rng r; rng_seed(&r, 12345);
  BrScen* sc = (BrScen*)malloc(sizeof(BrScen) * c.scen); for (int i = 0; i < c.scen; i++) gen_scen(&sc[i], &r, 0);
  float* G = (float*)malloc(sizeof(float) * (size_t)c.pop * P.nw);
  for (int i = 0; i < c.pop; i++) random_genome(G + (size_t)i * P.nw, &P, &r);
  if (T.champ && champ_matches(&P)) for (int i = 0; i < c.pop; i++) for (int j = 0; j < P.nw; j++) G[(size_t)i * P.nw + j] = T.champ[j] + (i ? (float)(gauss(&r) * 0.02) : 0);
  size_t n = (size_t)c.pop * c.scen; float* ref = (float*)malloc(sizeof(float) * 4 * n); float* out = (float*)malloc(sizeof(float) * 4 * n);
  const char* ids[8] = { "cpu", "metal", "opencl:0" }; int nIds = 3;
  for (int b = 0; b < nIds; b++) {
    char err[256] = ""; Backend* be = make_backend(ids[b], &c, err, sizeof err); if (!be) { printf("  %-9s not available (%s)\n", ids[b], err); continue; }
    float* dst = b == 0 ? ref : out;
    be->eval(be, G, c.pop < 32 ? c.pop : 32, sc, &P, dst);
    double best = 1e9; for (int k = 0; k < 2; k++) { double t0 = br_now(); be->eval(be, G, c.pop, sc, &P, dst); double t = br_now() - t0; if (t < best) best = t; }
    double steps = 0, hits = 0; for (size_t k = 0; k < n; k++) { steps += dst[k * 4 + 3]; hits += dst[k * 4 + 1]; }
    printf("  %-9s %-36s %7.1f M steps/s  %5.0f hits", ids[b], be->info, steps / best / 1e6, hits);
    if (mode == 2 && b > 0) { int same = 0; for (size_t k = 0; k < n; k++) same += dst[k * 4 + 1] == ref[k * 4 + 1]; printf("  same hit/miss as CPU: %d/%lu", same, (unsigned long)n); }
    printf("\n"); be->destroy(be);
  }
  return 0;
}
