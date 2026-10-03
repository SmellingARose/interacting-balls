// Ball Arena — native simulation core: reach-goal mode and tag mode.
// One source, three compilers: plain C (CPU backend), Metal Shading Language and OpenCL C.
// One physics model, sensor set and network shared by the CPU and both GPU backends.
// Tag mode: the chaser ball flies against a runner ball whose whole path was recorded beforehand (6 floats per
// physics step: position, velocity), so every network in a batch chases exactly the same runner.
#ifndef BR_SIM_CORE_H
#define BR_SIM_CORE_H

#if defined(BR_METAL)
  #define FN inline
  #define BR_GP device
  #define BR_PP thread
  #define SQRT sqrt
  #define SIN sin
  #define EXP exp
  #define LOG log
  #define TANH tanh
  #define FABS fabs
  #define FMIN fmin
  #define FMAX fmax
  #define BR_U32 uint
#elif defined(BR_OPENCL)
  #define FN inline
  #define BR_GP __global
  #define BR_PP
  #define SQRT sqrt
  #define SIN sin
  #define EXP exp
  #define LOG log
  #define TANH tanh
  #define FABS fabs
  #define FMIN fmin
  #define FMAX fmax
  #define BR_U32 uint
#else
  #include <math.h>
  #define FN static inline
  #define BR_GP
  #define BR_PP
  #define SQRT sqrtf
  #define SIN sinf
  #define EXP expf
  #define LOG logf
  #define TANH br_tanh_fast
  #define FABS fabsf
  #define FMIN fminf
  #define FMAX fmaxf
  #define BR_U32 unsigned int
#endif

#ifndef MAXW
#define MAXW 288            // widest layer incl. the input layer (29 × 5 frames + 128 memory = 273)
#endif
#define BR_NI 29            // sensor features per decision
#define BR_G 9.81f
#define BR_MAXL 12          // max layers incl. input and output

// Everything a flight needs to know, identical layout in C, Metal and OpenCL (only 4-byte scalars).
typedef struct {
  float mass, thrust, Cd, A, CNa, SM, len, I, dt, maxT, hitR, pad0;
  int ctrl;      // brain decides every `ctrl` physics steps (1 or 2)
  int nl;        // number of weight layers (hidden layers + 1)
  int K;         // past frames
  int rec;       // memory neurons on/off
  int memN;      // memory size (= width of the last hidden layer when rec)
  int nin;       // network inputs = 29·(K+1) + memN
  int nw;        // weights per genome
  int S;         // scenarios per genome
  int stride;    // floats of state per flight
  int chunk;     // physics steps per GPU dispatch
  int nRoll;     // flights in this batch
  int nActive;   // flights still in the air (GPU: compacted list)
  int mode;      // 0 = reach goal, 1 = tag (chase the recorded runner)
  int delay;     // tag: the chaser's sensors report the runner as it was this many physics steps ago
  float ballD;   // tag: centres this close = touching = tagged (balls are 2 m across)
  float noise;   // tag: sensor noise σ in metres (uniform, ±√3·σ on position, half that on velocity)
  float detect;  // tag: the chaser waits on its pad until the runner is this close (0 = launch at once)
  float pad1;
  int arch[BR_MAXL];
} BrParams;

// Goal (reach) or the runner's defended point (tag), wind, start, start tilt; tag: where this scenario's runner path
// starts in the path buffer (in steps), how many steps it has, and the seed for this scenario's sensor noise.
// atkHit: the runner reached its defended point; winnable: the algorithm with perfect sensors passed within 20 m.
typedef struct { float tx, ty, tz, wx, wy, wz, sx, sy, sz, q0, trajOff, trajLen, seed, atkHit, winnable, pad0; } BrScen;   // 64 bytes

// Per-flight state in a flat float buffer: [scalars | past frames (29·K) | memory (memN)]
#define S_PX 0
#define S_PY 1
#define S_PZ 2
#define S_VX 3
#define S_VY 4
#define S_VZ 5
#define S_QX 6
#define S_QY 7
#define S_QZ 8
#define S_QW 9
#define S_OX 10
#define S_OY 11
#define S_OZ 12
#define S_T 13
#define S_K 14
#define S_K0 15
#define S_MIND 16
#define S_ALIVE 17
#define S_HIT 18
#define S_U0 19
#define S_U1 20
#define S_U2 21
#define S_CLOSE 22          // tag: the chaser has started closing in on the runner
#define S_HIST 23

typedef struct { float px, py, pz, vx, vy, vz, qx, qy, qz, qw, ox, oy, oz, t, minD, u0, u1, u2; int k, k0, alive, hit, closing; } BrState;

// What the ball aims at this step: true position now and one step ago, true velocity, and what its sensors report.
typedef struct { float x, y, z, px, py, pz, vx, vy, vz, sx, sy, sz, svx, svy, svz; } BrTgt;

FN float br_clamp(float x, float a, float b) { return x < a ? a : (x > b ? b : x); }
#if !defined(BR_METAL) && !defined(BR_OPENCL)
// CPU tanh: Padé (7,6) approximant (max error ~1e-4), about 2× faster than tanhf; GPUs use their hardware tanh.
FN float br_tanh_fast(float x) { x = br_clamp(x, -4.97f, 4.97f); float x2 = x * x;
  return x * (135135 + x2 * (17325 + x2 * (378 + x2))) / (135135 + x2 * (62370 + x2 * (3150 + 28 * x2))); }
#endif
// Uniform random number in [-1, 1) from (seed, step, channel): stateless, so CPU and GPU draw identical noise.
FN float br_noise(BR_U32 seed, BR_U32 k, BR_U32 c) {
  BR_U32 h = seed * 0x9E3779B1u ^ k * 0x85EBCA77u ^ c * 0xC2B2AE3Du;
  h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
  return (float)(h >> 8) * (2.0f / 16777216.0f) - 1.0f;
}

FN void br_load(BR_PP BrState* s, BR_GP const float* g) {
  s->px = g[S_PX]; s->py = g[S_PY]; s->pz = g[S_PZ]; s->vx = g[S_VX]; s->vy = g[S_VY]; s->vz = g[S_VZ];
  s->qx = g[S_QX]; s->qy = g[S_QY]; s->qz = g[S_QZ]; s->qw = g[S_QW]; s->ox = g[S_OX]; s->oy = g[S_OY]; s->oz = g[S_OZ];
  s->t = g[S_T]; s->k = (int)g[S_K]; s->k0 = (int)g[S_K0]; s->minD = g[S_MIND]; s->alive = (int)g[S_ALIVE]; s->hit = (int)g[S_HIT];
  s->u0 = g[S_U0]; s->u1 = g[S_U1]; s->u2 = g[S_U2]; s->closing = (int)g[S_CLOSE];
}
FN void br_store(BR_PP const BrState* s, BR_GP float* g) {
  g[S_PX] = s->px; g[S_PY] = s->py; g[S_PZ] = s->pz; g[S_VX] = s->vx; g[S_VY] = s->vy; g[S_VZ] = s->vz;
  g[S_QX] = s->qx; g[S_QY] = s->qy; g[S_QZ] = s->qz; g[S_QW] = s->qw; g[S_OX] = s->ox; g[S_OY] = s->oy; g[S_OZ] = s->oz;
  g[S_T] = s->t; g[S_K] = (float)s->k; g[S_K0] = (float)s->k0; g[S_MIND] = s->minD; g[S_ALIVE] = (float)s->alive; g[S_HIT] = (float)s->hit;
  g[S_U0] = s->u0; g[S_U1] = s->u1; g[S_U2] = s->u2; g[S_CLOSE] = (float)s->closing;
}

// Fresh flight on the pad (history and memory zeroed).
FN void br_init(BR_GP float* g, BR_PP const BrParams* P, BR_PP const BrScen* sc) {
  for (int i = 0; i < P->stride; i++) g[i] = 0.0f;
  g[S_PX] = sc->sx; g[S_PY] = sc->sy; g[S_PZ] = sc->sz;
  g[S_QX] = sc->q0; g[S_QW] = 1.0f; g[S_MIND] = 1e30f; g[S_ALIVE] = 1.0f;
}

// Where the ball aims before step k. Reach: the fixed goal. Tag: the runner's recorded state after this step (so both
// balls are compared at the same instant), and, on decision steps, what the sensors report: the runner `delay` steps
// earlier plus noise. Returns 0 when the runner's path has ended (it landed, crashed or reached its goal).
FN int br_target(BR_PP const BrState* s, BR_PP const BrParams* P, BR_PP const BrScen* sc, BR_GP const float* traj, BR_PP BrTgt* tg) {
  if (!P->mode) {
    tg->x = tg->px = tg->sx = sc->tx; tg->y = tg->py = tg->sy = sc->ty; tg->z = tg->pz = tg->sz = sc->tz;
    tg->vx = tg->vy = tg->vz = tg->svx = tg->svy = tg->svz = 0.0f;
    return 1;
  }
  int j = s->k + 1, L = (int)sc->trajLen;
  if (j >= L) return 0;
  BR_GP const float* tr = traj + (long)sc->trajOff * 6;
  BR_GP const float* a = tr + (long)j * 6; BR_GP const float* b = a - 6;
  tg->x = a[0]; tg->y = a[1]; tg->z = a[2]; tg->vx = a[3]; tg->vy = a[4]; tg->vz = a[5];
  tg->px = b[0]; tg->py = b[1]; tg->pz = b[2];
  if (s->k % P->ctrl == 0) {
    int js = j - P->delay; if (js < 0) js = 0;
    BR_GP const float* c = tr + (long)js * 6;
    float n = P->noise * 1.732f; BR_U32 seed = (BR_U32)sc->seed, kk = (BR_U32)s->k;
    tg->sx = c[0] + br_noise(seed, kk, 0u) * n; tg->sy = c[1] + br_noise(seed, kk, 1u) * n; tg->sz = c[2] + br_noise(seed, kk, 2u) * n;
    tg->svx = c[3] + br_noise(seed, kk, 3u) * n * 0.5f; tg->svy = c[4] + br_noise(seed, kk, 4u) * n * 0.5f; tg->svz = c[5] + br_noise(seed, kk, 5u) * n * 0.5f;
  }
  return 1;
}

// The 29 sensor features (body frame): target position and velocity as
// the sensors report them (a fixed goal has zero velocity).
FN void br_inputs(BR_PP const BrState* s, BR_PP const BrTgt* tg, BR_PP float* x) {
  float qx = s->qx, qy = s->qy, qz = s->qz, qw = s->qw;
  float m00 = 1 - 2 * (qy * qy + qz * qz), m01 = 2 * (qx * qy + qw * qz), m02 = 2 * (qx * qz - qw * qy);
  float m10 = 2 * (qx * qy - qw * qz), m11 = 1 - 2 * (qx * qx + qz * qz), m12 = 2 * (qy * qz + qw * qx);
  float m20 = 2 * (qx * qz + qw * qy), m21 = 2 * (qy * qz - qw * qx), m22 = 1 - 2 * (qx * qx + qy * qy);
  float dx = tg->sx - s->px, dy = tg->sy - s->py, dz = tg->sz - s->pz;
  float rx = m00 * dx + m01 * dy + m02 * dz, ry = m10 * dx + m11 * dy + m12 * dz, rz = m20 * dx + m21 * dy + m22 * dz;
  float bvx = m00 * s->vx + m01 * s->vy + m02 * s->vz, bvy = m10 * s->vx + m11 * s->vy + m12 * s->vz, bvz = m20 * s->vx + m21 * s->vy + m22 * s->vz;
  float R = SQRT(rx * rx + ry * ry + rz * rz) + 1e-6f;
  x[0] = rx / 3000; x[1] = ry / 3000; x[2] = rz / 3000;
  x[3] = bvx / 300; x[4] = bvy / 300; x[5] = bvz / 300;
  x[6] = (m00 * s->ox + m01 * s->oy + m02 * s->oz) / 2; x[7] = (m10 * s->ox + m11 * s->oy + m12 * s->oz) / 2; x[8] = (m20 * s->ox + m21 * s->oy + m22 * s->oz) / 2;
  x[9] = m01; x[10] = m11; x[11] = m21; x[13] = s->py / 2000;
  x[14] = rx / R; x[15] = ry / R; x[16] = rz / R; x[17] = LOG(R / 500);
  x[18] = (m00 * tg->svx + m01 * tg->svy + m02 * tg->svz) / 300; x[19] = (m10 * tg->svx + m11 * tg->svy + m12 * tg->svz) / 300;
  x[20] = (m20 * tg->svx + m21 * tg->svy + m22 * tg->svz) / 300;
  float ux = (x[18] - x[3]) * 300, uy = (x[19] - x[4]) * 300, uz = (x[20] - x[5]) * 300;   // target velocity − own velocity
  float vc = -(rx * ux + ry * uy + rz * uz) / R, R2 = R * R;
  x[12] = vc / 300;
  x[21] = br_clamp((ry * uz - rz * uy) / R2 * 10, -5, 5); x[22] = br_clamp((rz * ux - rx * uz) / R2 * 10, -5, 5); x[23] = br_clamp((rx * uy - ry * ux) / R2 * 10, -5, 5);
  float tgo = vc > 1 ? FMIN(R / vc, 60.0f) : 60.0f; x[24] = tgo / 20;
  float zx = rx + ux * tgo, zy = ry + uy * tgo, zz = rz + uz * tgo, zm = SQRT(zx * zx + zy * zy + zz * zz) + 1e-6f;
  x[25] = zx / zm; x[26] = zy / zm; x[27] = zz / zm; x[28] = LOG(1 + zm) / 5;
}

// One decision: assemble [sensors now | past frames | memory], run the network, read throttle/pitch/yaw.
FN void br_control(BR_PP BrState* s, BR_PP const BrParams* P, BR_PP const BrTgt* tg, BR_GP const float* w, BR_GP float* hist, BR_GP float* mem) {
  float a[MAXW], b[MAXW];
  br_inputs(s, tg, a);
  if (P->K > 0) {
    for (int i = 0; i < BR_NI * P->K; i++) a[BR_NI + i] = hist[i];
    for (int i = BR_NI * P->K - 1; i >= BR_NI; i--) hist[i] = hist[i - BR_NI];
    for (int i = 0; i < BR_NI; i++) hist[i] = a[i];
  }
  if (P->rec) { int o = BR_NI * (P->K + 1); for (int i = 0; i < P->memN; i++) a[o + i] = mem[i]; }
  BR_PP float* src = a; BR_PP float* dst = b;
  int off = 0;
  for (int l = 0; l < P->nl; l++) {
    int ni = P->arch[l], no = P->arch[l + 1];
#if defined(BR_METAL) || defined(BR_OPENCL)
    // GPU: genome layout as stored (per layer: weights row-major out×in, then biases)
    BR_GP const float* W = w + off;
    BR_GP const float* B = w + off + no * ni;
    for (int j = 0; j < no; j++) {
      float acc = B[j];
      BR_GP const float* row = W + j * ni;
      for (int i = 0; i < ni; i++) acc += row[i] * src[i];
      dst[j] = TANH(acc);
    }
#else
    // CPU: layer repacked as [biases | weights transposed in×out]; the inner loop runs across neurons, so it
    // vectorises (NEON/AVX) with no dependency chain.
    const float* B = w + off; const float* WT = w + off + no;
    for (int j = 0; j < no; j++) dst[j] = B[j];
    for (int i = 0; i < ni; i++) { float xi = src[i]; const float* col = WT + i * no; for (int j = 0; j < no; j++) dst[j] += col[j] * xi; }
    for (int j = 0; j < no; j++) dst[j] = TANH(dst[j]);
#endif
    off += no * ni + no;
    if (P->rec && l == P->nl - 2) for (int i = 0; i < P->memN; i++) mem[i] = dst[i];   // last hidden layer = next memory
    BR_PP float* tmp = src; src = dst; dst = tmp;
  }
  s->u0 = (src[0] + 1) * 0.5f; s->u1 = src[1]; s->u2 = src[2];
}

// One physics step: thrust, gravity, drag with transonic rise, lift, weathercock torque, controls; then the hit test
// (reach: within hitR of the goal; tag: the balls touch at any moment of the step) and the end conditions.
FN void br_step(BR_PP BrState* s, BR_PP const BrParams* P, BR_PP const BrScen* sc, BR_PP const BrTgt* tg) {
  float dt = P->dt;
  // Tag, launch on detection: sit on the pad until the runner comes within detection range.
  if (P->mode && P->detect > 0 && s->vx == 0 && s->vy == 0 && s->vz == 0) {
    float ddx = tg->x - s->px, ddy = tg->y - s->py, ddz = tg->z - s->pz;
    if (SQRT(ddx * ddx + ddy * ddy + ddz * ddz) > P->detect) { s->t += dt; s->k0 = (int)(s->t / dt + 0.5f); return; }
  }
  float p0x = s->px, p0y = s->py, p0z = s->pz;
  float qx = s->qx, qy = s->qy, qz = s->qz, qw = s->qw, m = P->mass;
  float ax0 = 2 * (qx * qy - qw * qz), ax1 = 1 - 2 * (qx * qx + qz * qz), ax2 = 2 * (qy * qz + qw * qx);
  float rho = 1.225f * EXP(-FMAX(s->py, 0.0f) / 8500);
  float vrx = s->vx - sc->wx, vry = s->vy - sc->wy, vrz = s->vz - sc->wz;
  float V = SQRT(vrx * vrx + vry * vry + vrz * vrz), qd = 0.5f * rho * V * V;
  float mach = V / (340 - 0.004f * FMAX(s->py, 0.0f));
  float fx = 0, fy = -BR_G * m, fz = 0, Tx = 0, Ty = 0, Tz = 0, T = 0;
  if (s->u0 > 0) T = P->thrust * s->u0;
  fx += ax0 * T; fy += ax1 * T; fz += ax2 * T;
  if (V > 0.5f) {
    float hx = vrx / V, hy = vry / V, hz = vrz / V, ca = ax0 * hx + ax1 * hy + ax2 * hz;
    // wave drag: rises through the transonic band (half way at Mach 0.95), peaks ~2.1× near Mach 1 and stays high
    // supersonic, easing off as 1/√M (≈1.8× at Mach 2, 1.6× at Mach 3) like a projectile's drag curve
    float sin2 = 1 - ca * ca, rise = 1 / (1 + EXP(-(mach - 0.95f) / 0.05f)), wave = 1 + rise * 1.1f / SQRT(FMAX(mach, 1.0f));
    float D = qd * P->A * (P->Cd * wave + 1.4f * sin2);
    fx -= hx * D; fy -= hy * D; fz -= hz * D;
    float L = qd * P->A * P->CNa * FABS(ca);
    fx += (ax0 - hx * ca) * L; fy += (ax1 - hy * ca) * L; fz += (ax2 - hz * ca) * L;
    float kk = qd * P->A * P->CNa * P->SM;
    Tx += (ax1 * hz - ax2 * hy) * kk; Ty += (ax2 * hx - ax0 * hz) * kk; Tz += (ax0 * hy - ax1 * hx) * kk;
    float damp = kk * P->len / FMAX(V, 1.0f) * 2;
    Tx -= s->ox * damp; Ty -= s->oy * damp; Tz -= s->oz * damp;
  }
  float auth = T * 0.12f + qd * P->A * 0.15f + 40;
  float cx = br_clamp(s->u1, -1, 1) * auth, cz = br_clamp(s->u2, -1, 1) * auth;
  Tx += (1 - 2 * (qy * qy + qz * qz)) * cx + 2 * (qx * qz + qw * qy) * cz - s->ox * 2;
  Ty += 2 * (qx * qy + qw * qz) * cx + 2 * (qy * qz - qw * qx) * cz - s->oy * 2;
  Tz += 2 * (qx * qz - qw * qy) * cx + (1 - 2 * (qx * qx + qy * qy)) * cz - s->oz * 2;
  s->ox += Tx / P->I * dt; s->oy += Ty / P->I * dt; s->oz += Tz / P->I * dt;
  float h = 0.5f * dt, ox = s->ox, oy = s->oy, oz = s->oz;
  float nx = qx + h * (ox * qw + oy * qz - oz * qy), ny = qy + h * (oy * qw + oz * qx - ox * qz),
        nz = qz + h * (oz * qw + ox * qy - oy * qx), nw = qw - h * (ox * qx + oy * qy + oz * qz);
  float n = 1 / SQRT(nx * nx + ny * ny + nz * nz + nw * nw);
  s->qx = nx * n; s->qy = ny * n; s->qz = nz * n; s->qw = nw * n;
  s->vx += fx / m * dt; s->vy += fy / m * dt; s->vz += fz / m * dt;
  s->px += s->vx * dt; s->py += s->vy * dt; s->pz += s->vz * dt;
  s->t += dt;
  if (s->py < 1.6f && s->t - (float)s->k0 * dt < 4) { s->py = 1.6f; if (s->vy < 0) s->vy = 0; s->vx *= 0.9f; s->vz *= 0.9f; }
  float d;
  if (P->mode) {   // closest approach during this step (both balls move in straight lines), so fast passes can't tunnel
    float a0 = p0x - tg->px, a1 = p0y - tg->py, a2 = p0z - tg->pz;
    float e0 = s->px - tg->x - a0, e1 = s->py - tg->y - a1, e2 = s->pz - tg->z - a2, dd = e0 * e0 + e1 * e1 + e2 * e2;
    float u = dd > 0 ? br_clamp(-(a0 * e0 + a1 * e1 + a2 * e2) / dd, 0, 1) : 0;
    float c0 = a0 + e0 * u, c1 = a1 + e1 * u, c2 = a2 + e2 * u;
    d = SQRT(c0 * c0 + c1 * c1 + c2 * c2);
  } else {
    float ex = s->px - tg->x, ey = s->py - tg->y, ez = s->pz - tg->z;
    d = SQRT(ex * ex + ey * ey + ez * ez);
  }
  if (d < s->minD) s->minD = d;
  if (d < (P->mode ? P->ballD : P->hitR)) { s->alive = 0; s->hit = 1; return; }
  if (P->mode) {   // once it has closed in, the moment the gap starts widening it has passed the runner: a miss, no second try
    float rx = s->px - tg->x, ry = s->py - tg->y, rz = s->pz - tg->z;
    float rr = rx * (s->vx - tg->vx) + ry * (s->vy - tg->vy) + rz * (s->vz - tg->vz);
    if (rr < 0 && SQRT(rx * rx + ry * ry + rz * rz) < 3000) s->closing = 1;
    else if (rr > 0 && s->closing) { s->alive = 0; return; }
  }
  if (s->py < 0) s->alive = 0;
  else if (s->t > P->maxT) s->alive = 0;
}

// Guidance algorithm: pitch-over boost, long-range cruise for fixed goals, then
// zero-effort-miss guidance; coasts on fins when on course. Aims at what the sensors report (tg->s*); a target is
// "moving" in tag mode or whenever its reported velocity is non-zero. Writes throttle, pitch, yaw into u.
FN void br_algo_control(BR_PP const BrState* s, BR_PP const BrParams* P, BR_PP const BrScen* sc, BR_PP const BrTgt* tg, BR_PP float* u) {
  float qx = s->qx, qy = s->qy, qz = s->qz, qw = s->qw, dt = P->dt;
  float ax0 = 2 * (qx * qy - qw * qz), ax1 = 1 - 2 * (qx * qx + qz * qz), ax2 = 2 * (qy * qz + qw * qx);
  float vx = s->vx, vy = s->vy, vz = s->vz, V = SQRT(vx * vx + vy * vy + vz * vz); if (V == 0) V = 1e-6f;
  float r0 = tg->sx - s->px, r1 = tg->sy - s->py, r2 = tg->sz - s->pz, R = SQRT(r0 * r0 + r1 * r1 + r2 * r2);
  float hd = SQRT(r0 * r0 + r2 * r2); if (hd == 0) hd = 1;
  float dh0 = r0 / hd, dh2 = r2 / hd;
  int mov = P->mode || tg->svx != 0 || tg->svy != 0 || tg->svz != 0;
  float tl = s->t - (float)s->k0 * dt;   // time since this ball's own launch
  float d0, d1, d2, thr = 1;
  if (!mov && hd > 9000 && !(V < 90 && tl < 15)) {   // long-range cruise: level at ~2 km, ~250 m/s
    float climb = br_clamp((2000 - s->py) / 1500, -0.35f, 0.35f) - vy / V * 0.6f, n = SQRT(1 + climb * climb);
    d0 = dh0 / n; d1 = climb / n; d2 = dh2 / n;
    thr = br_clamp(0.55f + (250 - V) / 60, 0, 1);
  } else if (V < 90 && tl < 15) {                      // climb-out
    float lean = mov ? 0.35f : 0.55f, n = SQRT(lean * lean + 1);
    d0 = dh0 * lean / n; d1 = 1 / n; d2 = dh2 * lean / n;
  } else {
    float tgo, z0, z1, z2;
    if (mov) {
      float rvx = tg->svx - vx, rvy = tg->svy - vy, rvz = tg->svz - vz;
      float close = -(r0 * rvx + r1 * rvy + r2 * rvz) / (R != 0 ? R : 1);
      tgo = FMAX(R / FMAX(close, 80.0f), 0.3f);
      z0 = r0 + rvx * tgo; z1 = r1 + rvy * tgo; z2 = r2 + rvz * tgo;
    } else {
      tgo = FMAX(R / FMAX(V, 60.0f), 0.5f);
      z0 = r0 - vx * tgo; z1 = r1 - vy * tgo + 0.5f * BR_G * tgo * tgo; z2 = r2 - vz * tgo;
    }
    float gk = (mov ? 18.0f : 8.0f) / (tgo * tgo);
    float a0 = gk * z0, a1 = gk * z1, a2 = gk * z2, h0 = vx / V, h1 = vy / V, h2 = vz / V, ad = a0 * h0 + a1 * h1 + a2 * h2;
    float p0 = a0 - h0 * ad, p1 = a1 - h1 * ad, p2 = a2 - h2 * ad, apm = SQRT(p0 * p0 + p1 * p1 + p2 * p2);
    float Ta = P->thrust / P->mass, zm = SQRT(z0 * z0 + z1 * z1 + z2 * z2);
    if ((!mov && zm < 40 && V > 150) || V > 650) {
      thr = 0; float g = FMIN(0.006f, 0.35f / FMAX(apm, 1e-6f));
      d0 = h0 + p0 * g; d1 = h1 + p1 * g; d2 = h2 + p2 * g;
    } else {   // speed governor: above cruise speed, spend thrust only on steering
      float Vc = mov ? br_clamp(0.12f * R + 250, 300, 550) : br_clamp(SQRT(BR_G * R) * 1.3f, 160, 420);
      float along = SQRT(FMAX(Ta * Ta - apm * apm, 0.09f * Ta * Ta)) * (ad < -Ta * 0.5f ? 0.3f : 1.0f);
      if (V > Vc) { along = 0; thr = br_clamp(apm / Ta, 0, 1); }
      if (apm > 1e-3f || along > 0) { d0 = p0 + h0 * along; d1 = p1 + h1 * along; d2 = p2 + h2 * along; }
      else { d0 = h0; d1 = h1; d2 = h2; }
    }
    float n = SQRT(d0 * d0 + d1 * d1 + d2 * d2); if (n == 0) n = 1;
    d0 /= n; d1 /= n; d2 /= n;
  }
  if (!mov) thr = FMIN(thr, 2.6f * P->mass * BR_G / P->thrust);   // reach: cap thrust-to-weight near 2.6
  float e0 = ax1 * d2 - ax2 * d1, e1 = ax2 * d0 - ax0 * d2, e2 = ax0 * d1 - ax1 * d0;
  // dynamic pressure from the current state
  float rho = 1.225f * EXP(-FMAX(s->py, 0.0f) / 8500);
  float wx = vx - sc->wx, wy = vy - sc->wy, wz = vz - sc->wz, qd = 0.5f * rho * (wx * wx + wy * wy + wz * wz);
  float auth = (thr > 0 ? P->thrust * thr : 0) * 0.12f + qd * P->A * 0.15f + 40, scl = 352 / auth;
  float kp = (mov ? 12.0f : 6.0f) * scl, kd = 2.5f * scl;
  float c0 = kp * e0 - kd * s->ox, c1 = kp * e1 - kd * s->oy, c2 = kp * e2 - kd * s->oz;
  // world → body (inverse rotation): conjugate quaternion
  float ix = -qx, iy = -qy, iz = -qz;
  float t0 = 2 * (iy * c2 - iz * c1), t1 = 2 * (iz * c0 - ix * c2), t2 = 2 * (ix * c1 - iy * c0);
  float b0 = c0 + qw * t0 + iy * t2 - iz * t1, b2 = c2 + qw * t2 + ix * t1 - iy * t0;
  u[0] = thr; u[1] = br_clamp(b0, -1, 1); u[2] = br_clamp(b2, -1, 1);
}

// Weaving runner: the algorithm plus two sine weaves of amplitude `evade` on pitch and
// yaw after t > 6 s. Frequencies w1, w2 and phases f1, f2 are drawn by the caller (the core has no random source).
FN void br_evader_control(BR_PP const BrState* s, BR_PP const BrParams* P, BR_PP const BrScen* sc, BR_PP const BrTgt* tg,
                          float evade, float w1, float f1, float w2, float f2, BR_PP float* u) {
  br_algo_control(s, P, sc, tg, u);
  if (evade > 0 && s->t > 6) {
    u[1] = br_clamp(u[1] + evade * SIN(w1 * s->t + f1), -1, 1);
    u[2] = br_clamp(u[2] + evade * SIN(w2 * s->t + f2), -1, 1);
  }
}

// Fly up to `steps` physics steps (or until the flight ends). `traj` is the runner path buffer (tag mode only).
FN void br_run(BR_GP float* g, BR_PP const BrParams* P, BR_PP const BrScen* sc, BR_GP const float* w, BR_GP const float* traj, int steps) {
  BrState s; br_load(&s, g);
  BR_GP float* hist = g + S_HIST;
  BR_GP float* mem = g + S_HIST + BR_NI * P->K;
  BrTgt tg;
  tg.sx = tg.sy = tg.sz = tg.svx = tg.svy = tg.svz = 0.0f;   // set on the first decision step (k = 0)
  for (int i = 0; i < steps && s.alive; i++) {
    if (!br_target(&s, P, sc, traj, &tg)) { s.alive = 0; break; }   // tag: the runner's flight is over
    if (s.k % P->ctrl == 0) br_control(&s, P, &tg, w, hist, mem);
    s.k++;
    br_step(&s, P, sc, &tg);
  }
  br_store(&s, g);
}

#endif
