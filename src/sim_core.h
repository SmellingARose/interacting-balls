// Ballistic Range — native simulation core (reach-target mode).
// One source, three compilers: plain C (CPU backend), Metal Shading Language and OpenCL C.
// Same physics, sensor inputs, past frames, memory neurons and network as the browser version.
#ifndef BR_SIM_CORE_H
#define BR_SIM_CORE_H

#if defined(BR_METAL)
  #define FN inline
  #define BR_GP device
  #define BR_PP thread
  #define SQRT sqrt
  #define EXP exp
  #define LOG log
  #define TANH tanh
  #define FABS fabs
  #define FMIN fmin
  #define FMAX fmax
#elif defined(BR_OPENCL)
  #define FN inline
  #define BR_GP __global
  #define BR_PP
  #define SQRT sqrt
  #define EXP exp
  #define LOG log
  #define TANH tanh
  #define FABS fabs
  #define FMIN fmin
  #define FMAX fmax
#else
  #include <math.h>
  #define FN static inline
  #define BR_GP
  #define BR_PP
  #define SQRT sqrtf
  #define EXP expf
  #define LOG logf
  #define TANH br_tanh_fast
  #define FABS fabsf
  #define FMIN fminf
  #define FMAX fmaxf
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
  int ctrl;      // pilot decides every `ctrl` physics steps (1 or 2)
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
  int arch[BR_MAXL];
} BrParams;

typedef struct { float tx, ty, tz, wx, wy, wz, sx, sy, sz, q0, pad0, pad1; } BrScen;   // target, wind, start, start tilt

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
#define S_HIST 22

typedef struct { float px, py, pz, vx, vy, vz, qx, qy, qz, qw, ox, oy, oz, t, minD, u0, u1, u2; int k, k0, alive, hit; } BrState;

FN float br_clamp(float x, float a, float b) { return x < a ? a : (x > b ? b : x); }
#if !defined(BR_METAL) && !defined(BR_OPENCL)
// CPU tanh: Padé (7,6) approximant (max error ~1e-4), about 2× faster than tanhf; GPUs use their hardware tanh.
FN float br_tanh_fast(float x) { x = br_clamp(x, -4.97f, 4.97f); float x2 = x * x;
  return x * (135135 + x2 * (17325 + x2 * (378 + x2))) / (135135 + x2 * (62370 + x2 * (3150 + 28 * x2))); }
#endif

FN void br_load(BR_PP BrState* s, BR_GP const float* g) {
  s->px = g[S_PX]; s->py = g[S_PY]; s->pz = g[S_PZ]; s->vx = g[S_VX]; s->vy = g[S_VY]; s->vz = g[S_VZ];
  s->qx = g[S_QX]; s->qy = g[S_QY]; s->qz = g[S_QZ]; s->qw = g[S_QW]; s->ox = g[S_OX]; s->oy = g[S_OY]; s->oz = g[S_OZ];
  s->t = g[S_T]; s->k = (int)g[S_K]; s->k0 = (int)g[S_K0]; s->minD = g[S_MIND]; s->alive = (int)g[S_ALIVE]; s->hit = (int)g[S_HIT];
  s->u0 = g[S_U0]; s->u1 = g[S_U1]; s->u2 = g[S_U2];
}
FN void br_store(BR_PP const BrState* s, BR_GP float* g) {
  g[S_PX] = s->px; g[S_PY] = s->py; g[S_PZ] = s->pz; g[S_VX] = s->vx; g[S_VY] = s->vy; g[S_VZ] = s->vz;
  g[S_QX] = s->qx; g[S_QY] = s->qy; g[S_QZ] = s->qz; g[S_QW] = s->qw; g[S_OX] = s->ox; g[S_OY] = s->oy; g[S_OZ] = s->oz;
  g[S_T] = s->t; g[S_K] = (float)s->k; g[S_K0] = (float)s->k0; g[S_MIND] = s->minD; g[S_ALIVE] = (float)s->alive; g[S_HIT] = (float)s->hit;
  g[S_U0] = s->u0; g[S_U1] = s->u1; g[S_U2] = s->u2;
}

// Fresh flight on the pad (history and memory zeroed).
FN void br_init(BR_GP float* g, BR_PP const BrParams* P, BR_PP const BrScen* sc) {
  for (int i = 0; i < P->stride; i++) g[i] = 0.0f;
  g[S_PX] = sc->sx; g[S_PY] = sc->sy; g[S_PZ] = sc->sz;
  g[S_QX] = sc->q0; g[S_QW] = 1.0f; g[S_MIND] = 1e30f; g[S_ALIVE] = 1.0f;
}

// The 29 sensor features (body frame), same formulas as the browser's nnInputs() with a fixed target.
FN void br_inputs(BR_PP const BrState* s, BR_PP const BrScen* sc, BR_PP float* x) {
  float qx = s->qx, qy = s->qy, qz = s->qz, qw = s->qw;
  float m00 = 1 - 2 * (qy * qy + qz * qz), m01 = 2 * (qx * qy + qw * qz), m02 = 2 * (qx * qz - qw * qy);
  float m10 = 2 * (qx * qy - qw * qz), m11 = 1 - 2 * (qx * qx + qz * qz), m12 = 2 * (qy * qz + qw * qx);
  float m20 = 2 * (qx * qz + qw * qy), m21 = 2 * (qy * qz - qw * qx), m22 = 1 - 2 * (qx * qx + qy * qy);
  float dx = sc->tx - s->px, dy = sc->ty - s->py, dz = sc->tz - s->pz;
  float rx = m00 * dx + m01 * dy + m02 * dz, ry = m10 * dx + m11 * dy + m12 * dz, rz = m20 * dx + m21 * dy + m22 * dz;
  float bvx = m00 * s->vx + m01 * s->vy + m02 * s->vz, bvy = m10 * s->vx + m11 * s->vy + m12 * s->vz, bvz = m20 * s->vx + m21 * s->vy + m22 * s->vz;
  float R = SQRT(rx * rx + ry * ry + rz * rz) + 1e-6f;
  x[0] = rx / 3000; x[1] = ry / 3000; x[2] = rz / 3000;
  x[3] = bvx / 300; x[4] = bvy / 300; x[5] = bvz / 300;
  x[6] = (m00 * s->ox + m01 * s->oy + m02 * s->oz) / 2; x[7] = (m10 * s->ox + m11 * s->oy + m12 * s->oz) / 2; x[8] = (m20 * s->ox + m21 * s->oy + m22 * s->oz) / 2;
  x[9] = m01; x[10] = m11; x[11] = m21; x[13] = s->py / 2000;
  x[14] = rx / R; x[15] = ry / R; x[16] = rz / R; x[17] = LOG(R / 500);
  x[18] = 0; x[19] = 0; x[20] = 0;                         // target velocity: the beacon does not move
  float ux = -x[3] * 300, uy = -x[4] * 300, uz = -x[5] * 300;   // relative velocity = 0 − own velocity
  float vc = -(rx * ux + ry * uy + rz * uz) / R, R2 = R * R;
  x[12] = vc / 300;
  x[21] = br_clamp((ry * uz - rz * uy) / R2 * 10, -5, 5); x[22] = br_clamp((rz * ux - rx * uz) / R2 * 10, -5, 5); x[23] = br_clamp((rx * uy - ry * ux) / R2 * 10, -5, 5);
  float tgo = vc > 1 ? FMIN(R / vc, 60.0f) : 60.0f; x[24] = tgo / 20;
  float zx = rx + ux * tgo, zy = ry + uy * tgo, zz = rz + uz * tgo, zm = SQRT(zx * zx + zy * zy + zz * zz) + 1e-6f;
  x[25] = zx / zm; x[26] = zy / zm; x[27] = zz / zm; x[28] = LOG(1 + zm) / 5;
}

// One decision: assemble [sensors now | past frames | memory], run the network, read throttle/pitch/yaw.
FN void br_control(BR_PP BrState* s, BR_PP const BrParams* P, BR_PP const BrScen* sc, BR_GP const float* w, BR_GP float* hist, BR_GP float* mem) {
  float a[MAXW], b[MAXW];
  br_inputs(s, sc, a);
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

// One physics step (reach-target mode): thrust, gravity, drag with transonic rise, lift, weathercock torque, controls.
FN void br_step(BR_PP BrState* s, BR_PP const BrParams* P, BR_PP const BrScen* sc) {
  float qx = s->qx, qy = s->qy, qz = s->qz, qw = s->qw, dt = P->dt, m = P->mass;
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
  float ex = s->px - sc->tx, ey = s->py - sc->ty, ez = s->pz - sc->tz, d = SQRT(ex * ex + ey * ey + ez * ez);
  if (d < s->minD) s->minD = d;
  if (d < P->hitR) { s->alive = 0; s->hit = 1; return; }
  if (s->py < 0) s->alive = 0;
  else if (s->t > P->maxT) s->alive = 0;
}

// Fly up to `steps` physics steps (or until the flight ends).
FN void br_run(BR_GP float* g, BR_PP const BrParams* P, BR_PP const BrScen* sc, BR_GP const float* w, int steps) {
  BrState s; br_load(&s, g);
  BR_GP float* hist = g + S_HIST;
  BR_GP float* mem = g + S_HIST + BR_NI * P->K;
  for (int i = 0; i < steps && s.alive; i++) {
    if (s.k % P->ctrl == 0) br_control(&s, P, sc, w, hist, mem);
    s.k++;
    br_step(&s, P, sc);
  }
  br_store(&s, g);
}

#endif
