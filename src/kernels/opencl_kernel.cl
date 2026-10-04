#ifndef BR_SW_LOCAL   // (the work-group battle kernel is built separately, with battle state on chip)
// OpenCL kernel: one work-item = one flight, advanced `chunk` physics steps per dispatch.
__kernel void br_kernel(__global float* state, __global const float* weights,
                        __constant BrScen* scen, __constant BrParams* pp, __global volatile uint* alive,
                        __global const int* idx, __global int* flags, __global const float* traj) {
  BrParams P = *pp;
  int gid = (int)get_global_id(0);
  if (gid >= P.nActive) return;
  int r = idx[gid];                       // only flights still in the air are launched (keeps wavefronts full)
  __global float* g = state + (size_t)r * (size_t)P.stride;
  int gi = r / P.S, si = r % P.S;
  BrScen sc = scen[si];
  br_run(g, &P, &sc, weights + (size_t)gi * (size_t)P.nw, traj, P.chunk);
  int a = g[S_ALIVE] != 0.0f;
  flags[r] = a;
  if (a) atomic_inc(alive);
}

// Swarm: one work-item = one battle (bat[3r..3r+2]: attacker genome, defender genome, scenario), `chunk` steps per dispatch.
__kernel void br_sw_kernel(__global float* state, __global const float* wA, __global const float* wD, __global const BrScen* scen,
                           __constant BrParams* pp, __global volatile uint* alive, __global const int* idx, __global int* flags,
                           __global const int* bat) {
  BrParams P = *pp;
  int gid = (int)get_global_id(0);
  if (gid >= P.nActive) return;
  int r = idx[gid];
  __global float* g = state + (size_t)r * (size_t)P.stride;
  BrScen sc = scen[bat[r * 3 + 2]];
  br_sw_run(g, &P, &sc, wA + (size_t)bat[r * 3] * (size_t)P.attNw, wD + (size_t)bat[r * 3 + 1] * (size_t)P.defNw, P.chunk);
  int a = g[SWH_DONE] == 0.0f;
  flags[r] = a;
  if (a) atomic_inc(alive);
}

#endif

#ifdef BR_SW_LOCAL
// Swarm, large battles: one work-group per battle, one work-item per ball (looping when there are more balls than
// work-items). The battle is copied into __local memory (BR_SG = __local, SW_STRIDE floats), flown there for `chunk`
// steps with barriers between phases, then copied back. A commander side's network is split across the work-items:
// each computes whole neurons of a layer, in the same order as br_sw_forward, so results do not change.
__kernel void br_sw_group_kernel(__global float* state, __global const float* wA, __global const float* wD, __global const BrScen* scen,
                                 __constant BrParams* pp, __global volatile uint* alive, __global const int* idx, __global int* flags,
                                 __global const int* bat) {
  __local float g[SW_STRIDE];
  __local float ta[SW_MAXW], tb[SW_MAXW];
  BrParams P = *pp;
  int grp = (int)get_group_id(0), T = (int)get_local_id(0), N = (int)get_local_size(0);
  if (grp >= P.nActive) return;   // uniform for the whole group
  int r = idx[grp], nA = P.attN, nB = nA + P.defN;
  __global float* gg = state + (size_t)r * (size_t)P.stride;
  for (int i = T; i < P.stride; i += N) g[i] = gg[i];
  BrScen sc = scen[bat[r * 3 + 2]];
  __global const float* A = wA + (size_t)bat[r * 3] * (size_t)P.attNw; __global const float* D = wD + (size_t)bat[r * 3 + 1] * (size_t)P.defNw;
  BrParams PA = P; PA.thrust = P.thrustAtk;
  uint seed = (uint)sc.seed; float nz = P.noise * 1.732f;
  float a[SW_MAXW], b[SW_MAXW];
  for (int it = 0; it < P.chunk; it++) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (g[SWH_DONE] != 0) break;
    int k = (int)g[SWH_K];
    if (k % P.ctrl == 0) {
      if (T == 0) br_sw_assign(g, &P);
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int side = 0; side < 2; side++) {
        if (!(side ? P.defAI && P.defCmd : P.attAI && P.attCmd)) continue;
        int nin = side ? P.defNin : P.attNin, nout = side ? P.defNout : P.attNout; __global const float* w = side ? D : A;
        if (T == 0) { br_sw_cmd_inputs(g, &P, &sc, side, seed, (uint)k, nz, a); for (int i = 0; i < nin; i++) ta[i] = a[i]; }
        barrier(CLK_LOCAL_MEM_FENCE);
        __local float* src = ta; __local float* dst = tb; int off = 0;
        for (int l = 0; l < P.nl; l++) {
          int ni = l == 0 ? nin : P.arch[l], no = l == P.nl - 1 ? nout : P.arch[l + 1];
          __global const float* W = w + off; __global const float* Bv = w + off + no * ni;
          for (int j = T; j < no; j += N) { float acc = Bv[j]; __global const float* row = W + j * ni; for (int i = 0; i < ni; i++) acc += row[i] * src[i]; dst[j] = TANH(acc); }
          barrier(CLK_LOCAL_MEM_FENCE);
          off += no * ni + no; __local float* t2 = src; src = dst; dst = t2;
        }
        int o0 = side ? nA : 0, o1 = side ? nB : nA;
        for (int bi = o0 + T; bi < o1; bi += N) { __local float* q = g + SW_H + bi * SW_B; int j = (bi - o0) * 3;
          q[S_U0] = (src[j] + 1) * 0.5f; q[S_U1] = src[j + 1]; q[S_U2] = src[j + 2]; }
        barrier(CLK_LOCAL_MEM_FENCE);
      }
      for (int bi = T; bi < nB; bi += N) br_sw_decide_ball(g, &P, &PA, &sc, A, D, bi, seed, k, nz, a, b);
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int bi = T; bi < nB; bi += N) br_sw_move_ball(g, &P, &PA, &sc, bi);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int d = nA + T; d < nB; d += N) br_sw_scan(g, &P, d);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (T == 0) br_sw_resolve(g, &P, &sc);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int i = T; i < P.stride; i += N) gg[i] = g[i];
  if (T == 0) { int a2 = g[SWH_DONE] == 0.0f; flags[r] = a2; if (a2) atomic_inc(alive); }
}
#endif
