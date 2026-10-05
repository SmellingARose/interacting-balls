#ifndef BR_SW_LOCAL   // (the work-group battle kernel is built separately, with battle state on chip)
// Metal compute kernel: one thread = one flight, advanced `chunk` physics steps per dispatch.
kernel void br_kernel(device float* state [[buffer(0)]],
                      device const float* weights [[buffer(1)]],
                      device const BrScen* scen [[buffer(2)]],   // each thread reads a different one
                      constant BrParams& pp [[buffer(3)]],
                      device atomic_uint* alive [[buffer(4)]],
                      device const int* idx [[buffer(5)]],
                      device int* flags [[buffer(6)]],
                      device const float* traj [[buffer(7)]],
                      uint gid [[thread_position_in_grid]]) {
  BrParams P = pp;
  if ((int)gid >= P.nActive) return;
  int r = idx[gid];                       // only flights still in the air are launched (keeps SIMD groups full)
  device float* g = state + (ulong)r * (ulong)P.stride;
  int gi = r / P.S, si = r % P.S;
  BrScen sc = scen[si];
  br_run(g, &P, &sc, weights + (ulong)gi * (ulong)P.nw, traj, P.chunk);
  int a = g[S_ALIVE] != 0.0f;
  flags[r] = a;
  if (a) atomic_fetch_add_explicit(alive, 1u, memory_order_relaxed);
}

// Swarm: one thread = one battle (bat[3r..3r+2]: attacker genome, defender genome, scenario), `chunk` steps per dispatch.
kernel void br_sw_kernel(device float* state [[buffer(0)]],
                         device const float* wA [[buffer(1)]],
                         device const float* wD [[buffer(2)]],
                         device const BrScen* scen [[buffer(3)]],
                         constant BrParams& pp [[buffer(4)]],
                         device atomic_uint* alive [[buffer(5)]],
                         device const int* idx [[buffer(6)]],
                         device int* flags [[buffer(7)]],
                         device const int* bat [[buffer(8)]],
                         uint gid [[thread_position_in_grid]]) {
  BrParams P = pp;
  if ((int)gid >= P.nActive) return;
  int r = idx[gid];
  device float* g = state + (ulong)r * (ulong)P.stride;
  BrScen sc = scen[bat[r * 3 + 2]];
  br_sw_run(g, &P, &sc, wA + (ulong)bat[r * 3] * (ulong)P.attNw, wD + (ulong)bat[r * 3 + 1] * (ulong)P.defNw, P.chunk);
  int a = g[SWH_DONE] == 0.0f;
  flags[r] = a;
  if (a) atomic_fetch_add_explicit(alive, 1u, memory_order_relaxed);
}

#endif

#ifdef BR_SW_LOCAL
// Swarm, large battles: one threadgroup per battle, one thread per ball (looping when there are more balls than
// threads). The battle is copied into threadgroup memory (BR_SG = threadgroup, SW_STRIDE floats), flown there for
// `chunk` steps with barriers between phases, then copied back. A commander side's network is split across the
// threads: each computes whole neurons of a layer, in the same order as br_sw_forward, so results do not change.
kernel void br_sw_group_kernel(device float* state [[buffer(0)]],
                               device const float* wA [[buffer(1)]],
                               device const float* wD [[buffer(2)]],
                               device const BrScen* scen [[buffer(3)]],
                               constant BrParams& pp [[buffer(4)]],
                               device atomic_uint* alive [[buffer(5)]],
                               device const int* idx [[buffer(6)]],
                               device int* flags [[buffer(7)]],
                               device const int* bat [[buffer(8)]],
                               uint grp [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint tpg [[threads_per_threadgroup]]) {
  threadgroup float g[SW_STRIDE];
  threadgroup float ta[SW_MAXW], tb[SW_MAXW];
  BrParams P = pp;
  if ((int)grp >= P.nActive) return;   // uniform for the whole group
  int r = idx[grp], nA = P.attN, nB = nA + P.defN, T = (int)tid, N = (int)tpg;
  device float* gg = state + (ulong)r * (ulong)P.stride;
  for (int i = T; i < P.stride; i += N) g[i] = gg[i];
  BrScen sc = scen[bat[r * 3 + 2]];
  device const float* A = wA + (ulong)bat[r * 3] * (ulong)P.attNw; device const float* D = wD + (ulong)bat[r * 3 + 1] * (ulong)P.defNw;
  BrParams PA = P; PA.thrust = P.thrustAtk;
  uint seed = (uint)sc.seed; float nz = P.noise * 1.732f;
  float a[SW_MAXW], b[SW_MAXW];
  for (int it = 0; it < P.chunk; it++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (g[SWH_DONE] != 0) break;
    int k = (int)g[SWH_K];
    if (k % P.ctrl == 0) {
      if (T == 0) br_sw_assign(g, &P);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int side = 0; side < 2; side++) {
        if (!(side ? P.defAI && P.defCmd : P.attAI && P.attCmd)) continue;
        int nin = side ? P.defNin : P.attNin, nout = side ? P.defNout : P.attNout; device const float* w = side ? D : A;
        if (T == 0) { br_sw_cmd_inputs(g, &P, &sc, side, seed, (uint)k, nz, a); for (int i = 0; i < nin; i++) ta[i] = a[i]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* src = ta; threadgroup float* dst = tb; int off = 0;
        for (int l = 0; l < P.nl; l++) {
          int ni = l == 0 ? nin : P.arch[l], no = l == P.nl - 1 ? nout : P.arch[l + 1];
          device const float* W = w + off; device const float* Bv = w + off + no * ni;
          for (int j = T; j < no; j += N) { float acc = Bv[j]; device const float* row = W + j * ni; for (int i = 0; i < ni; i++) acc += row[i] * src[i]; dst[j] = TANH(acc); }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          off += no * ni + no; threadgroup float* t2 = src; src = dst; dst = t2;
        }
        int o0 = side ? nA : 0, o1 = side ? nB : nA;
        for (int bi = o0 + T; bi < o1; bi += N) { threadgroup float* q = g + SW_H + bi * SW_B; int j = (bi - o0) * 3;
          q[S_U0] = (src[j] + 1) * 0.5f; q[S_U1] = src[j + 1]; q[S_U2] = src[j + 2]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      for (int bi = T; bi < nB; bi += N) br_sw_decide_ball(g, &P, &PA, &sc, A, D, bi, seed, k, nz, a, b);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int bi = T; bi < nB; bi += N) br_sw_move_ball(g, &P, &PA, &sc, bi);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int d = nA + T; d < nB; d += N) br_sw_scan(g, &P, d);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (T == 0) br_sw_resolve(g, &P, &sc);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = T; i < P.stride; i += N) gg[i] = g[i];
  if (T == 0) { int a2 = g[SWH_DONE] == 0.0f; flags[r] = a2; if (a2) atomic_fetch_add_explicit(alive, 1u, memory_order_relaxed); }
}
#endif
