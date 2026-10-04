// Metal compute kernel: one thread = one flight, advanced `chunk` physics steps per dispatch.
kernel void br_kernel(device float* state [[buffer(0)]],
                      device const float* weights [[buffer(1)]],
                      constant BrScen* scen [[buffer(2)]],
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

// Swarm, large battles: one threadgroup per battle, one thread per ball (looping when there are more balls than
// threads); phases separated by barriers, single-thread phases on thread 0. Same battle as br_sw_run.
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
  BrParams P = pp;
  if ((int)grp >= P.nActive) return;   // uniform for the whole group
  int r = idx[grp], nB = P.attN + P.defN;
  device float* g = state + (ulong)r * (ulong)P.stride;
  BrScen sc = scen[bat[r * 3 + 2]];
  device const float* A = wA + (ulong)bat[r * 3] * (ulong)P.attNw; device const float* D = wD + (ulong)bat[r * 3 + 1] * (ulong)P.defNw;
  BrParams PA = P; PA.thrust = P.thrustAtk;
  uint seed = (uint)sc.seed; float nz = P.noise * 1.732f;
  float a[SW_MAXW], b[SW_MAXW];
  for (int it = 0; it < P.chunk; it++) {
    threadgroup_barrier(mem_flags::mem_device);
    if (g[SWH_DONE] != 0) break;
    int k = (int)g[SWH_K];
    if (k % P.ctrl == 0) {
      if (tid == 0) br_sw_assign(g, &P);
      threadgroup_barrier(mem_flags::mem_device);
      if (tid < 2) br_sw_decide_cmd(g, &P, &sc, A, D, (int)tid, seed, k, nz, a, b);
      for (int bi = (int)tid; bi < nB; bi += (int)tpg) br_sw_decide_ball(g, &P, &PA, &sc, A, D, bi, seed, k, nz, a, b);
      threadgroup_barrier(mem_flags::mem_device);
    }
    for (int bi = (int)tid; bi < nB; bi += (int)tpg) br_sw_move_ball(g, &P, &PA, &sc, bi);
    threadgroup_barrier(mem_flags::mem_device);
    for (int d = P.attN + (int)tid; d < nB; d += (int)tpg) br_sw_scan(g, &P, d);
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) br_sw_resolve(g, &P, &sc);
  }
  threadgroup_barrier(mem_flags::mem_device);
  if (tid == 0) { int a2 = g[SWH_DONE] == 0.0f; flags[r] = a2; if (a2) atomic_fetch_add_explicit(alive, 1u, memory_order_relaxed); }
}
