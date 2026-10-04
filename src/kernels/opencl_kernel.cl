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

// Swarm, large battles: one work-group per battle, one work-item per ball (looping when there are more balls than
// work-items); phases separated by barriers, single-thread phases on item 0. Same battle as br_sw_run.
__kernel void br_sw_group_kernel(__global float* state, __global const float* wA, __global const float* wD, __global const BrScen* scen,
                                 __constant BrParams* pp, __global volatile uint* alive, __global const int* idx, __global int* flags,
                                 __global const int* bat) {
  BrParams P = *pp;
  int grp = (int)get_group_id(0), tid = (int)get_local_id(0), tpg = (int)get_local_size(0);
  if (grp >= P.nActive) return;   // uniform for the whole group
  int r = idx[grp], nB = P.attN + P.defN;
  __global float* g = state + (size_t)r * (size_t)P.stride;
  BrScen sc = scen[bat[r * 3 + 2]];
  __global const float* A = wA + (size_t)bat[r * 3] * (size_t)P.attNw; __global const float* D = wD + (size_t)bat[r * 3 + 1] * (size_t)P.defNw;
  BrParams PA = P; PA.thrust = P.thrustAtk;
  uint seed = (uint)sc.seed; float nz = P.noise * 1.732f;
  float a[SW_MAXW], b[SW_MAXW];
  for (int it = 0; it < P.chunk; it++) {
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (g[SWH_DONE] != 0) break;
    int k = (int)g[SWH_K];
    if (k % P.ctrl == 0) {
      if (tid == 0) br_sw_assign(g, &P);
      barrier(CLK_GLOBAL_MEM_FENCE);
      if (tid < 2) br_sw_decide_cmd(g, &P, &sc, A, D, tid, seed, k, nz, a, b);
      for (int bi = tid; bi < nB; bi += tpg) br_sw_decide_ball(g, &P, &PA, &sc, A, D, bi, seed, k, nz, a, b);
      barrier(CLK_GLOBAL_MEM_FENCE);
    }
    for (int bi = tid; bi < nB; bi += tpg) br_sw_move_ball(g, &P, &PA, &sc, bi);
    barrier(CLK_GLOBAL_MEM_FENCE);
    for (int d = P.attN + tid; d < nB; d += tpg) br_sw_scan(g, &P, d);
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (tid == 0) br_sw_resolve(g, &P, &sc);
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (tid == 0) { int a2 = g[SWH_DONE] == 0.0f; flags[r] = a2; if (a2) atomic_inc(alive); }
}
