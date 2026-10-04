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
