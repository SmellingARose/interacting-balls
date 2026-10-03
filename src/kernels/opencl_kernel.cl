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
