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
