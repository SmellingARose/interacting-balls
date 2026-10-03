// Metal compute kernel: one thread = one flight, advanced `chunk` physics steps per dispatch.
kernel void br_kernel(device float* state [[buffer(0)]],
                      device const float* weights [[buffer(1)]],
                      constant BrScen* scen [[buffer(2)]],
                      constant BrParams& pp [[buffer(3)]],
                      device atomic_uint* alive [[buffer(4)]],
                      device const int* idx [[buffer(5)]],
                      device int* flags [[buffer(6)]],
                      uint gid [[thread_position_in_grid]]) {
  BrParams P = pp;
  if ((int)gid >= P.nActive) return;
  int r = idx[gid];                       // only flights still in the air are launched (keeps SIMD groups full)
  device float* g = state + (ulong)r * (ulong)P.stride;
  int gi = r / P.S, si = r % P.S;
  BrScen sc = scen[si];
  br_run(g, &P, &sc, weights + (ulong)gi * (ulong)P.nw, P.chunk);
  int a = g[S_ALIVE] != 0.0f;
  flags[r] = a;
  if (a) atomic_fetch_add_explicit(alive, 1u, memory_order_relaxed);
}
