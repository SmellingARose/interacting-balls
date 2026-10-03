# Architecture

| File | Role |
|---|---|
| `src/sim_core.h` | Physics, 29 sensor inputs, past frames, memory neurons, network, the guidance algorithm (`br_algo_control`, runner weave). Compiles as C, Metal and OpenCL. |
| `src/backend_cpu.c` | Threaded CPU evaluation (transposed weights for vectorisation). |
| `src/backend_metal.m` | Metal host: chunked dispatches, compaction of finished flights. |
| `src/backend_opencl.c`, `src/br_cl.h` | OpenCL host; library loaded at runtime, no SDK needed. |
| `src/trainer.c` | Training service: scenarios (reach goals; tag setups with recorded runner paths), CMA-ES / GA, islands, validation, per-mode autosave, auto-tune, flight playback. |
| `src/server.c` | Local HTTP server (127.0.0.1 only, Host-checked) and app window launch. |
| `web/index.html` | UI. Displays only; never simulates. |
| `cmake/embed.cmake` | Embeds kernels and UI into the binary. |

**Flow:** UI → `/api/start` → training thread → backend `eval` (genomes × scenarios) → fitness → optimizer.
UI polls `/api/status`; `/api/fly` returns star paths for the 3D view (tag: plus the runner path and launch sites,
sampled at the same steps so they replay in lockstep); `/api/mode` switches mode and loads that mode's star.

**Physics:** 105 kg ball, thrust-to-weight setting, drag with transonic rise, lift, weathercock torque,
gimbal + fin + RCS control. Starts 10 m up, pointing straight up. Hit = within 30 m of the target.

**Intercept (tag) mode:** each scenario first flies the runner on the CPU (guidance algorithm, its own
thrust-to-weight, optional sine weave) and records its path (position + velocity per physics step). All of a
generation's paths are packed into one buffer that every backend uploads; each scenario holds its offset and length.
The chaser's sensors see the runner `delay` steps late plus uniform noise (stateless hash, so CPU and GPU draw the
same noise). Optional launch on detection: the chaser waits on its pad until the runner is within range. A tag is
the balls' centres within 2 m at any moment of a step (swept, so fast passes can't tunnel); once the gap starts
widening after closing in, the chaser has passed and the flight ends. Scenario layout: runner launches
range–(range + 1.5 km) from a defended point 0–3 km from the origin; chaser 0.5–3 km from it, at least 0.7·range from
the runner's site. "Only reachable tags" keeps a setup only if the algorithm as chaser, with perfect sensors, passes
within 20 m (up to 30 tries). Runner paths are built in parallel on all cores.

**Fitness:** `−ln(max(closest, floor)/30 m)` with floor 30 m (reach) or 2 m (tag); a hit adds `+5` (and an optional
speed bonus, else −0.01·t). Tag: −2 when the runner reached its defended point. Validation uses a fixed seed, so
its hit rate is comparable between generations.
