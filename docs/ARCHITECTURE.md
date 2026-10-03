# Architecture

| File | Role |
|---|---|
| `src/sim_core.h` | Physics, 29 sensor inputs, past frames, memory neurons, network. Compiles as C, Metal and OpenCL. |
| `src/backend_cpu.c` | Threaded CPU evaluation (transposed weights for vectorisation). |
| `src/backend_metal.m` | Metal host: chunked dispatches, compaction of finished flights. |
| `src/backend_opencl.c`, `src/br_cl.h` | OpenCL host; library loaded at runtime, no SDK needed. |
| `src/trainer.c` | Training service: CMA-ES / GA, islands, validation, autosave, auto-tune, flight playback. |
| `src/server.c` | Local HTTP server (127.0.0.1 only, Host-checked) and app window launch. |
| `web/index.html` | UI. Displays only; never simulates. |
| `cmake/embed.cmake` | Embeds kernels and UI into the binary. |

**Flow:** UI → `/api/start` → training thread → backend `eval` (genomes × scenarios) → fitness → optimizer.
UI polls `/api/status`; `/api/fly` returns champion paths for the 3D view.

**Physics:** 105 kg ball, thrust-to-weight setting, drag with transonic rise, lift, weathercock torque,
gimbal + fin + RCS control. Starts 10 m up, pointing straight up. Hit = within 30 m of the target.

**Fitness:** `−ln(max(closest, 30 m)/30 m)`; hit adds `+5` (and an optional speed bonus).
