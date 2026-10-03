# Interacting Balls

Native trainer for a neural pilot that flies a thrust-vectored ball to a ground target. Training, auto-tune and
flight playback run natively; a local window shows settings, live stats and a 3D range.

## Run
Open the app (`Ballistic Range Trainer.app` on Mac, `BallisticRangeTrainer.exe` on Windows), press
**Auto-tune for this computer**, then **Train**. Watch **Validation** (champion on 64 fresh scenarios).
The champion autosaves to `~/BallisticRange-champion.json`.

## Compute
| Backend | Hardware |
|---|---|
| CPU | all cores, auto-vectorised C |
| Metal | Apple GPUs |
| OpenCL | AMD/other GPUs (loaded from the driver at runtime) |

All three compile one source, `src/sim_core.h`.

## Build
```bash
cmake -S . -B build && cmake --build build -j
```
Windows cross-build: see [docs/BUILD.md](docs/BUILD.md).

## Browser version
[browser/ballistic-range.html](browser/ballistic-range.html): the original single-page sandbox with Reach target and Intercept modes, in-browser training (CPU workers, WebAssembly, WebGL). Open it in a browser.

## Docs
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): code layout and data flow
- [docs/TRAINING.md](docs/TRAINING.md): settings, auto-tune, tips
- [docs/memory-neurons.html](docs/memory-neurons.html), [docs/pilot-memory.html](docs/pilot-memory.html): interactive explainers
