# Interacting Balls

Native trainer for a ball brain that flies a thrust-vectored ball. Two modes:
- **Reach target**: fly from the pad to a goal 1.5 km to a set max distance (up to 100 km) away.
- **Intercept**: a red runner ball, flown by a guidance algorithm, launches 4–100 km out and heads for a defended
  point; your ball launches nearby (when its radar sees the runner) and must touch it (2 m), or get within the
  blast radius, before it gets there.

Training, auto-tune and flight playback run natively; a local window shows settings, live stats and a 3D range.

## Run
Open the app (`Ball Arena Trainer.app` on Mac, `BallArenaTrainer.exe` on Windows), press
**Auto-tune for this computer**, then **Train**. Watch **Validation** (the star on a fixed set of 64 scenarios it never trains on).
Each mode keeps its own star ball: `~/BallArena-star-reach.json` and `~/BallArena-star-tag.json`.

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
Windows cross-build: see [docs/BUILD.md](docs/BUILD.md). Developer command line: `brtrain --help`
(e.g. `brtrain --train --mode tag --atk-range 25000 --blast 5 --detect 20000 --gens 500`).
Other flags: `--reach-max M`, `--tw X`, `--ceiling` (the guidance algorithm's tag rate on your settings).
Command-line runs save stars to your home folder like the app: set `HOME` to a scratch folder when testing.

Next planned feature: a Swarm mode, see [docs/SWARM_PLAN.md](docs/SWARM_PLAN.md).

## Docs
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): code layout and data flow
- [docs/TRAINING.md](docs/TRAINING.md): settings, auto-tune, tips
- [docs/SWARM_PLAN.md](docs/SWARM_PLAN.md): build plan for the Swarm mode (attackers vs defenders)
- [docs/memory-neurons.html](docs/memory-neurons.html), [docs/brain-memory.html](docs/brain-memory.html): interactive explainers
