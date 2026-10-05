# Interacting Balls

Native trainer for ball brains that fly thrust-vectored balls. Three modes:
- **Reach target**: fly from the pad to a goal 1.5 km to a set max distance (up to 100 km) away.
- **Intercept**: a red runner ball, flown by a guidance algorithm, launches 4–100 km out and heads for a defended
  point; your ball launches nearby (when its radar sees the runner) and must touch it (2 m), or get within the
  blast radius, before it gets there.
- **Swarm**: 1–32 attackers fly at a defended point while 1–32 defenders try to catch them (a catch removes both).
  Each side is flown by the guidance algorithm or by an AI brain: **Nearest-K** (every ball runs its own copy of one
  small network that sees its K nearest enemies and teammates; works at any swarm size) or **Commander** (one network
  steers the whole side). AI vs AI trains both sides together (self-play).

Training, auto-tune and flight playback run natively; a local window shows settings, live stats and a 3D range.

## Run
Open the app (`Ball Arena Trainer.app` on Mac, `BallArenaTrainer.exe` on Windows), press
**Auto-tune for this computer**, then **Train**. Watch **Validation** (the star on a fixed set of 64 scenarios it never trains on).
Each mode keeps its own star ball: `~/BallArena-star-reach.json` and `~/BallArena-star-tag.json`; swarm stars are kept per side
and matchup (`~/BallArena-star-swarm-nk-K2-defend.json`, `~/BallArena-star-swarm-cmd-A10-D3-attack.json`, …).

## Compute
| Backend | Hardware |
|---|---|
| CPU | all cores, auto-vectorised C |
| Metal | Apple GPUs |
| OpenCL | AMD/other GPUs (loaded from the driver at runtime) |

All three compile one source, `src/sim_core.h`. Swarm battles on a GPU use one thread per battle (small swarms) or one
work-group per battle with a thread per ball (large swarms); auto-tune measures both.

## Build
```bash
cmake -S . -B build && cmake --build build -j
```
Windows cross-build: see [docs/BUILD.md](docs/BUILD.md). Developer command line: `brtrain --help`
(e.g. `brtrain --train --mode tag --atk-range 25000 --blast 5 --detect 20000 --gens 500`).
Other flags: `--reach-max M`, `--tw X`, `--ceiling` (the guidance algorithm's tag rate on your settings).
Swarm: `brtrain --train --mode swarm --attackers 4 --defenders 4 --att algo --def ai --k 2 --gens 300`; `--ceiling --mode swarm`
(algorithm vs algorithm), `--compare --mode swarm` (the same battles on every backend).
Command-line runs save stars to your home folder like the app: set `HOME` to a scratch folder when testing.


## Security and network notes
- The app's interface listens on `127.0.0.1` only (ports 8642–8661). It answers only requests addressed to this computer
  from its own page, and commands need a header other web pages cannot send, so a web page you visit cannot control it.
  The page loads nothing from the internet (three.js is included, see [THIRD_PARTY.md](THIRD_PARTY.md)).
- Trained networks are saved as JSON files in your home folder (`~/BallArena-star-*.json`).

## License
MIT, see [LICENSE](LICENSE). three.js (MIT) is bundled in `web/vendor/`, see [THIRD_PARTY.md](THIRD_PARTY.md).

## Docs
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): code layout and data flow
- [docs/TRAINING.md](docs/TRAINING.md): settings, auto-tune, tips
- [docs/SWARM_PLAN.md](docs/SWARM_PLAN.md): how the Swarm mode was planned and built (with what changed from the plan)
- [docs/memory-neurons.html](docs/memory-neurons.html), [docs/brain-memory.html](docs/brain-memory.html): interactive explainers
