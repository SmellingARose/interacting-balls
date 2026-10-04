# Swarm mode: build plan

> **Status: built** (October 2026, commits `775ab28` … Phase 6). The plan below is kept as written; these are the
> places where the build differs from it, and why:
>
> | Plan | What was built | Why |
> |---|---|---|
> | Phase 0/1: Reach and Intercept results byte-identical | Hits, CPU/GPU agreement and guidance ceilings unchanged; float sums differ in the last digits (one ceiling moved by 1 flight in 400) | Splitting the ball physics into `br_move` (shared with battles) changes how the compiler orders float operations. Keeping two copies of the physics was the alternative. |
> | Step 17: move the single population into `Side side[2]` | Swarm has its own `SwSide` state; Reach/Intercept training is untouched | No risk to the existing modes |
> | Swarm networks | No past frames, memory neurons or sensor delay in swarm | Each ball already sees its neighbours (or the whole battle) every step; delay would need a position history per ball |
> | Radar | A defender launches one step after its radar sees an attacker | Radar is checked after every ball has moved, so no GPU thread reads a position another is writing |
> | Step 35, layout B | Built for nearest-K and algorithm sides; a commander pass runs on one thread, so commander battles use layout A | Splitting a commander's layers across threads was not needed for speed at the measured sizes |
> | Step 38: 100% match | 100% for algorithm and random brains; ~99% with trained brains | GPU fast-math flips rare borderline 2 m catches (as in Intercept at high thrust) |
> | Auto backend | Auto-tune measures battles (both GPU layouts and work-group sizes) | On Apple M4 the GPU wins at generation-size batches (3v3: Metal 8,400 battles/s vs CPU 5,600; 16v16: Metal group 720 vs CPU 571) |

A third mode, **Swarm**: N attacker balls fly at a defended point while M defender balls try to catch them. Either
side can be an AI brain or the built-in guidance algorithm, so the app can train AI vs algorithm or AI vs AI
(self-play). Everything stays native (CPU, Metal, OpenCL), compiled from the one shared simulation source.

This document is written for someone with no other context. Work **one phase at a time**: finish a phase, run its
checks, commit, and stop for review before starting the next.

---

## 1. Decisions already made (requirements)

| Topic | Decision |
|---|---|
| Who plays | Per side: **Attackers: AI or Algorithm**, **Defenders: AI or Algorithm**. AI vs AI = both brains train together (self-play). Algorithm vs Algorithm = demo only, nothing trains. |
| Swarm size | Two separate sliders: attacker count 1–32, defender count 1–32. |
| Brain type | Per side, user picks: **Commander** (one network sees every ball and steers every ball on its side; its size depends on the counts) or **Nearest-K** (every ball runs its own copy of one small network that sees itself, the defended point and its K nearest enemies and teammates; works at any swarm size). Sides may use different types. |
| Catch | A defender within max(2 m touch, blast radius) of an attacker catches it. **Both are removed.** |
| Leak | An attacker reaching the defended point (within 30 m, like reach mode) is removed and counts as a leak. |
| End of battle | No attackers left, or time limit. |
| Score shown | Catches and leaks per battle; catch rate and leak rate over the validation set. |
| Replay | A **Launch** button flies one full battle in the 3D view. **View: Attackers / Defenders** picks which side the camera follows; ◀ ▶ (and `[` `]`) cycle the followed ball. |
| Reused settings | Blast radius, radar range (defenders wait until an attacker is in range), launch range, weaving (algorithm attackers), sensor noise/delay, thrust-to-weight (both sides), runner thrust-to-weight (attackers), physics step. |
| Speed | Must stay fast and native. CPU first; GPU after the CPU version is the reference. |

**Adjustable, not fixed:** the scoring weights in step 19 and the GPU layout B design in step 35 are starting points.
Change them if tests show something better, and note why in the commit.

---

## 2. Ground rules

- **Never touch the user's saved brains.** The app and the CLI save stars to `$HOME/BallArena-star-*.json`. Run every
  test with a scratch home: `HOME=/path/to/scratch ./build/brtrain ...`. Before anything that might write to the real
  home folder, copy the star files somewhere safe that is **not** a temp folder.
- **Don't change Reach or Intercept behaviour.** Their results must stay byte-identical (see the Phase 0 baseline).
- **One source for physics.** All simulation code lives in `src/sim_core.h`, written in the subset that compiles as C,
  Metal and OpenCL (use the `FN`, `BR_GP`, `BR_PP`, `SQRT`… macros at the top; only 4-byte fields in shared structs;
  no recursion, no dynamic allocation, no doubles).
- **Determinism.** Anything with ordering (catch resolution, nearest-K ties) is resolved by ball index, so CPU, Metal
  and OpenCL give the same results.
- Match the surrounding code style: dense C with short comments explaining *why*.

## 3. Build, run, test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```
- `web/index.html` and the kernels are **embedded into the binary** (`cmake/embed.cmake`): rebuild after editing them.
- `./build/brtrain` starts the app: an HTTP server on `127.0.0.1:8642` and a window. If one is already running, a
  second launch only reopens its window. Stop it with `curl -X POST http://127.0.0.1:8642/api/quit` (it autosaves).
  `--no-window` starts it without a window.
- Headless: `brtrain --train --mode reach|tag --gens N ...`, `--bench`, `--compare` (CPU vs Metal vs OpenCL; prints
  "same hit/miss as CPU: n/N"), `--ceiling` (guidance algorithm's tag rate), `--help`. CLI flags are parsed in
  `cli_cfg()` in `src/trainer.c`; `src/main.c` lists which flags run headless.

## 4. How the current code is shaped (what the plan works around)

| Fact | Consequence |
|---|---|
| `sim_core.h` simulates **one ball**: state is a flat float array indexed by `S_PX … S_HIST`; `br_run()` steps one flight against one target. | Swarm needs a battle-level state and step function. |
| Intercept's runner is a **pre-recorded path** (`traj` buffer built on the CPU by `br_make_attack`). | Swarm attackers must be simulated live, because they react to defenders. |
| `Backend.eval()` (`src/backend.h`) returns `BR_OUT` = 5 floats per genome × scenario. | Battles need a second entry point with per-battle results. |
| "Mode" is treated as a boolean in many places: `c->mode != 0` in `cfg_from_json`, `T.starMode ? "tag" : "reach"`, `trainer_set_mode` (`mode = mode != 0`), `save_path_mode`, `fitness()`, `sim_core.h` `P->mode ?`, the UI's `!!S.mode`. | Phase 0 makes it a real 3-way value first. |
| The trainer holds **one population and one star** (`Island* isl`, `T.star`, `starArch`, `starK`, `starMem`, `starDt`…). | Self-play needs two of each. |
| GPU backends run **one thread per flight**, `chunk` steps per dispatch, compacting finished flights (`src/backend_metal.m`, `src/backend_opencl.c`, kernels in `src/kernels/`). | Small battles fit this; large commander battles need a work-group per battle. |

---

## Phase 0: Make mode 3-way (no visible change)

1. Mode becomes 0 = reach, 1 = intercept, 2 = swarm. Update:
   - `src/trainer.c`: `cfg_from_json`, `trainer_set_mode`, `load_mode_star`, `save_path_mode`, `star_json_locked`,
     `load_json_mode`, `fitness()`, `fly_json`, `scen_make`, `setup_params`, `cli_cfg` (`--mode reach|tag|swarm`).
   - `src/sim_core.h`: every `P->mode ?` / `if (P->mode)` becomes `P->mode == 1`.
   - `web/index.html`: `S.mode`, `syncMode`, `setMode`, `MODE_TEXT`, `cfg()`.
2. Swarm (2) may exist but do nothing yet (UI button hidden).

**Checks**
- Before starting, record a baseline with a scratch HOME:
  `--compare` for reach and for `--mode tag`, and `--train --gens 20` for both (fixed seeds make output reproducible;
  compare hits, best fitness and validation, not steps/s).
- After: identical numbers. Commit: "Make mode a 3-way value".

## Phase 1: Swarm simulation on the CPU (`src/sim_core.h`)

4. Constants `BR_MAXA 32`, `BR_MAXD 32`, and a settings struct `BrSwarm` (only 4-byte fields): `attN, defN, attAI,
   defAI, attBrain, defBrain, K, catchR, radar, leakR, twRunner…`.
5. Battle state: one block per battle with each ball's state (same fields as `BrState`), an alive flag per ball, the
   defenders' current assignments, and counters (catches, leaks, closest approaches). Index macros `SW_*`, like `S_*`.
6. `br_sw_init()`: defended point (0–3 km from the origin), attackers spread over a ring at launch range (reuse
   `ring_pt` logic from `trainer.c`), defenders 0.5–3 km from the defended point, wind as in tag mode.
7. `br_sw_step()`: for every live ball, controls (AI or algorithm), then physics, then catches and leaks.
   - Refactor `br_step` so the per-ball physics takes its target from a struct, not from `traj`; Intercept keeps
     calling it the same way.
   - Catch test: reuse the swept closest-approach test from `br_step` (the `if (P->mode)` block), radius
     `catchR = max(2, blast)`.
   - **Deterministic resolution:** go through defenders in index order; each catches its nearest still-alive attacker
     within range; both are removed.
8. Algorithm controls: attackers use `br_evader_control` toward the defended point (with weave). Defenders get
   `br_sw_assign()` (each takes the nearest unclaimed live attacker, re-picks when it dies) then `br_algo_control`.
9. Radar: defenders stay on the pad until any attacker is within `radar` (reuse the waiting logic near
   `sim_core.h:251`).
10. `br_sw_nearest()`: K nearest enemies and teammates with a fixed-size insertion sort; ties broken by index.
11. Sensors:
    - `br_sw_inputs_nk()`: own state, defended point, K nearest enemies and teammates (relative position and
      velocity), with noise and delay like `br_inputs`. Empty slots are zeros plus a "present" flag.
    - `br_sw_inputs_cmd()`: every ball on both sides plus the defended point, fixed order, dead balls zeroed.
12. Networks:
    - Nearest-K: reuse `br_control`'s forward pass per ball.
    - Commander: one bigger forward pass per side per decision; outputs = 3 controls per ball on that side.
    - Commander inputs reach ~64 balls × 12 features: raise `MAXW` or size layer buffers per battle.
13. `br_sw_run()`: battle version of `br_run`; ends when no attackers remain or time runs out; writes per-battle
    results (catches, leaks, attackers' closest approach to the defended point, defenders' closest approach to
    attackers).

**Checks**
- A temporary CLI demo (Phase 3 makes it permanent) flies algorithm vs algorithm battles: 1×1 behaves like
  Intercept; 10×10 catches most attackers with a large radar range; radar 5 km leaks most.
- Phase 0 baseline still identical. Commit.

## Phase 2: Training service (`src/trainer.c`, `src/trainer.h`)

14. `TrainCfg` gains `attN, defN, attAI, defAI, attBrain, defBrain, swK`. Add them to `cfg_defaults`,
    `cfg_from_json`, `cfg_to_json`, `scen_same`, and CLI flags `--attackers N --defenders M --att ai|algo
    --def ai|algo --att-brain cmd|nk --def-brain cmd|nk --k K`.
15. `setup_params` for swarm: each side's network input and output sizes from brain type and counts.
16. `sw_scen_make()`: battle scenarios are only seeds, wind and the defended point (no recorded paths).
17. Two sides: move the single population (`Island* isl`, `nIsl`, `T.star`, `starNw`, `starArch`, `starNl`, `starK`,
    `starMem`, `starDt`, `starTw`, `starEvery`, `starFit`, `starVersion`) into a `Side` struct; `T` gets `side[2]`.
    Reach and Intercept use `side[0]` only and behave exactly as before.
18. Self-play generation (swarm branch of `train_thread`):
    (a) ask each AI side's CMA-ES for candidates; (b) pair each candidate with opponents from the other side's
    **pool** (its last 8 stars) or the algorithm if that side isn't AI; (c) evaluate battles; (d) tell each side its
    scores; (e) every N generations, push each side's star into the other side's pool.
19. `sw_fitness()` (starting weights, adjustable):
    - Attackers: +5 per leak, −2 per attacker lost, plus −ln(max(closest to defended point, 30 m)/30 m).
    - Defenders: +5 per catch, −5 per leak, plus −ln(max(closest to an attacker, catchR)/30 m).
    - Normalise by counts so scores compare across swarm sizes.
20. Validation every 10 generations: each AI star vs **fixed algorithm opponents** on 64 fixed-seed battles, so
    progress is comparable even though the self-play pool changes.
21. Saving: `save_path_swarm(side, brain, attN, defN, K)`:
    - commander: `BallArena-star-swarm-cmd-A{n}-D{m}-{attack|defend}.json`
    - nearest-K: `BallArena-star-swarm-nk-K{k}-{attack|defend}.json` (loads at any swarm size)
    - store each brain's physics (dt, thrust-to-weight, control rate) as the existing stars do; Reset deletes only the
      current matchup's files.
22. `trainer_status_json` adds `swarm: {attBest, defBest, catchRate, leakRate, valCatch, valLeak, histAtt, histDef}`.

**Checks**
- AI defenders vs algorithm attackers, 3×3, nearest-K: validation catch rate rises over 200 generations.
- AI vs AI: both curves move; no crash when changing counts mid-training (a commander resets, nearest-K continues).

## Phase 3: CPU backend (`src/backend.h`, `src/backend_cpu.c`)

23. New entry point:
    `int (*evalBattles)(Backend*, const float* attW, const float* defW, const int* pairs, int nBattles,
    const BrSwScen*, BrParams*, const BrSwarm*, float* out)`. `pairs` maps each battle to an attacker brain and a
    defender brain (for opponent pools); `out` is 6 floats per battle.
24. CPU version: same pattern as `cpu_worker` (atomic counter, one battle per worker), per-thread scratch for the
    battle state.
25. CLI: `--mode swarm` in `--train`, `--bench`, and `--ceiling` (algorithm vs algorithm: baseline catch rate).
    Add `--ceiling` and friends to the headless list in `src/main.c`.

**Checks**
- Measure battles/s and steps/s at 4×4, 10×3 and 32×32 (record them in the commit message).
- Commit: CPU swarm works end to end from the CLI.

## Phase 4: Interface (`web/index.html`, `src/server.c`)

27. Mode buttons: add **Swarm**. New `swarmOpts` panel: per side, AI/Algorithm, Commander/Nearest-K and a count
    slider (1–32); a K slider (1–8, shown only when a side uses nearest-K). Use the existing `bind()` / `check()`
    helpers and add new keys to the `S` defaults.
28. `syncMode()` shows the right panels and hides reach's goal distance and Intercept's panel.
29. `/api/battle` in `server.c` → `battle_json()` in `trainer.c`: flies one battle natively with each side's star (or
    the algorithm), with each brain's saved physics; returns every ball's sampled path (`[x,y,z,throttle]` every
    ~0.05 s, like `fly_json`) plus catch and leak events with times and positions.
30. Watch panel in Swarm mode: Fly 1 / Fly 5 become **Launch**; add **View: Attackers / Defenders** and ◀ ▶ (`[` `]`)
    to cycle the followed ball.
31. 3D:
    - `addBattle()` draws attackers (red) and defenders (blue) with trails, reusing `addFlight` / `addRunner` pieces.
    - Catches: `burst(pos, true)`. Leaks: a red flash and ring at the defended point.
    - Radar ring and sweep at the defenders' base (`addRadar`).
    - Follow cam: reuse the `follow` logic, switching to the next live ball when the followed one dies. Overview
      camera frames the whole battle.
    - Nothing is drawn while 3D is off (the render loop already stops).
32. Training panel: tiles for catch rate, leak rate, attacker best, defender best; the chart draws two lines (attack
    red, defence blue) when both sides are AI.

**Checks**
- In the app (scratch HOME): train 3×3 nearest-K defenders vs algorithm, press Launch, follow both sides, toggle 3D
  off and on, switch modes back and forth; no console errors.
- Commit: Swarm mode usable on the CPU in the app.

## Phase 5: GPU backends (Metal, OpenCL)

34. **Layout A, one battle per thread** (small swarms, about ≤ 4×4): kernels `br_sw_kernel` in
    `src/kernels/metal_kernel.metal` and `src/kernels/opencl_kernel.cl` calling `br_sw_run` in chunks; same chunking
    and finished-battle compaction as `backend_metal.m` (the loop around lines 61–83).
35. **Layout B, one work-group per battle, one thread per ball** (large swarms, commander brains):
    - battle state in threadgroup / `__local` memory;
    - per step: each thread computes its ball's sensors → barrier → nearest-K: each thread runs its own forward pass;
      commander: threads split each layer's neurons (parallel matrix-vector multiply), barrier between layers →
      each thread applies its physics → barrier → thread 0 resolves catches and leaks in index order → barrier.
    - Only the catch step is single-threaded, which keeps results deterministic.
36. `evalBattles` in `backend_metal.m` and `backend_opencl.c` picks A or B from counts and brain type (or what
    auto-tune chose).
37. Auto-tune (`tune_thread`): also measure A vs B and work-group size for swarm, ranked by battles per second.
38. `--compare --mode swarm`: "same catches/leaks as CPU: n/N" per backend.

**Checks**
- 100% match with the CPU at 4×4, 10×3 and 32×32 for both brain types (small float differences can flip rare
  borderline catches at very high thrust; investigate anything above ~1%).
- Record speeds. Commit.

## Phase 6: Finish

40. Docs: `README.md` (three modes, new CLI flags), `docs/ARCHITECTURE.md` (battle simulation, two-sided training,
    GPU layouts A and B), `docs/TRAINING.md` (self-play tips, commander vs nearest-K).
41. Regression: the Phase 0 baseline for Reach and Intercept is still identical.
42. Commit.

---

## Risks

- **Commander size at 32×32** (~770 inputs): a large input layer and many weights make CMA-ES slow to learn.
  Nearest-K will train far faster; that is why both exist.
- **Self-play can stall or cycle.** The opponent pool (step 18) and fixed-algorithm validation (step 20) guard
  against it; watch both curves.
- **GPU layout B** is the most complex part. It comes last, after the CPU version is a working reference.
