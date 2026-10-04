# Architecture

| File | Role |
|---|---|
| `src/sim_core.h` | Physics, 29 sensor inputs, past frames, memory neurons, network, the guidance algorithm (`br_algo_control`, runner weave), swarm battles (`br_sw_*`). Compiles as C, Metal and OpenCL. |
| `src/backend_cpu.c` | Threaded CPU evaluation (transposed weights for vectorisation). |
| `src/backend_metal.m` | Metal host: chunked dispatches, compaction of finished flights. |
| `src/backend_opencl.c`, `src/br_cl.h` | OpenCL host; library loaded at runtime, no SDK needed. |
| `src/trainer.c` | Training service: scenarios (reach goals; tag setups with recorded runner paths; swarm battle setups), CMA-ES / GA, islands, validation, per-mode autosave, swarm self-play, auto-tune, flight and battle playback. |
| `src/server.c` | Local HTTP server (127.0.0.1 only, Host-checked) and app window launch. |
| `web/index.html` | UI. Displays only; never simulates. |
| `cmake/embed.cmake` | Embeds kernels and UI into the binary. |

**Flow:** UI → `/api/start` → training thread → backend `eval` (genomes × scenarios) → fitness → optimizer.
UI polls `/api/status`; `/api/fly` returns star paths for the 3D view (tag: plus the runner path and launch sites,
sampled at the same steps so they replay in lockstep); `/api/mode` switches mode and loads that mode's star.
**Playback:** each press launches from a new spot on the range (random offset up to 15 km; the network only sees the goal relative to itself), always with the star's own training physics. Fly 1 locks a chase camera to the ball for the whole flight. Tag replays also return the radar range, drawn as a ring, dome and rotating sweep around your pad.

**Physics:** 105 kg ball, thrust-to-weight setting, drag with transonic rise, lift, weathercock torque,
gimbal + fin + RCS control. Starts 10 m up, pointing straight up. Hit = within 30 m of the target. Reach goals are 1.5 km to the max goal distance (2–100 km); the flight time limit grows with it.

**Intercept (tag) mode:** each scenario first flies the runner on the CPU (guidance algorithm, its own
thrust-to-weight, optional sine weave) and records its path (position + velocity per physics step). All of a
generation's paths are packed into one buffer that every backend uploads; each scenario holds its offset and length.
The chaser's sensors see the runner `delay` steps late plus uniform noise (stateless hash, so CPU and GPU draw the
same noise). Radar range: the chaser waits on its pad until the runner is within range (0 = launch at once). A tag is
the balls' centres within max(2 m, blast radius) at any moment of a step (swept, so fast passes can't tunnel); once the gap starts
widening after closing in, the chaser has passed and the flight ends. Scenario layout: runner launches
range–(range + 1.5 km) from a defended point 0–3 km from the origin; chaser 0.5–3 km from it, at least 0.7·range from
the runner's site.Every setup is kept (no reachability filter). Runner paths are built in parallel on all cores.

**Fitness:** `−ln(max(closest, floor)/30 m)` with floor 30 m (reach) or the catch distance (tag); a hit adds `+5` (and an optional
speed bonus, else −0.01·t). Tag: −2 when the runner reached its defended point. Optional altitude reward: up to +0.5 when the ball stays
above 5 m through mid-flight (the backends report, per flight, the fraction of mid-flight steps below 5 m).

**Flight results:** `BR_OUT` = 5 floats per flight: closest approach, hit, flight time, physics steps, low fraction. Validation uses a fixed seed, so
its hit rate is comparable between generations.

**Swarm mode:** a battle is one float block: a header (step, catches, leaks, done, ball-steps) and `SW_B` floats per
ball (the per-flight `S_*` fields plus target, weave, launched, fate, start-of-step position, nearest attacker).
Attackers are balls `0 .. attN−1`, defenders follow. Start positions are built on the CPU (`sw_gen_scen`: attackers on the
ground between the minimum and maximum launch range, from any direction; defenders 0.5–3 km from the defended point) and passed in a
buffer, so every backend starts from identical states. Each step runs in phases:
1. **decide** (every `ctrl` steps): defenders are assigned targets (nearest unclaimed live attacker, in index order);
   commander sides run one network pass for the whole side; other balls run nearest-K networks (29 own-target sensors +
   7 features for each of K nearest enemies and K nearest teammates) or the algorithm (attackers: `br_evader_control`
   toward the defended point; defenders: `br_algo_control` toward their target);
2. **move**: `br_move` (the same physics as flights; attackers use the runner thrust);
3. **scan**: each defender's nearest live attacker during the step (swept, so fast passes can't tunnel);
4. **resolve** (one thread): catches in defender index order (within max(2 m, blast radius): both removed), leaks
   (attacker within 30 m of the defended point), crashes, radar launches for the next step, the end of the battle.

The CPU runs the phases in order (one battle per core). GPUs either give each battle one thread (`br_sw_kernel`, best for
small swarms) or a work-group with a thread per ball and barriers between phases (`br_sw_group_kernel`, best from about
12 balls); both produce the same battle. Swarm networks have no past frames or memory. Sensors report the present plus
noise (no delay). Per-battle results (`BR_SWOUT`): catches, leaks, closeness scores for each side, ball-steps, time,
crashes.

**Swarm training:** one CMA-ES per AI side. AI vs algorithm trains against the algorithm; AI vs AI trains each side
against a pool of the other side's last 8 stars (refreshed every 5 generations). Scores, normalised by the counts:
defenders `(5·catches − 5·leaks)/attackers + 0.5·closeness/defenders`; attackers `(5·leaks − 2·losses)/attackers +
0.5·closeness/attackers`. Validation flies each AI star against the algorithm on 64 fixed battles. `/api/battle` flies
one battle natively (each side's star with its saved physics) and returns every ball's path and the catch/leak events.
