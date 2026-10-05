# Training

## Auto-tune
Measures every compute option on the machine: GPU work-group size (32–512), dispatch length (20/40/80 ms) and batch
size until throughput plateaus. Picks the option with the most **generations per second** at a useful population
(512 networks for CMA-ES, 256 for GA), not the highest raw steps/s: a bigger population improves each generation only
slightly while making it slower. Spare capacity (bigger batches costing ≤25% more time per generation) then goes to:
- Islands on: more islands of ~48 networks (CMA-ES) or ~32 (GA).
- Islands off: more scenarios per network (steadier scores).

Speed is measured; population/island sizing is rule-based. In intercept mode the CPU time to build the scenarios
(flying every runner) is included in every timing, spread over the generations that reuse
them.

## Settings
| Setting | Effect |
|---|---|
| CMA-ES / GA | CMA-ES adapts per-weight step sizes (default). GA needs decay. |
| CMA-ES variant | Automatic: sep-CMA-ES up to 2,000 weights, LM-MA-ES above (`--opt sep|lm`). Both draw mirrored pairs (z, −z). |
| Head start | A new network first copies the guidance algorithm (behaviour cloning, 3 DAgger rounds), then evolves (`--no-imitate`). |
| Automatic difficulty | A new network starts on an easy version of the settings and steps up a level (of 10) when validation reaches the threshold twice in a row (`--auto-diff`). |
| Restart when stuck | No validation or training-score progress for 6 checks: restart around the star, alternating wide/large and fine/small searches (`--no-restarts`). |
| Normalise sensors | Running mean/spread per sensor, folded into the first layer (reach, intercept; `--no-norm`). |
| Islands | Parallel populations swapping their best every N generations. |
| Scenarios / reuse | 16–32 scenarios, kept 5–10 generations, for fair comparisons. |
| Past frames K, memory neurons | History for noisy tasks. Change the network shape (fresh network). |
| Physics step, decide every step | Smaller step = finer control, slower training. |
| Reward staying above 5 m | Optional: up to +0.5 for flying above 5 m through mid-flight (from 3 s after launch until 300 m from the target); 0 at half the time, full at 80%. Discourages skimming the ground. |
| Max goal distance (reach) | Training goals are 1.5 km to this far (2–100 km). Raise it in steps; the flight time limit grows with it. |
| Thrust-to-weight | 1.2–20. High thrust needs a small physics step (5 ms) to learn well. |
| Mode | Reach target or Intercept. Each mode trains and saves its own network; a saved network only loads in its own mode. |
| Blast radius | 0–10 m. Off: the balls must touch (2 m). On: getting this close counts as a catch, like a proximity warhead. |
| Runner launch range | 4–100 km. Longer runs take longer to fly and simulate. |
| Runner weaving | Sine weaves on the runner's steering after 6 s (0–1). |
| Sensor noise / delay | What your ball's sensors see of the runner: uniform noise σ in metres, and a delay in ms. |
| Radar range | 0–100 km. Your ball waits on its pad until its radar sees the runner within this range (0 = launch at once). Short ranges make many setups impossible (see `--ceiling`). |
| Runner thrust-to-weight | The runner's engine, separate from yours (swarm: the attackers'). |
| Swarm: Algorithm / AI | Per side. AI vs AI trains both together (self-play); Algorithm vs Algorithm only shows battles. |
| Swarm: Nearest-K / Commander | Nearest-K: one small brain per ball, sees K nearest enemies and K nearest teammates, works at any count. Commander: one brain steers the whole side and is saved per attacker/defender count. |
| Swarm: launch range | Attackers launch anywhere between the minimum and maximum distance, from any direction. |
| Swarm: counts, K | 1–32 attackers and defenders; K = 1–8 (more context per ball, more weights). |

## Tips
- Watch Validation, not Star hits (best-of-population is often lucky).
- Decay ×0.995 with a 1e-4 floor; ×0.9 freezes training.
- Turn on speed reward only after it hits reliably.
- The replay (Fly 1 / Fly 5) always flies the star with the physics it was trained with (time step, thrust-to-weight, control rate), whatever the panel says. Training uses the panel's physics: change them and the star relearns.
- Intercept: `brtrain --ceiling --mode tag ...` measures how often the guidance algorithm (perfect sensors) tags on your settings: a practical ceiling.
- Intercept: start with no noise, delay or weaving and a short range (7 km); add difficulty once Validation is high.
- Swarm: start with AI defenders vs algorithm attackers, 3 vs 3, Nearest-K with K = 2, 7 km and no radar delay; once
  the validation catch rate is high, add attackers, radar range or weaving. Nearest-K brains carry over to other counts.
- Swarm AI vs AI: opponents are drawn from the other side's last 16 stars, weighted toward the ones this side still
  loses to (prioritised fictitious self-play).
- New networks: keep the head start on; with automatic difficulty on, set your real target and let it climb.
- Swarm AI vs AI: watch the validation rates (each side against the algorithm), not the training scores, which move as
  the opponents change. Commander brains have many more weights and learn much more slowly.
- Swarm auto-tune measures battles per generation on every backend and both GPU layouts with your current matchup.

The ideas behind these, with the maths: [docs/explainers](explainers/README.md).
