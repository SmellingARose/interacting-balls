# Training

## Auto-tune
Measures every compute option on the machine: GPU work-group size (32–512), dispatch length (20/40/80 ms) and batch
size until throughput plateaus. Picks the option with the most **generations per second** at a useful population
(512 networks for CMA-ES, 256 for GA), not the highest raw steps/s: a bigger population improves each generation only
slightly while making it slower. Spare capacity (bigger batches costing ≤25% more time per generation) then goes to:
- Islands on: more islands of ~48 networks (CMA-ES) or ~32 (GA).
- Islands off: more scenarios per network (steadier scores).

Speed is measured; population/island sizing is rule-based. In intercept mode the CPU time to build the scenarios
(flying every runner and the reachability check) is included in every timing, spread over the generations that reuse
them.

## Settings
| Setting | Effect |
|---|---|
| CMA-ES / GA | CMA-ES adapts per-weight step sizes (default). GA needs decay. |
| Islands | Parallel populations swapping their best every N generations. |
| Scenarios / reuse | 16–32 scenarios, kept 5–10 generations, for fair comparisons. |
| Past frames K, memory neurons | History for noisy tasks. Change the network shape (fresh network). |
| Physics step, decide every step | Smaller step = finer control, slower training. |
| Mode | Reach target or Intercept. Each mode trains and saves its own network; a saved network only loads in its own mode. |
| Only reachable tags | Drops setups the algorithm itself cannot reach with perfect sensors (they only add noise). Keep on. |
| Runner launch range | 4–100 km. Longer runs take longer to fly and simulate. |
| Runner weaving | Sine weaves on the runner's steering after 6 s (0–1). |
| Sensor noise / delay | What your ball's sensors see of the runner: uniform noise σ in metres, and a delay in ms. |
| Launch on detection | Your ball waits on its pad until the runner is this close (0 = launch at once). |
| Runner thrust-to-weight | The runner's engine, separate from yours. |

## Tips
- Watch Validation, not Star hits (best-of-population is often lucky).
- Decay ×0.995 with a 1e-4 floor; ×0.9 freezes training.
- Turn on speed reward only after it hits reliably.
- Intercept: start with no noise, delay or weaving and a short range (7 km); add difficulty once Validation is high.
