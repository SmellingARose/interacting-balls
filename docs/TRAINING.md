# Training

## Auto-tune
Measures every compute option on the machine: GPU work-group size (32–512), dispatch length (20/40/80 ms) and batch
size until throughput plateaus. Picks the fastest; uses the smallest batch within 8% of peak. Then:
- Islands on: ~48 networks per CMA-ES island (~32 for GA) fill the batch.
- Islands off: CMA-ES capped at 512 networks; extra capacity becomes more scenarios per network.

Speed is measured; population/island sizing is rule-based.

## Settings
| Setting | Effect |
|---|---|
| CMA-ES / GA | CMA-ES adapts per-weight step sizes (default). GA needs decay. |
| Islands | Parallel populations swapping their best every N generations. |
| Scenarios / reuse | 16–32 scenarios, kept 5–10 generations, for fair comparisons. |
| Past frames K, memory neurons | History for noisy tasks. Change the network shape (fresh network). |
| Physics step, decide every step | Smaller step = finer control, slower training. |

## Tips
- Watch Validation, not Champ hits (best-of-population is often lucky).
- Decay ×0.995 with a 1e-4 floor; ×0.9 freezes training.
- Turn on speed reward only after it hits reliably.
