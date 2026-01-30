# Consider NSGA then ADAM

In the `src/nsga_node/orchestrator.py`, we architected the `train` algorithm to perform the adam update before conducting the NSGA phase.  What if we did the NSGA phase first, then did the ADAM phase?  What's the potential cuase and effects for doing that?

## Cause and Effects
Short answer: NSGA‑first is feasible, but it changes the meaning of each epoch. You’ll be using NSGA as the primary search and ADAM as a local refinement step, which can help escape local minima but can also collapse the Pareto trade‑off you just discovered.

### Here’s the cause/effect breakdown:

* **Cause: you want global exploration first.**</br>Effect: NSGA finds a diverse Pareto front; ADAM then picks one point and “pulls” it toward a local optimum. This often reduces diversity and can bias toward whichever objective ADAM emphasizes.

* **Cause: you want NSGA to act as initialization each epoch.**</br>Effect: ADAM becomes a fine‑tuner of the chosen individual. This can improve data‑fit or physics loss quickly, but you may undo the multi‑objective balance NSGA created unless you re‑select from the front after ADAM.

* **Cause: you want to escape ADAM stagnation.**</br>Effect: NSGA‑first can “jump” to new regions, but it also makes ADAM steps less stable because optimizer state (momenta) is now mismatched; you should reinitialize optimizer state after loading the NSGA‑selected weights.

* **Cause: you want to run fewer ADAM steps.**</br>Effect: NSGA‑first + short ADAM can work, but if ADAM is too short the epoch won’t actually refine; if it’s too long, it can erase the multi‑objective trade‑off.

* **Cause: you want easier fitness evaluation.**</br>Effect: NSGA fitness stays clean (forward‑only), but you must separate metrics: measure “NSGA improvement” before ADAM, otherwise you’re attributing ADAM gains to NSGA.

## Why we currently do ADAM → NSGA in the hybrid design:

* ADAM first gives a stable local baseline and clean gradients for the epoch.
* NSGA then explores around a well‑behaved solution to recover trade‑offs or improve physics vs data.
* The “hand‑off” is explicit: NSGA picks weights; next epoch ADAM starts from those.

## If you swap the order, I’d recommend:

1. Reinitialize optimizer state every epoch (or after NSGA).
1. Evaluate and log metrics before and after ADAM so you know which phase is driving improvements.
1. If multi‑objective fidelity matters, consider re‑selecting from the Pareto front after ADAM or keep the NSGA‑best model as a separate checkpoint.

## The Motivation for NSGA first
The idea of running NSGA fist is to search a wider landscape for the valey with the deepest canyon, then let ADAM hone that into the final answer.  Are you saying that by doing this we would reduce diversity and create a biases solution that is not the global minimum?

Not exactly. The diversity issue isn’t that NSGA‑first prevents a global minimum; it’s that once you hand a single individual to ADAM, you collapse the Pareto set to one point, and ADAM will then optimize whatever scalarized objective you give it, which can bias the solution toward one trade‑off.

**A few clarifications:**

* In multi‑objective problems there is no single “global minimum.” There is a Pareto front of non‑dominated trade‑offs. NSGA explores that front; ADAM refines one point on it.
* Diversity loss happens at the hand‑off. The moment you select one genome and run ADAM, you are choosing a trade‑off. That’s expected, not necessarily wrong.
* ADAM doesn’t “bias away from the global minimum,” it biases toward the scalar objective you optimize (e.g., weighted data+physics loss). If that scalarization doesn’t represent your intended trade‑off, the refined point can drift away from the Pareto front.

If your goal is **“NSGA finds the deep valley, ADAM sharpens it,”** you can preserve the intent by:

1. Keep an archive of the Pareto front and log it each epoch.
1. Select a knee point or a fixed trade‑off rule for the NSGA → ADAM hand‑off.
1. Re‑evaluate the front after ADAM, and optionally keep the best of both (pre‑ and post‑ADAM) for the next epoch.

So yes: NSGA‑first doesn’t inherently ruin global optimality; it narrows the multi‑objective solution space to one chosen compromise, and ADAM then refines that compromise. That’s the key trade‑off.

## How?

So if we wanted to see what would happen if we changed the order such that NSGA-first, I'm thinking that we create a sister `train` function like `train_nsga-first` in `src.nsga_node.orchestrator.HybridNodeOrchestrator` and then call that as an alternate "orchestrator" in `examples.verify_node_oscillator.main` based upon a CLI that user specifiies as nsga-first flag.  Would this be a good way to implement the nsga-first without breaking the existing orchestrator?

### Implementation considerations

Yes — that’s a good, low‑risk way to explore NSGA‑first without breaking the existing ADAM‑first path. Just keep the current `train()` intact and add an alternate path, then select it via a CLI flag in `verify_node_oscillator.py`.

**A couple of suggestions so it stays clean:**
* Avoid duplication. Factor the phase steps into helpers (e.g., `_run_adam_phase`, `_run_nsga_phase`) and call them in different order from `train()` and `train_nsga_first()`.
* Keep outputs identical. Make both methods return the same tuple structure (model, history, etc.) so the caller doesn’t need special casing beyond the flag.
* Reset optimizer state after NSGA. If NSGA picks new weights, the Adam optimizer state is stale; reinit it or zero its state before the ADAM phase.
* Keep schedule logic consistent. If you’re using adaptive ADAM step scheduling, make sure the same signals are logged in both orders.
* CLI flag defaults to current behavior. Something like `--nsga_first` (default False) or `--phase_order adam-first|nsga-first`.

## Actions taken

Added a configurable phase order without touching the existing ADAM‑first path. `HybridNodeOrchestrator` now routes through a shared `_train_loop`, with a new `train_nsga_first()` for the NSGA‑first experiment, and `verify_node_oscillator.py` gets a `--phase_order` flag that selects the appropriate training method while keeping baseline behavior unchanged.

### Details

**New orchestration** path with `phase_order` support in orchestrator.py.
CLI flag `--phase_order {adam-first,nsga-first}` wired into `verify_node_oscillator.py` and used only for the hybrid run.

For example usage, 

> `python -m examples.verify_node_oscillator --phase_order nsga-first`

## Compare all methods

Updated `verify_node_oscillator.py` to run and compare both hybrid orders in the same run, alongside Baseline and Classical. The script now defaults to `--phase_order both` and adds separate models, evaluators, training runs, metrics, and plot lines for:

* Exact
* NSGA‑NODE (ADAM‑first)
* NSGA‑NODE (NSGA‑first)
* Baseline (ADAM)
* Classical NN Baseline

**Notes:**

* Each hybrid run uses its own model/evaluator and separate Pareto GIF (`pareto_front_node_adam_first.gif`, `pareto_front_node_nsga_first.gif`).
* The plot and stats table only include whichever hybrid runs were actually executed (based on `--phase_order`).

**Example usage:**

Run the full comparison (default both):
```bash
python -m examples.verify_node_oscillator`
```

Run a single hybrid order if desired:
```bash
python -m examples.verify_node_oscillator --phase_order adam-first

python -m examples.verify_node_oscillator --phase_order nsga-first
```

## Results from Comparing all methods

```bash
python -m examples.verify_node_oscillator
Running on cpu

--- Training Hybrid NSGA-NODE (ADAM-first), GA generations: 20 ---
Hybrid: 100%|███████████████████████| 50/50 [01:58<00:00,  2.37s/Epoch, adam=0.044, steps=50, F=[0.000,0.001]]
Hybrid (ADAM-first) Time: 120.97s

--- Training Hybrid NSGA-NODE (NSGA-first), GA generations: 20 ---
Hybrid: 100%|███████████████████████| 50/50 [01:57<00:00,  2.35s/Epoch, adam=0.044, steps=50, F=[0.000,0.005]]
Hybrid (NSGA-first) Time: 120.13s

--- Training Baseline (Pure ADAM) ---
Baseline: 100%|██████████████████████████████████████| 50/50 [00:46<00:00,  1.07Epoch/s, adam=0.041, steps=20]
Baseline Time: 46.70s

--- Training Classical Baseline (NN Residual) ---
Classical: 100%|██████████████████████████████████████████| 1000/1000 [00:00<00:00, 1351.35Step/s, loss=0.014]
Classical Time: 0.74s

--- Evaluation ---
Hybrid NODE (ADAM-first) MSE   (1876 ADAM Steps, 120.97s): 0.0701
Hybrid NODE (NSGA-first) MSE   (1807 ADAM Steps, 120.13s): 0.0058
Baseline NODE MSE (1000 ADAM Steps,  46.70s): 0.0052
Classical NN MSE  (1000 ADAM Steps,   0.74s): 0.0190
Hybrid NODE (ADAM-first) Tail Mean:  0.3081 -0.0901, Std:  0.1384  0.4343
Hybrid NODE (NSGA-first) Tail Mean:  0.1311 -0.0109, Std:  0.0510  0.1811
Baseline NODE Tail Mean:  0.1218 -0.0087, Std:  0.0486  0.1732
Classical NN Tail Mean: -0.0304 -0.0177, Std:  0.0108  0.0053
Exact Tail Mean:  0.0724  0.0067
Plot saved to examples/node_oscillator_comparison.png
```

## Results from Comparing all methods with more epochs and generations e400as100gen100

```bash
python -m examples.verify_node_oscillator --epochs 400 --adam_steps 100 --nsga_gens 100 --output_plot examples/node_oscillator_comparison_e400as100gen100.png
Running on cpu

--- Training Hybrid NSGA-NODE (ADAM-first), GA generations: 100 ---
Hybrid: 100%|██████████████████████████████████████████████| 400/400 [40:51<00:00,  6.13s/Epoch, adam=0.046, steps=50, F=[0.000,0.003]]
Hybrid (ADAM-first) Time: 2472.47s

--- Training Hybrid NSGA-NODE (NSGA-first), GA generations: 100 ---
Hybrid: 100%|███████████████████████████████████████████████| 400/400 [41:40<00:00,  6.25s/Epoch, adam=0.044, steps=50, F=[0.000,0.001]]
Hybrid (NSGA-first) Time: 2520.76s

--- Training Baseline (Pure ADAM) ---
Baseline: 100%|█████████████████████████████████████████████████████████████| 400/400 [43:14<00:00,  6.49s/Epoch, adam=0.615, steps=100]
Baseline Time: 2594.76s

--- Training Classical Baseline (NN Residual) ---
Classical: 100%|██████████████████████████████████████████████████████████████████████| 400/400 [00:00<00:00, 1140.80Step/s, loss=0.016]
Classical Time: 0.35s

--- Evaluation ---
Hybrid NODE (ADAM-first) MSE   (20200 ADAM Steps, 2472.47s): 0.0689
Hybrid NODE (NSGA-first) MSE   (20350 ADAM Steps, 2520.76s): 0.0047
Baseline NODE MSE (40000 ADAM Steps, 2594.76s): 0.1001
Classical NN MSE  (0400 ADAM Steps,   0.35s): 0.0207
Hybrid NODE (ADAM-first) Tail Mean:  0.3122 -0.0869, Std:  0.1402  0.4461
Hybrid NODE (NSGA-first) Tail Mean:  0.1381 -0.0061, Std:  0.0510  0.1832
Baseline NODE Tail Mean: -0.2630 -0.0014, Std:  0.0147  0.0528
Classical NN Tail Mean: -0.0279 -0.0211, Std:  0.0130  0.0047
Exact Tail Mean:  0.0724  0.0067
Plot saved to examples/node_oscillator_comparison_e400as100gen100.png
```

## Results from Comparing all methods with more epochs and generations e400as100gen400
```bash
xpython -m examples.verify_node_oscillator --epochs 400 --adam_steps 100 --nsga_gens 400 --output_plot examples/node_oscillator_comparison_e400as100gen400.png
Running on cpu

--- Training Hybrid NSGA-NODE (ADAM-first), GA generations: 400 ---
Hybrid: 100%|████████████████████| 400/400 [2:05:10<00:00, 18.78s/Epoch, adam=0.048, steps=50, F=[0.000,0.003]]
Hybrid (ADAM-first) Time: 7531.33s

--- Training Hybrid NSGA-NODE (NSGA-first), GA generations: 400 ---
Hybrid: 100%|████████████████████| 400/400 [1:56:05<00:00, 17.41s/Epoch, adam=0.053, steps=50, F=[0.000,0.007]]
Hybrid (NSGA-first) Time: 6984.23s

--- Training Baseline (Pure ADAM) ---
Baseline: 100%|███████████████████████████████████| 400/400 [34:29<00:00,  5.17s/Epoch, adam=17.181, steps=100]
Baseline Time: 2069.32s

--- Training Classical Baseline (NN Residual) ---
Classical: 100%|█████████████████████████████████████████████| 400/400 [00:00<00:00, 1417.46Step/s, loss=0.017]
Classical Time: 0.29s

--- Evaluation ---
Hybrid NODE (ADAM-first) MSE   (20150 ADAM Steps, 7531.33s): 0.0683
Hybrid NODE (NSGA-first) MSE   (20700 ADAM Steps, 6984.23s): 0.0057
Baseline NODE MSE (40000 ADAM Steps, 2069.32s): 3.0192
Classical NN MSE  (0400 ADAM Steps,   0.29s): 0.0223
Hybrid NODE (ADAM-first) Tail Mean:  0.3018 -0.0938, Std:  0.1411  0.4401
Hybrid NODE (NSGA-first) Tail Mean:  0.1216 -0.0046, Std:  0.0479  0.1751
Baseline NODE Tail Mean: -0.8857 -0.0908, Std:  0.4403  1.5797
Classical NN Tail Mean: -0.0256 -0.0224, Std:  0.0138  0.0051
Exact Tail Mean:  0.0724  0.0067
Plot saved to examples/node_oscillator_comparison_e400as100gen400.png
```

## Summary of Results

--epochs 400 --adam_steps 100 --nsga_gens 400
Hybrid NODE (ADAM-first) MSE   (20150 ADAM Steps, 7531.33s): 0.0683
Hybrid NODE (NSGA-first) MSE   (20700 ADAM Steps, 6984.23s): 0.0057
Baseline NODE MSE (40000 ADAM Steps, 2069.32s): 3.0192
Classical NN MSE  (0400 ADAM Steps,   0.29s): 0.0223

--epochs 400 --adam_steps 100 --nsga_gens 100
Hybrid NODE (ADAM-first) MSE   (20200 ADAM Steps, 2472.47s): 0.0689
Hybrid NODE (NSGA-first) MSE   (20350 ADAM Steps, 2520.76s): 0.0047
Baseline NODE MSE (40000 ADAM Steps, 2594.76s): 0.1001
Classical NN MSE  (0400 ADAM Steps,   0.35s): 0.0207

**Table of MSE:**
| Method | MSE| Total ADAMs | Outer ADAMs | Inner ADAMs | Inner NSGA | Time (min) |
|:------ |:-----------:|:-----------:|:-----------:|:-----------:|:----------:|:----------:|
| ADAM-first | 0.069   | 20200   | 400  | 100 | 100 | 41    |
| NSGA-first | 0.005   | 20350   | 400  | 100 | 100 | 42    |
| NODE base  | 0.100   | 40000   |  NA  | NA  | NA  | 43    |
| NN classic | 0.021   | 400     |  NA  | NA  | NA  | 0.35s |
| ADAM-first | 0.068   | 20150   | 400  | 100 | 400 | 126   |
| NSGA-first | 0.006   | 20700   | 400  | 100 | 400 | 116   |
| NODE base  | 3.019   | 40000   |  NA  | NA  | NA  | 34    |
| NN classic | 0.022   | 400     |  NA  | NA  | NA  | 0.35s |

**NOTE: the table above reports total epochs before the "actual" steps method was added to the code. The above table is slightly over counted.** Future runs will generate this table with the correct ADAM counts.

# How adaptive ADAM steps work (`adapt_adam_steps=True`)
## General approach
* Each epoch starts with the current `adam_steps_per_epoch`. That value is **snapshotted** as adam_steps_used and used for *this* epoch’s ADAM loop.
* After both phases run, the orchestrator looks at recent improvement histories:
    * **ADAM improvement**: change in average ADAM loss per epoch.
    * **NSGA improvement**: change in best Pareto sum per epoch.
*If **NSGA is consistently improving** and **ADAM is flat**, it **reduces** `adam_steps_per_epoch`.
*If **NSGA stagnates or is noisy**, it increases `adam_steps_per_epoch`, and can optionally do an **ADAM‑only** warmup by skipping NSGA for a few epochs.
* The update to `adam_steps_per_epoch` only affects future epochs (not the one just completed).
## Total ADAM steps (ADAM‑first)
* The total is the sum of `adam_steps_used` over all epochs in the ADAM‑first run:
> `total_adams = sum(history[i]["adam_steps_used"])`
* This is exact because adam_steps_used is captured *before* any adaptive change.
## Total ADAM steps (NSGA‑first)
* Same formula:
> `total_adams = sum(history[i]["adam_steps_used"])`
* The **counting is identical**; the only difference is *which signals drive adaptation*:
    * NSGA‑first changes model weights before ADAM, affecting ADAM loss trends.
    * ADAM‑first changes model weights before NSGA, affecting NSGA improvement trends.
* That difference can lead to **different step schedules**, even though the summation is computed the same way.
## Key point:
The order (ADAM‑first vs NSGA‑first) doesn’t change how total steps are computed; it changes the **adaptive trajectory** that determines how many steps are used each epoch.