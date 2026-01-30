# Migration Guide: Adaptive Phase Order (ADAM-first → Selectable Phase Order)

This document lists the concrete implementation tasks needed to migrate an **ADAM-first hybrid NSGA-NODE** training loop to a **selectable phase-order** approach (ADAM-first or NSGA-first). It also includes minimal code snippets derived from this repo’s implementation. This is intended for developers integrating the same pattern elsewhere. **No comparison analysis is included.**

## Goal
Enable a hybrid training loop to run in either order per epoch:
- **ADAM → NSGA** (current default)
- **NSGA → ADAM** (optional)

The solution must keep existing ADAM-first behavior intact while enabling an alternate phase order.

---

## Task List (Implementation Requirements)

### 1) Refactor Orchestrator Into a Shared Loop
**Objective:** Avoid duplicating logic while allowing phase order control.

- Create a private/shared method (e.g., `_train_loop`) that contains the full epoch loop.
- Add a `phase_order` parameter (string or enum), validated to `"adam-first"` or `"nsga-first"`.
- Keep the existing public `train()` signature unchanged, but route it into `_train_loop(..., phase_order="adam-first")`.

**Acceptance Criteria**
- Existing callers of `train()` behave the same.
- Phase order can be switched without changing other arguments.

### 2) Add an Explicit NSGA-First Entry Point
**Objective:** Provide a clear alternate training method without breaking callers.

- Add a public method, e.g., `train_nsga_first()`.
- It should call `_train_loop(..., phase_order="nsga-first")` and accept the same arguments as `train()`.

**Acceptance Criteria**
- `train_nsga_first()` mirrors `train()` signature and return format.
- No code duplication of the training loop logic.

### 3) Make Phase Order Explicit in the Epoch Loop
**Objective:** Execute ADAM and NSGA phases in the requested order per epoch.

- Define a `phase_sequence = ("adam", "nsga")` or `(“nsga”, “adam”)` based on `phase_order`.
- Execute each phase using small helper closures (e.g., `run_adam_phase()`, `run_nsga_phase()`).
- Ensure the **same step logic** is applied regardless of ordering.

**Acceptance Criteria**
- Both orders run without code branching inside each phase.
- ADAM and NSGA phases are re-usable and isolated.

### 4) Preserve Adaptive ADAM Steps (Accurate Per-Epoch Accounting)
**Objective:** Track the exact ADAM steps used **per epoch**, even as adaptive logic changes the next epoch’s value.

- Snapshot `adam_steps_used = adam_steps_per_epoch` at the **start** of the epoch.
- Use `adam_steps_used` for the ADAM loop and average loss calculation.
- Store `adam_steps_used` in history for accurate totals.

**Acceptance Criteria**
- `history[i]["adam_steps_used"]` reflects actual steps executed in epoch `i`.
- Adaptive changes affect the **next epoch** only.

### 5) Align UI Prints With Steps Used in the Epoch
**Objective:** UI should show the steps that were actually executed.

- Print and progress bar should use `adam_steps_used` instead of `adam_steps_per_epoch`.

**Acceptance Criteria**
- Terminal output matches actual steps executed per epoch.

### 6) Add a CLI Flag to Select Phase Order
**Objective:** Allow easy toggling without code changes.

- Add a CLI argument, e.g. `--phase_order`, with choices `adam-first` and `nsga-first`.
- Default should be the current behavior (`adam-first`), or optionally `both` if you want to run both.
- Call the appropriate method (`train()` or `train_nsga_first()`) based on the flag.

**Acceptance Criteria**
- CLI switch controls phase order without changing any other logic.
- Existing runs (no flag) keep old behavior.

---

## Sample Code Snippets

### A) Orchestrator: Public Methods and Shared Loop
```python
class HybridNodeOrchestrator:
    def train(...):
        return self._train_loop(..., phase_order="adam-first")

    def train_nsga_first(...):
        return self._train_loop(..., phase_order="nsga-first")

    def _train_loop(..., phase_order="adam-first"):
        if phase_order not in {"adam-first", "nsga-first"}:
            raise ValueError("phase_order must be 'adam-first' or 'nsga-first'.")

        phase_sequence = (
            ("adam", "nsga") if phase_order == "adam-first" else ("nsga", "adam")
        )

        for epoch in range(epochs):
            adam_steps_used = adam_steps_per_epoch

            def run_adam_phase():
                # uses adam_steps_used
                pass

            def run_nsga_phase():
                # NSGA-II, selection, model handoff
                pass

            for phase in phase_sequence:
                if phase == "adam":
                    run_adam_phase()
                else:
                    run_nsga_phase()

            history.append({
                "epoch": epoch,
                "adam_steps_used": adam_steps_used,
                ...
            })
```

### B) Accurate ADAM Step Tracking
```python
adam_steps_used = adam_steps_per_epoch
for _ in range(adam_steps_used):
    # optimizer step
    pass

avg_adam_loss = adam_loss_accum / max(1, adam_steps_used)
```

### C) UI Print Alignment
```python
print(
    f"Epoch {epoch}: ADAM Loss={avg_adam_loss:.6f}, "
    f"NSGA Best F={nsga_best_f}, steps={adam_steps_used}"
)
```

### D) CLI Switch to Select Order
```python
parser.add_argument(
    "--phase_order",
    choices=["adam-first", "nsga-first"],
    default="adam-first",
)

train_fn = orchestrator.train_nsga_first if args.phase_order == "nsga-first" else orchestrator.train
hist = train_fn(...)
```

---

## Notes for Developers
- **Do not change the default** phase order in existing training code unless required. Keep ADAM-first as the default to preserve behavior.
- After NSGA changes the model weights, **reinitialize the optimizer** so that ADAM momentum does not carry over from the pre-NSGA state.
- Ensure metrics and schedules remain comparable across both orders (i.e., use consistent loss scalarization and evaluation modes).
- If `adapt_adam_steps` is enabled, totals must be computed from `adam_steps_used`, not the mutable `adam_steps_per_epoch`.

---

## Done-When Checklist
- [ ] `train()` still works unchanged (ADAM-first default).
- [ ] `train_nsga_first()` exists and calls the same internal loop.
- [ ] `phase_order` validated and respected.
- [ ] Per-epoch ADAM steps recorded as `adam_steps_used`.
- [ ] UI prints show `adam_steps_used`.
- [ ] CLI flag selects phase order without code edits.
