# NSGA-NODE Development Tasks

This document outlines the requirements and tasks for implementing the NSGA-NODE module. This module builds upon the `nsga_neuro_evolution_core` and introduces `torchode` for parallel differential equation solving.

**Tech Stack:**
*   Python 3.12+
*   Package Manager: `uv`
*   Core Libraries: `pytorch`, `pymoo`, `numpy`, `torchode`

---

## Phase 2: NSGA-NODE Implementation (`nsga_node`)

### Task 2.1: Dependency & Environment Update
**Goal:** Add Neural ODE specific dependencies.

*   **Action Items:**
    1.  Add `torchode` to project dependencies (`uv add torchode`).
    2.  Verify `torchode` installation works with current PyTorch version.
    3.  Create directory structure:
        ```text
        src/
        ├── nsga_node/
        │   ├── __init__.py
        │   ├── dynamics.py
        │   ├── evaluator.py
        │   └── problem.py
        tests/
        ├── node/
        ```

### Task 2.2: Implement `NodeDynamicsWrapper`
**Goal:** Create a dynamics function $f(t, y, \theta)$ that is compatible with `torchode` and batched parameters.
**Location:** `src/nsga_node/dynamics.py`

*   **Action Items:**
    1.  Create class `NodeDynamicsWrapper(torch.nn.Module)`.
    2.  Implement `__init__(self, neural_net_template, physics_f=None)`.
        *   `physics_f` is the optional known physics term.
    3.  Implement `forward(self, t, y, params)`:
        *   **Critical:** This function must handle a *single* instance of parameters if `vmap` is applied externally, OR handle the batching internally.
        *   *Design Decision:* `torchode` usually expects `forward(t, y, args)`. If `args` (params) has a batch dim, `torchode` vectorizes automatically over it?
        *   *Refined Approach:* Implement `forward` to accept a *batch* of `y` and a *batch* of `params`.
        *   Use `torch.func.functional_call` mapped over the batch dimension to evaluate the neural net part: $f_{NN}(y_i, t; \theta_i)$.
        *   Combine with $f_{physics}(y_i, t)$ if present.
*   **Unit Tests (`tests/node/test_dynamics.py`):**
    *   **Test:** `test_dynamics_shape_consistency`
        *   Batch size = 10. `y` shape `(10, State_Dim)`. `params` shape `(10, Param_Dim)`.
        *   Call `forward(t, y, params)`.
        *   Assert output shape is `(10, State_Dim)`.

### Task 2.3: Implement `VectorizedNodeEvaluator`
**Goal:** Evaluate ODE trajectories for the entire population in parallel using `torchode`.
**Location:** `src/nsga_node/evaluator.py`

*   **Action Items:**
    1.  Create class `VectorizedNodeEvaluator`.
    2.  Implement `__init__(self, genome_interface, t_eval, data_y, dynamics)`.
    3.  Implement `evaluate_population(self, population: np.ndarray) -> np.ndarray`:
        *   Convert `population` -> `batched_state_dict`.
        *   Construct `torchode.ODETerm` using `dynamics` (ensure `with_args=True` or compatible API).
        *   Setup `torchode.InitialValueProblem` with batched `y0` and batched `params` (as `args`).
        *   Select Solver (e.g., `torchode.Dopri5` with `IntegralController`).
        *   **Step:** `sol = solver.solve(problem)`
        *   **Optimization:** Apply `torch.compile` to the solve call if possible (mark as Optional/Advanced task).
        *   Compute Data Loss: MSE between `sol.ys` and `data_y`.
        *   Compute Physics/Regularization Loss: e.g., Magnitude of NN correction term, or smoothness.
        *   **Error Handling:** Catch NaNs (unstable dynamics) and assign penalty fitness.
*   **Unit Tests (`tests/node/test_evaluator.py`):**
    *   **Test:** `test_batch_solver_execution`
        *   Simple ODE: $\dot{y} = -\theta y$.
        *   Population: $\theta \in [0.1, 1.0]$.
        *   Verify solver runs and produces distinct trajectories for each $\theta$.
    *   **Test:** `test_nan_handling`
        *   Inject a "bad" genome that causes divergence.
        *   Verify evaluator returns penalty (inf) instead of crashing.

### Task 2.4: Implement `NsgaNodeProblem`
**Goal:** `pymoo` integration for NODE.
**Location:** `src/nsga_node/problem.py`

*   **Action Items:**
    1.  Create class `NsgaNodeProblem(pymoo.core.problem.Problem)`.
    2.  Similar to `NsgaPinnProblem`, but uses `VectorizedNodeEvaluator`.
    3.  Objectives:
        *   Obj 1: Trajectory Error (Data fit).
        *   Obj 2: Complexity/Regularization (e.g., L2 of weights, or integral of correction term).
*   **Unit Tests:**
    *   **Test:** Basic `pymoo` loop functional test.

### Task 2.5: Validation - Hybrid Oscillator
**Goal:** Verify NSGA-NODE on a Physics-Augmented task.
**Location:** `examples/verify_node_oscillator.py`

*   **Action Items:**
    1.  Problem: Damped Harmonic Oscillator where the damping term is unknown.
        *   True System: $\ddot{x} + c \dot{x} + k x = 0$.
        *   Physics Model: $\ddot{x} + k x = u_{NN}(\dot{x}, x)$.
    2.  Data: Generate trajectories with fixed $c, k$.
    3.  Run `NSGA-NODE`.
    4.  **Validation Check:**
        *   Verify Pareto Front exists (Accuracy vs Neural Net Magnitude).
        *   Verify the "Best Fit" model recovers the correct damping dynamics.
*   **Run Configuration:**
    *   Support `--device` argument.
    *   Log solve times.

---

## Testing Strategy (NODE Specific)
*   **Unit Tests:** Focus on the shape correctness of the "Batch-over-Parameters" logic. `torchode` API can be tricky with argument passing; tests must isolate this.
*   **Validation:** Start with very small time horizons ($T=1.0$) to prevent massive divergence during initial debugging.
