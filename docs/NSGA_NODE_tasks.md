# NSGA-NODE Development Tasks

This document outlines the detailed requirements and tasks for implementing the NSGA-NODE module. It implements the **Hybrid NSGA-NODE** architecture, combining `torchode` based solvers with evolutionary strategies.

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
        │   ├── problem.py
        │   ├── orchestrator.py
        tests/
        ├── node/
        ```

### Task 2.2: Implement `NodeDynamicsWrapper`
**Goal:** Create a dynamics function $f(t, y, \theta)$ compatible with `torchode`.
**Location:** `src/nsga_node/dynamics.py`

*   **Action Items:**
    1.  Create class `NodeDynamicsWrapper(torch.nn.Module)`.
    2.  Implement `forward(self, t, y, batched_state_dict)`:
        *   **Batching Contract:** Explicitly handles a *batch* of `y` and a *batch* of `state_dict` (reconstructed via `genome_interface.batch_to_state_dict`).
        *   Use `torch.func.functional_call` mapped over the batch dimension.
        *   *Why Batched State Dict?* To avoid reconstructing the dict inside the solver loop (efficiency).
*   **Unit Tests (`tests/node/test_dynamics.py`):**
    *   **Test:** `test_dynamics_batched_state_dict_input`: Verify shape correctness when passed pre-batched dictionaries.

### Task 2.3: Implement `VectorizedNodeEvaluator` (Hybrid Modes)
**Goal:** Evaluate ODE trajectories. Must support "Fitness Mode" (Forward-only) and "Gradient Mode" (Adjoint/Backprop).
**Location:** `src/nsga_node/evaluator.py`

*   **Action Items:**
    1.  Create class `VectorizedNodeEvaluator`.
    2.  Implement `evaluate_population(self, population: np.ndarray, mode='fitness')`:
        *   **Mode='fitness' (NSGA):**
            *   Use `nsga_evaluation_context` (weights detached).
            *   Solver: `torchode` standard solver.
            *   **Strict No-Autograd:** Ensure the solve is "Forward-Only". No adjoint method is used. No graph is built. Return detached losses.
        *   **Mode='gradient' (ADAM):**
            *   Solver: `torchode` with **Adjoint** or **Autograd** enabled.
            *   Return: `(DataLoss, PhysicsLoss)` attached to graph.
    3.  **Objectives Definition:**
        *   **Data Loss:** MSE(Trajectory).
        *   **Physics/Correction Loss:** L2 Norm of the Neural Network output (Correction Term) over the trajectory. This encourages the model to rely on the known physics $f_{phys}$ and use the NN only when necessary.
*   **Unit Tests (`tests/node/test_evaluator.py`):**
    *   **Test:** `test_evaluator_modes`: Verify `grad_fn` is present in 'gradient' mode and absent in 'fitness' mode.

### Task 2.4: Implement `NsgaNodeProblem`
**Goal:** `pymoo` integration for NODE.
**Location:** `src/nsga_node/problem.py`

*   **Action Items:**
    1.  Create class `NsgaNodeProblem`.
    2.  **Initialization:**
        *   Support `current_adam_weights` and `sampling_radius`.
    3.  **Objectives:** Explicitly label Obj 1 as "Data Error" and Obj 2 as "Correction Magnitude" (Physics Deviation).

### Task 2.5: Implement `HybridNodeOrchestrator`
**Goal:** The master loop for NODE (ADAM <-> NSGA).
**Location:** `src/nsga_node/orchestrator.py`

*   **Action Items:**
    1.  Create class `HybridNodeOrchestrator`.
    2.  Implement `train()` loop mirroring the PINN orchestrator.
        *   **ADAM Phase:** Update `NodeDynamics` weights using `evaluate_population(mode='gradient')`.
        *   **NSGA Phase:** Evolve population using `evaluate_population(mode='fitness')`.
        *   **Handoff Protocol:**
            1.  Select best "balanced" individual (Knee/Hybrid) from Pareto Front.
            2.  **Convert:** Use `interface.batch_to_state_dict` to transform the genome into a state dict.
            3.  **Load:** `dynamics.load_state_dict(...)`.
            4.  **Reset Optimizer:** Clear ADAM state buffers (momentum/variance) to prevent stale state issues.
*   **Unit Tests:**
    *   **Test:** `test_hybrid_loop_handoff`: Verify weights change after NSGA phase and optimizer state is reset.

### Task 2.6: Validation - Hybrid Oscillator (Identifiability)
**Goal:** Verify NSGA-NODE on a well-posed problem.
**Location:** `examples/verify_node_oscillator.py`

*   **Action Items:**
    1.  **Problem Definition:**
        *   True System: $\ddot{x} + 0.5 \dot{x} + 2.0 x = 0$ (Damped Oscillator).
        *   Physics Model (Incomplete): $\ddot{x} + 2.0 x = u_{NN}(\dot{x})$.
        *   *Constraint:* We assume Mass (1.0) and Stiffness (2.0) are known, Damping is unknown.
    2.  **Identifiability:** This setup ensures the NN *must* learn the damping term $-0.5\dot{x}$ to minimize Data Loss, while the "Correction Magnitude" objective encourages it to be as small as possible (avoiding overfitting noise).
    3.  **Run Hybrid Training:**
        *   Check if the NN converges to $-0.5\dot{x}$ (linear relationship).
        *   Verify "Correction Loss" prevents the NN from learning high-frequency noise in the data.

---

## Testing Strategy
*   **Unit Tests:** Verify `torchode` integration in both autograd and no-grad contexts.
*   **Validation:** Ensure the validation problem is "Physically Augmented" (Incomplete Physics + NN) rather than "Pure Blackbox", as this highlights the multi-objective benefit (Tradeoff: Fit vs Trust in Physics).
