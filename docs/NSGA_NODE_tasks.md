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
*   **Unit Tests (`tests/node/test_dynamics.py`):**
    *   **Test:** `test_dynamics_batched_state_dict_input`: Verify shape correctness when passed pre-batched dictionaries.

### Task 2.3: Implement `VectorizedNodeEvaluator` (Hybrid Modes)
**Goal:** Evaluate ODE trajectories. Must support "Fitness Mode" (Forward-only) and "Gradient Mode" (Autograd).
**Location:** `src/nsga_node/evaluator.py`

*   **Action Items:**
    1.  Create class `VectorizedNodeEvaluator`.
    2.  Implement `evaluate_population(self, population: np.ndarray, mode='fitness')`:
        *   **Mode='fitness' (NSGA):**
            *   Use `nsga_evaluation_context(self.model, self.inputs)` (weights detached).
            *   **API Usage:** `term = torchode.ODETerm(self.dynamics, with_args=True)`.
            *   **API Usage:** `sol = solver.solve(problem, args=batched_state_dict)`.
            *   **Strict No-Autograd:** Ensure the solve is "Forward-Only". No adjoint, no autograd graph. Return detached losses.
        *   **Mode='gradient' (ADAM):**
            *   Solver: `torchode` with **Direct Autograd** enabled (Backprop through time).
            *   *Note:* Using Direct Autograd as default over Adjoint for stability on validation problems. Adjoint support can be added later.
            *   Return: `(DataLoss, PhysicsLoss)` attached to graph.
    3.  **Objectives Definition:**
        *   **Data Loss:** MSE(Trajectory).
        *   **Correction Loss (Physics Proxy):** L2 Norm of the Neural Network output (Correction Term) over the trajectory.
        *   *Rationale:* Minimizing this term encourages the model to adhere to the base physics model $f_{phys}$ and use the NN $u_{\theta}$ only where necessary to fit the data (Principle of Parsimony).
*   **Unit Tests (`tests/node/test_evaluator.py`):**
    *   **Test:** `test_evaluator_modes`: Verify `grad_fn` is present in 'gradient' mode and absent in 'fitness' mode.

### Task 2.4: Implement `NsgaNodeProblem`
**Goal:** `pymoo` integration for NODE.
**Location:** `src/nsga_node/problem.py`

*   **Action Items:**
    1.  Create class `NsgaNodeProblem`.
    2.  **Initialization:**
        *   Support `current_adam_weights` and `sampling_radius`.
    3.  **Objectives:** Explicitly label Obj 1 as "Data Error" and Obj 2 as "Correction Magnitude".

### Task 2.5: Implement `HybridNodeOrchestrator`
**Goal:** The master loop for NODE (ADAM <-> NSGA).
**Location:** `src/nsga_node/orchestrator.py`

*   **Action Items:**
    1.  Create class `HybridNodeOrchestrator`.
    2.  Implement `train()` loop mirroring the PINN orchestrator.
        *   **ADAM Phase:**
            *   Ensure `model.requires_grad_(True)` (using `adam_update_context` or explicit call).
            *   Update `NodeDynamics` weights using `evaluate_population(mode='gradient')`.
        *   **NSGA Phase:** Evolve population using `evaluate_population(mode='fitness')`.
        *   **Handoff Protocol:**
            1.  Select best "balanced" individual (Knee/Hybrid) from Pareto Front.
            2.  **Convert:** Use `interface.genome_to_state_dict(best_individual)` (Single genome helper).
            3.  **Load:** `dynamics.load_state_dict(converted_weights)`.
            4.  **Reset Optimizer:** Clear ADAM state buffers (momentum/variance) to prevent stale state issues.

#### 2.5.2 Adaptive ADAM/NSGA Schedule (Optional)
**Goal:** Mirror the PINN schedule logic to balance NODE refinement vs exploration.

*   **Condition A (NSGA strong, ADAM weak):**
    *   If NSGA consistently improves the best Pareto score while ADAM improvement is near zero, **reduce** `adam_steps_per_epoch`.
*   **Condition B (NSGA stagnates or noisy):**
    *   If NSGA improvement is below a threshold for multiple epochs **or** the Pareto front becomes noisy, **increase** `adam_steps_per_epoch`.
    *   Optionally run **ADAM-only warm-up** for a few epochs (skip NSGA) before re-enabling NSGA.
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
    2.  **Identifiability:** This setup ensures the NN *must* learn the damping term $-0.5\dot{x}$ to minimize Data Loss, while the "Correction Magnitude" objective encourages it to be as small as possible.
    3.  **Run Hybrid Training:**
        *   Check if the NN converges to $-0.5\dot{x}$ (linear relationship).
        *   Verify "Correction Loss" prevents the NN from learning high-frequency noise.

---

## Testing Strategy
*   **Unit Tests:** Verify `torchode` integration in both autograd and no-grad contexts.
*   **Validation:** Ensure the validation problem is "Physically Augmented" (Incomplete Physics + NN) rather than "Pure Blackbox".
