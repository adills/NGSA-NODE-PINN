# NSGA-PINN Development Tasks

This document outlines the detailed requirements, tasks, and testing strategies for setting up the project and implementing the NSGA-PINN module. It implements the **Hybrid NSGA-PINN** architecture (ADAM outer loop, NSGA inner loop) as defined in DOI 10.3390/a16040194.

**Tech Stack:**
*   Python 3.12+
*   Package Manager: `uv`
*   Core Libraries: `pytorch`, `pymoo`, `numpy`

---

## Phase 0: Project Initialization & Core Module (`nsga_neuro_evolution_core`)

### Task 0.1: Project Skeleton & Environment Setup
**Goal:** Initialize a modern Python project structure using `uv`.

*   **Action Items:**
    1.  Initialize project with `uv init`.
    2.  Configure `pyproject.toml` with dependencies:
        *   `torch` (Latest Stable)
        *   `pymoo` (Latest Stable)
        *   `numpy`
        *   `pytest` (Dev)
    3.  Create directory structure:
        ```text
        src/
        ├── nsga_neuro_evolution_core/
        │   ├── __init__.py
        │   ├── interface.py
        │   └── utils.py
        ├── nsga_pinn/
        │   ├── __init__.py
        │   ├── evaluator.py
        │   ├── problem.py
        │   └── orchestrator.py
        tests/
        ├── core/
        └── pinn/
        ```

### Task 0.2: Implement `PytorchGenomeInterface`
**Goal:** Create the bridge between flat genetic vectors and hierarchical PyTorch state dictionaries.
**Location:** `src/nsga_neuro_evolution_core/interface.py`

*   **Action Items:**
    1.  Create class `PytorchGenomeInterface`.
    2.  Implement `__init__(self, model_template: torch.nn.Module)`.
    3.  Implement `to_genome(self, model: torch.nn.Module) -> np.ndarray`:
        *   Flatten all trainable parameters into a single 1D numpy array.
    4.  Implement `batch_to_state_dict(self, population: np.ndarray) -> Dict[str, torch.Tensor]`:
        *   Input: `(Pop_Size, Num_Params)`
        *   Output: `state_dict` where each tensor has shape `(Pop_Size, ...original_dims...)`.
    5.  Implement `genome_to_state_dict(self, individual: np.ndarray) -> Dict[str, torch.Tensor]`:
        *   Input: `(Num_Params,)` (1D array).
        *   Output: Standard `state_dict` (unbatched) suitable for `model.load_state_dict`.
        *   *Note:* Handle shape checking (1D vs 2D) to prevent bugs.
*   **Unit Tests (`tests/core/test_interface.py`):**
    *   **Test:** `test_flatten_unflatten_consistency`: Round-trip check.
    *   **Test:** `test_batch_unflatten_shapes`: Leading batch dim check.
    *   **Test:** `test_single_genome_unflatten`: Verify `genome_to_state_dict` returns unbatched tensors.

### Task 0.3: Implement Scoped Gradient Contexts
**Goal:** Helper context managers to control gradient flow for Hybrid training.
**Location:** `src/nsga_neuro_evolution_core/utils.py`

*   **Action Items:**
    1.  Implement context manager `nsga_evaluation_context(model: nn.Module, inputs: torch.Tensor)`:
        *   **CRITICAL:** Do **NOT** use `torch.no_grad()`. Using `no_grad` prevents `autograd.grad` from working on inputs, which breaks PINN residual calculation.
        *   **On Enter:**
            1.  Save current `requires_grad` state of all `model.parameters()`.
            2.  Set `param.requires_grad_(False)` for all model parameters (Freezes weights).
            3.  Set `inputs.requires_grad_(True)` (Enables input sensitivity).
        *   **On Exit:**
            1.  Restore original `requires_grad` state to `model.parameters()` (usually True).
        *   **Purpose:** This configuration forces the computational graph to track operations *only* for input derivatives, keeping the weights treated as constants. This explicitly deconflicts the inner-loop physics calculation from outer-loop weight updates.
    2.  Implement context manager `adam_update_context()`:
        *   Standard `torch.enable_grad()`.
*   **Unit Tests (`tests/core/test_utils.py`):**
    *   **Test:** `test_nsga_context_behavior`:
        *   Create dummy model and input `x`.
        *   Inside `with nsga_evaluation_context(model, x):`:
            *   Assert `model.weight.requires_grad` is False.
            *   Assert `x.requires_grad` is True.
            *   Compute `y = model(x)`.
            *   `torch.autograd.grad(y, x)` **Succeeds**.
            *   `torch.autograd.grad(y, model.weight)` **Fails** (Expected).
        *   Outside context:
            *   Assert `model.weight.requires_grad` is True.

---

## Phase 1: NSGA-PINN Implementation (`nsga_pinn`)

### Task 1.1: Implement `VectorizedPinnEvaluator` (Hybrid Modes)
**Goal:** The core engine that computes Data Loss and Physics Loss. Must support "Fitness Mode" (NSGA) and "Gradient Mode" (ADAM/Debug).
**Location:** `src/nsga_pinn/evaluator.py`

*   **Action Items:**
    1.  Create class `VectorizedPinnEvaluator`.
    2.  Implement `evaluate_population(self, population: np.ndarray, mode='fitness') -> np.ndarray`:
        *   **Mode='fitness':**
            *   Use `with nsga_evaluation_context(self.model_template, self.inputs):`.
            *   Vectorize (`vmap`) over population.
            *   **Strict Output:** Return `(DataLoss, PhysicsLoss)` as detached Tensors/Numpy arrays.
        *   **Mode='gradient':**
            *   Used for standard ADAM training (single individual) or debugging.
            *   Enable full autograd.
            *   Return: `(DataLoss, PhysicsLoss)` with graph attached.
    3.  Define `compute_loss(params, inputs)`:
        *   Compute Data Loss (MSE).
        *   Compute Physics Loss (Residual squared) using `torch.autograd.grad` w.r.t inputs.
*   **Unit Tests (`tests/pinn/test_evaluator.py`):**
    *   **Test:** `test_evaluator_fitness_mode`: Verify no grad on weights, yes grad on inputs using the context manager.
    *   **Test:** `test_evaluator_output_shape`: `(Pop, 2)`.

### Task 1.2: Implement `NsgaPinnProblem`
**Goal:** The `pymoo` Problem definition.
**Location:** `src/nsga_pinn/problem.py`

*   **Action Items:**
    1.  Create class `NsgaPinnProblem(pymoo.core.problem.Problem)`.
    2.  **Initialization Policy:**
        *   Accept `current_adam_weights` (genome) in `__init__`.
        *   Define `bounds`: e.g., `[current - 1.0, current + 1.0]` or fixed global bounds.
        *   *Recommendation:* Implement `sampling` strategy that initializes population around `current_adam_weights`.
    3.  Implement `_evaluate`:
        *   Call `evaluator.evaluate_population(x, mode='fitness')`.
*   **Unit Tests:**
    *   **Test:** `test_problem_sampling_around_adam`: Verify initial population is centered on provided weights.

### Task 1.3: Implement `ParetoSelector`
**Goal:** Logic to select the best candidate from the Pareto Front to hand back to ADAM.
**Location:** `src/nsga_pinn/selector.py`

*   **Action Items:**
    1.  Create class `ParetoSelector`.
    2.  Implement methods:
        *   `select_best_data(front, F)`: Returns individual with min Data Loss.
        *   `select_best_physics(front, F)`: Returns individual with min Physics Loss.
        *   `select_knee_point(front, F)`: Returns individual at the "elbow".
        *   `select_hybrid(front, F, alpha=0.5)`: Returns min `alpha*Data + (1-alpha)*Physics`.
*   **Unit Tests:**
    *   **Test:** `test_knee_point_selection`: Provide synthetic convex front, verify knee selection.

### Task 1.4: Implement `HybridPinnOrchestrator`
**Goal:** The master loop implementing the paper's contract.
**Location:** `src/nsga_pinn/orchestrator.py`

*   **Action Items:**
    1.  Create class `HybridPinnOrchestrator`.
    2.  Implement `train(self, epochs, adam_steps_per_epoch, nsga_gens_per_epoch)`:
        *   **Outer Loop (Epochs):**
            *   **Step A (ADAM):** Run standard gradient descent on `current_model` for `adam_steps_per_epoch`.
            *   **Step B (NSGA - The "Jump"):**
                *   Initialize `NsgaPinnProblem` centered on `current_model`.
                *   Run `pymoo.NSGA2` for `nsga_gens_per_epoch`.
            *   **Step C (Handoff Protocol):**
                1.  Get Pareto Front from NSGA result.
                2.  Use `ParetoSelector` to pick `best_individual` (genome).
                3.  **Convert:** Use `interface.genome_to_state_dict(best_individual)` (Single genome helper).
                4.  **Load:** `current_model.load_state_dict(converted_weights)`.
                5.  **Reset Optimizer:** Explicitly reset the ADAM optimizer state (clear buffers `m`, `v`) to avoid instability.
                6.  **Deconflict:** Explicitly clear any lingering autograd states/caches if necessary.

#### 1.4.2 Adaptive ADAM/NSGA Schedule (Optional)
**Goal:** Adjust `adam_steps_per_epoch` based on observed performance to balance exploration (NSGA) and refinement (ADAM).

*   **Condition A (NSGA strong, ADAM weak):**
    *   If NSGA consistently improves the best Pareto score while ADAM improvement is near zero, **reduce** `adam_steps_per_epoch` (down to a minimum).
*   **Condition B (NSGA stagnates or noisy):**
    *   If NSGA improvement is below a threshold for multiple epochs **or** the Pareto front becomes noisy, **increase** `adam_steps_per_epoch`.
    *   Optionally run **ADAM-only warm-up** for a few epochs (skip NSGA) before re-enabling NSGA.
*   **Validation (`examples/verify_hybrid_pinn.py`):**
    *   Solve Burgers' Equation.
    *   Compare `Hybrid` convergence vs `Pure ADAM`.
    *   Verify "Jump" in loss curves where NSGA finds better basins.

---

## Testing Strategy
*   **Unit Tests:** Focus on gradient context safety and selection logic.
*   **Validation:** Explicitly check the "Handoff" mechanism—ensure the model weights actually update after the NSGA phase and optimizer buffers are cleared.
