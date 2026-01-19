# NSGA-PINN Development Tasks

This document outlines the detailed requirements, tasks, and testing strategies for setting up the project and implementing the NSGA-PINN module. It assumes a modular architecture with a shared core.

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
        │   └── problem.py
        tests/
        ├── core/
        └── pinn/
        ```
*   **Verification:**
    *   Run `uv sync` to install dependencies.
    *   Verify `import torch`, `import pymoo` work in a test script.

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
        *   *Note:* Ensure efficient reshaping without unnecessary copying.
*   **Unit Tests (`tests/core/test_interface.py`):**
    *   **Test:** `test_flatten_unflatten_consistency`
        *   Create a simple CNN or MLP model.
        *   Flatten it to genome.
        *   Pass genome (batch size 1) to `batch_to_state_dict`.
        *   Compare original `state_dict` vs reconstructed. Assert strict equality (`torch.allclose`).
    *   **Test:** `test_batch_unflatten_shapes`
        *   Input a random population matrix of size `(10, Num_Params)`.
        *   Verify output tensors all have leading dimension 10.

### Task 0.3: Implement Vectorization Utilities
**Goal:** Helper functions for `torch.func` operations.
**Location:** `src/nsga_neuro_evolution_core/utils.py`

*   **Action Items:**
    1.  Implement a helper to wrap `torch.func.functional_call` for batched execution.
    2.  Implement a context manager/decorator for `torch.no_grad()` to be used during evaluation (except where gradients are explicitly needed for physics residuals).
*   **Unit Tests (`tests/core/test_utils.py`):**
    *   **Test:** `test_functional_call_batching`
        *   Verify `vmap` over parameters works for a simple function.

---

## Phase 1: NSGA-PINN Implementation (`nsga_pinn`)

### Task 1.1: Implement `VectorizedPinnEvaluator`
**Goal:** The core engine that computes Data Loss and Physics Loss for the entire population on GPU/CPU.
**Location:** `src/nsga_pinn/evaluator.py`

*   **Action Items:**
    1.  Create class `VectorizedPinnEvaluator`.
    2.  Implement `__init__(self, genome_interface, data_points, collocation_points, pde_residual_fn)`.
    3.  Implement `evaluate_population(self, population: np.ndarray) -> np.ndarray`:
        *   Convert `population` to `batched_state_dict` using `genome_interface`.
        *   Define `compute_loss(params, inputs)`:
            *   Use `torch.func.functional_call` to evaluate model with `params`.
            *   Compute Data Loss (MSE).
            *   Compute Physics Loss (Residual squared) using `torch.autograd.grad` (or `torch.func.grad`).
        *   Use `torch.vmap` to vectorize `compute_loss` over the population dimension (dim 0).
        *   **Crucial:** Handle the mix of `no_grad` (for weights) and `grad` (for input coordinates x, t).
*   **Unit Tests (`tests/pinn/test_evaluator.py`):**
    *   **Test:** `test_evaluator_output_shape`
        *   Mock a simple PDE (e.g., $u_x = 0$).
        *   Pass population size 5.
        *   Assert output is `(5, 2)` (Data Loss, Physics Loss).
    *   **Test:** `test_evaluator_correctness_serial`
        *   Compare vector output against a standard `for` loop calculation for 3 specific genomes. Ensure results match.

### Task 1.2: Implement `NsgaPinnProblem` Orchestrator
**Goal:** The `pymoo` Problem definition.
**Location:** `src/nsga_pinn/problem.py`

*   **Action Items:**
    1.  Create class `NsgaPinnProblem(pymoo.core.problem.Problem)`.
    2.  Implement `__init__(self, model_template, evaluator, bounds=None)`.
        *   Determine `n_var` from `genome_interface`.
        *   Set `n_obj=2`.
    3.  Implement `_evaluate(self, x, out, *args, **kwargs)`:
        *   Call `evaluator.evaluate_population(x)`.
        *   Assign results to `out["F"]`.
*   **Unit Tests (`tests/pinn/test_problem.py`):**
    *   **Test:** `test_pymoo_integration`
        *   Instantiate `NsgaPinnProblem`.
        *   Run one generation of `NSGA2` with population size 4.
        *   Verify no runtime errors and valid fitness values.

### Task 1.3: Validation - Burgers' Equation
**Goal:** Verify the system solves a real PDE.
**Location:** `examples/verify_pinn_burgers.py`

*   **Action Items:**
    1.  Define Burgers' Equation Residual: $u_t + u u_x - (0.01/\pi) u_{xx} = 0$.
    2.  Setup Data: Generate synthetic exact solution data.
    3.  Setup Model: MLP (e.g., 3 layers, 20 neurons, Tanh).
    4.  Run `NSGA-II` for ~50-100 generations.
    5.  **Validation Check:**
        *   Check that the Pareto Front is non-trivial (trade-off exists).
        *   Select "Best Data Fit" individual and verify MSE < Threshold.
        *   Select "Best Physics Fit" individual and verify Residual < Threshold.
*   **Run Configuration:**
    *   Allow command line arg `--device` to select 'cpu', 'cuda', or 'mps'.
    *   Default to 'cpu'.

---

## Testing Strategy
*   **Unit Tests:** Run with `pytest`. Must pass on CPU. Mock heavy GPU ops where possible.
*   **Validation Tests:** Python scripts in `examples/`. Should log timing to demonstrate `vmap` speedup if run on GPU.
