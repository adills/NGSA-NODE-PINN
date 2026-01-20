# Hybrid NSGA-PINN and NSGA-NODE Framework

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

A robust, modular framework for **Multi-Objective Evolutionary Scientific Machine Learning**. This project implements the **Hybrid NSGA-PINN** and **NSGA-NODE** architectures, combining the local refinement capabilities of gradient-based optimization (ADAM) with the global search and multi-objective handling of evolutionary algorithms (NSGA-II). The NSGA-PINN theory is documented in NSGA-PINN paper but it has no promised code.[1](https://www.mdpi.com/1999-4893/16/4/194).  

[1](https://www.mdpi.com/1999-4893/16/4/194) B. Lu, C. Moya, and G. Lin, NSGA-PINN: A Multi-Objective Optimization Method for Physics-Informed Neural Network Training, Algorithms 16, 194 (2023).

## Motivation

The primary objective is to operationalize the NSGA-PINN framework, as conceptualized in recent literature (e.g., [1] MDPI 2023), which fundamentally redefines the training of scientific machine learning models. Instead of the traditional scalarization approach—where data loss and physics loss are combined into a weighted sum—this framework treats them as competing objectives to be optimized simultaneously via the Non-dominated Sorting Genetic Algorithm II (NSGA-II). [bibliography](docs/bibliography.md)

The motivation for this architectural shift lies in the limitations of gradient-based optimizers in the context of scientific machine learning. While Stochastic Gradient Descent (SGD) and ADAM are highly effective for convex or smooth loss landscapes, they frequently encounter severe pathologies in the training of PINNs, including spectral bias, entrapment in suboptimal local minima, and sensitivity to hyperparameter tuning of loss weights. By employing an evolutionary approach, specifically NSGA-II, the proposed system leverages a global search capability that generates a Pareto front of optimal solutions. This allows researchers to rigorously analyze the trade-offs between adherence to physical laws (physics loss) and fidelity to empirical measurements (data loss) without the arbitrary selection of weighting coefficients.

Following an exhaustive evaluation of the Python evolutionary computation ecosystem—specifically analyzing `PyGAD`, `DEAP`, and `pymoo`—the evidence advocates for a hybrid architectural approach. We recommend utilizing `pymoo` as the evolutionary orchestration engine due to its rigorous object-oriented design, superior handling of multi-objective constraints, and native support for vectorized problem definitions. This is coupled with **PyTorch’s Functional API (**<code>torch.func</code>**)** to manage the massive throughput required for evaluating populations of neural networks on hardware accelerators (GPUs).

Furthermore, the project addresses the specific extension of this framework to Neural ODEs (NSGA-NODE). It identifies a critical computational bottleneck in standard ODE solvers (`torchdiffeq`) when applied to evolutionary populations: the inability to efficiently batch-solve differential equations where the dynamics parameters vary across the batch. To resolve this, the blueprint proposes the integration of `torchode`, a parallel ODE solver library capable of independent batch-parameter handling. This integration is essential for scaling the NSGA-NODE approach, ensuring that the evolutionary evaluation remains computationally feasible.

The project uses the class-based design of the module, encompassing the `NeuroEvolutionOrchestrator`, `PytorchGenomeInterface`, and `VectorizedEvaluator`. It provides a capability for implementation, emphasizing performance optimization through JIT compilation and vectorization, ultimately delivering a robust tool for advanced scientific discovery.

**Current status:**
* [NGSA-PINN development tasks](docs/NSGA_PINN_tasks.md)
    * Completed, Verified with Burgers' Equation.
    * See [verify_hybrid_pinn.py](examples/verify_hybrid_pinn.py) for details.
* [NSGA-NODE development tasks](docs/NSGA_NODE_tasks.md)
    * TODO: Verify with Damped Oscillator.
    * TODO: [verify_node_oscillator.py](examples/verify_node_oscillator.py) for details.
 


## 🎯 Project Goals

1.  **Escape Local Minima:** Use NSGA-II to explore the parameter landscape and find better initialization basins for PINNs and Neural ODEs.
2.  **Pareto Optimality:** Explicitly trade off **Data Fitting** vs. **Physical Consistency** (or Model Parsimony) without manual loss weighting.
3.  **Hybrid Training:** seamless "handoff" protocols between evolutionary search (gradient-free/input-grad only) and standard gradient descent (ADAM).
4.  **High Performance:** Leverage `torch.func` (vmap) and `torchode` to parallelize evaluations across entire populations on GPU/CPU.

## 🛠️ Tech Stack & Prerequisites

-   **Python:** 3.12+
-   **Package Manager:** [uv](https://github.com/astral-sh/uv) (Extremely fast, modern Python package installer)
-   **Core Libraries:**
    -   `pytorch`: Deep learning and Autograd.
    -   `pymoo`: Multi-objective optimization algorithms.
    -   `torchode`: Parallel ODE solvers for Neural ODEs.
    -   `numpy`: Numerical operations.

## 📦 Installation

This project uses `uv` for dependency management.

1.  **Install uv** (if not already installed):
    ```bash
    # MacOS / Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Windows
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

2.  **Clone and Sync:**
    ```bash
    git clone <repository_url>
    cd <repository_name>

    # Create virtualenv and install dependencies
    uv sync
    ```

## ⚡ Quick Start

### 1. Validate PINN Module (Burgers' Equation)
Run the hybrid training verification script for the Physics-Informed Neural Network:

```bash
uv run examples/verify_hybrid_pinn.py --device cpu
# Or on GPU
uv run examples/verify_hybrid_pinn.py --device cuda
```

### 2. Validate NODE Module (Damped Oscillator)
Run the hybrid training verification script for the Neural ODE:

```bash
uv run examples/verify_node_oscillator.py --device cpu
```

## 📂 Project Structure

```text
.
├── src/
│   ├── nsga_neuro_evolution_core/  # Shared utilities (Genome interface, Gradient contexts)
│   ├── nsga_pinn/                  # NSGA-PINN specific logic (Evaluator, Problem, Orchestrator)
│   └── nsga_node/                  # NSGA-NODE specific logic (Dynamics, Torchode wrapper)
├── tests/
│   ├── core/                       # Unit tests for shared modules
│   ├── pinn/                       # Unit tests for PINN
│   └── node/                       # Unit tests for NODE
├── docs/
│   ├── NSGA_PINN_tasks.md          # Detailed development roadmap for PINN
│   └── NSGA_NODE_tasks.md          # Detailed development roadmap for NODE
├── examples/                       # Validation scripts and demos
├── pyproject.toml                  # Project configuration and dependencies
└── uv.lock                         # Lock file for reproducible builds
```

## 🔬 Methodology

### The Hybrid Loop
This framework implements a specific contract to alternate between optimization modes:

1.  **ADAM Phase (Epochs):**
    -   Standard Gradient Descent.
    -   **Mode:** `gradient` (Full Autograd).
    -   Optimizes weights locally.

2.  **NSGA-II Phase (Generations):**
    -   Evolutionary Strategy.
    -   **Mode:** `fitness` (Weights Frozen, Input Grads enabled for Residuals).
    -   Explores global landscape using a population centered on the current ADAM solution.

3.  **Handoff Protocol:**
    -   Select best "Knee Point" or balanced individual from the Pareto Front.
    -   **Reset** optimizer state (momentum buffers) to avoid stale history.
    -   Initialize next ADAM phase with the evolved weights.

### Performance Optimizations
-   **PINN Vectorization:** Uses `torch.vmap` and `torch.func.functional_call` to evaluate hundreds of networks in a single forward pass.
-   **NODE Batching:** Uses `torchode` with batched parameters to solve differential equations for the entire population in parallel.

## 📚 Documentation
Detailed development task lists and architectural blueprints can be found in the `docs/` directory:
-   [NSGA-PINN Roadmap](docs/NSGA_PINN_tasks.md)
-   [NSGA-NODE Roadmap](docs/NSGA_NODE_tasks.md)
-   [bibliography](docs/bibliography.md)

## 📄 License
[MIT License](LICENSE)
