# Hybrid NSGA-PINN and NSGA-NODE Framework

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

A robust, modular framework for **Multi-Objective Evolutionary Scientific Machine Learning**. This project implements the **Hybrid NSGA-PINN** and **NSGA-NODE** architectures, combining the local refinement capabilities of gradient-based optimization (ADAM) with the global search and multi-objective handling of evolutionary algorithms (NSGA-II).

## 🚀 Project Goals

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

## 📄 License
[MIT License](LICENSE)
