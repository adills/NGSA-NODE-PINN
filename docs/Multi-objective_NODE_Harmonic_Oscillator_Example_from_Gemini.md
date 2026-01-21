## Baseline (Non-Hybrid) Multi-Objective NODE Harmonic Oscillator

**Purpose**: A simple, non-hybrid baseline for later comparison with the hybrid NSGA-NODE approach in `docs/NSGA_NODE_tasks.md`. This baseline uses **NSGA-II only** (no ADAM handoff), and keeps the model small so it is easy to run and interpret.

**Key properties**
- NSGA-II only (population-based, gradient-free).
- `n_obj = 2` (Data Loss + Correction Magnitude or Physics Residual).
- Forward-only `torchode` solve (no adjoint, no autograd through the solver).
- Small parameterization (e.g., tiny NN correction or direct damping/stiffness parameters).

### 1. Damped Oscillator Equation

The core is the second-order ODE: $m\ddot{x} + b\dot{x} + kx = 0$. We convert this to two first-order equations:

- $z_1 = x$ (position)
- $z_2 = \dot{x}$ (velocity)
- $\dot{z}_1 = z_2$
- $\dot{z}_2 = -\frac{b}{m}z_2 - \frac{k}{m}z_1$

### 2. Baseline Dynamics (Physics + NN Correction)

Define a hybrid vector field with a small correction:

- $f_{phys}(z) = \begin{bmatrix} z_2 \\ -\frac{k}{m} z_1 \end{bmatrix}$
- $u_{\theta}(z)$ is a small neural correction term (tiny MLP or direct parameters).
- $f_{\theta}(z) = f_{phys}(z) + u_{\theta}(z)$

### 3. Objectives (n_obj=2)

Use two objectives:

1. **Data Loss:** MSE between predicted trajectory and observed data.
2. **Correction Magnitude:** L2 norm of $u_{\theta}(z)$ over the trajectory (a physics proxy).

### 4. Optimization (NSGA-II Only)

No ADAM phase. The solver is forward-only and the fitness evaluation is fully detached:

- Use `pymoo` with `n_obj=2`.
- Evaluate a population of parameters with `torchode`.
- Return objective values only (no gradients).

### 5. Sketch (Pseudo-Code)

```python
from pymoo.core.problem import Problem

class DampedOscillatorProblem(Problem):
    def __init__(self, data_t, data_y, solver, genome_interface, device="cpu"):
        super().__init__(n_var=param_dim, n_obj=2, xl=lower_bounds, xu=upper_bounds)
        self.data_t = data_t.to(device)
        self.data_y = data_y.to(device)
        self.solver = solver
        self.genome_interface = genome_interface

    def _evaluate(self, X, out, *args, **kwargs):
        # X: (pop, param_dim)
        batched_state_dict = self.genome_interface.batch_to_state_dict(X)
        sol = self.solver.solve(problem, args=batched_state_dict)
        data_loss = mse(sol.ys, self.data_y)            # (pop,)
        correction_loss = l2_correction(sol, X)         # (pop,)
        out["F"] = stack(data_loss, correction_loss)    # (pop, 2)
```

### 6. Not in Scope (Baseline Only)

- No ADAM phase or handoff.
- No adjoint or backpropagation through the solver.
- No hybrid schedule logic.

*AI responses may include mistakes.*

[1] [https://openreview.net/references/pdf?id=drZYgwyC9D](https://openreview.net/references/pdf?id=drZYgwyC9D)

[2] [https://openreview.net/pdf?id=uiKVKTiUYB0](https://openreview.net/pdf?id=uiKVKTiUYB0)

[3] [https://torchode.readthedocs.io/](https://torchode.readthedocs.io/)

[4] [https://medium.com/@souvikat/solving-differential-equations-with-physics-informed-neural-networks-pinns-a-mild-introduction-5570634149b8](https://medium.com/@souvikat/solving-differential-equations-with-physics-informed-neural-networks-pinns-a-mild-introduction-5570634149b8#:~:text=The%20Damped%20Harmonic%20Oscillator:%20Governing%20Physics%20and,incorporate%20physics%20directly%20into%20a%20learning%20model.)

[5] [https://www.sciencedirect.com/science/article/pii/S0925231225020302](https://www.sciencedirect.com/science/article/pii/S0925231225020302)

[6] [https://www.vizuaranewsletter.com/p/recover-unknown-science-using-machine](https://www.vizuaranewsletter.com/p/recover-unknown-science-using-machine#:~:text=The%20known%20part%20of%20the%20system%20is,d%5E2x/dt%5E2+kx+NN%CE%B8=0%2C%20where%20NN%CE%B8%20is%20the%20learned%20function.)

[7] [https://github.com/martenlienen/torchode](https://github.com/martenlienen/torchode)

[8] [https://www.sciencedirect.com/science/article/abs/pii/S0378778823007545](https://www.sciencedirect.com/science/article/abs/pii/S0378778823007545#:~:text=Minimizing%20an%20error%20metric%20using%20a%20specific,to%20address%20single%2D%20and%20multi%2Dobjective%20optimization%20problems.)
