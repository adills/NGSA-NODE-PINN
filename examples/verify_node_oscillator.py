import argparse
import torch
import torch.nn as nn
import numpy as np
import time
from os.path import join
import matplotlib.pyplot as plt

from src.nsga_neuro_evolution_core.interface import PytorchGenomeInterface
from src.nsga_neuro_evolution_core.selector import ParetoSelector
from src.nsga_node.evaluator import VectorizedNodeEvaluator
from src.nsga_node.problem import NsgaNodeProblem
from src.nsga_node.orchestrator import HybridNodeOrchestrator

# --- Problem Setup ---
# True System: x'' + 0.5 x' + 2.0 x = 0
# State y = [x, v]
# y' = [v, -2x - 0.5v]

class OscillatorModel(nn.Module):
    def __init__(self):
        super().__init__()
        # NN approximates the damping term u(v) = -0.5 * v
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, t, y):
        # y is (Batch, 2) [x, v] or (2,)
        # Handle shape
        if y.ndim == 1:
            x, v = y[0], y[1]
            v_in = v.unsqueeze(0) # (1,)
        else:
            x, v = y[..., 0], y[..., 1]
            v_in = v.unsqueeze(-1) # (Batch, 1)

        correction = self.net(v_in).squeeze(-1) # (Batch,)

        # Dynamics: x' = v, v' = -2x + correction
        # True physics: v' = -2x - 0.5v
        dxdt = v
        dvdt = -2.0 * x + correction

        return torch.stack([dxdt, dvdt], dim=-1)

class ClassicalBaselineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, t):
        return self.net(t)

def train_classical_baseline(t_train, x0, v0, w0, xi, steps=1000, lr=1e-3, verbose=True):
    device = t_train.device
    model = ClassicalBaselineNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if verbose:
        step_iter = range(steps)
        pbar = None
    else:
        from tqdm import tqdm
        pbar = tqdm(range(steps), desc="Classical", unit="Step")
        step_iter = pbar

    for _ in step_iter:
        t = t_train.clone().detach().requires_grad_(True)
        x = model(t)
        x_t = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
        x_tt = torch.autograd.grad(x_t, t, torch.ones_like(x_t), create_graph=True)[0]
        residual = x_tt + xi * x_t + (w0 ** 2) * x

        t0 = torch.zeros((1, 1), device=device, requires_grad=True)
        x0_pred = model(t0)
        x_t0 = torch.autograd.grad(x0_pred, t0, torch.ones_like(x0_pred), create_graph=True)[0]

        loss_res = torch.mean(residual ** 2)
        loss_ic = torch.mean((x0_pred - x0) ** 2) + torch.mean((x_t0 - v0) ** 2)
        loss = loss_res + loss_ic

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if pbar is not None:
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

    if pbar is not None:
        pbar.close()

    return model

def get_correction_term(model, params, t, y):
    # Helper to extract ONLY the NN output magnitude for loss
    # functional_call used by Evaluator passes 'params'
    from torch.func import functional_call

    # We need to expose the inner net call
    # But we can just run the model and compute correction logic again?
    # Or, cleaner: define a sub-function.
    # Let's reproduce the forward logic strictly for the NN part.

    # model.net is the NN.
    # params is a dict of all weights.
    # We need to filter params for 'net'?
    # No, functional_call handles the full module hierarchy if params matches keys.

    if y.ndim == 1:
        v = y[1].unsqueeze(0)
    else:
        v = y[..., 1].unsqueeze(-1)

    # We only want to run self.net(v).
    # But functional_call(model, ...) runs model.forward().
    # We need functional_call(model.net, subset_params, (v,))
    # OR redefine forward to return correction and we handle it in wrapper? No wrapper expects dy/dt.

    # Solution: Run full forward, extract components?
    # dy_dt = functional_call(model, params, (t, y))
    # dy_dt = [v, -2x + u]
    # u = dy_dt[1] + 2x

    dydt = functional_call(model, params, (t, y))
    if y.ndim == 1:
        x = y[0]
        dvdt = dydt[1]
    else:
        x = y[..., 0]
        dvdt = dydt[..., 1]

    correction = dvdt + 2.0 * x
    return torch.abs(correction) # Return magnitude

def analytical_solution(t, x0=1.0, v0=0.0):
    # underdamped
    w0 = np.sqrt(2.0)
    xi = 0.5 / (2 * w0) # xi = c / (2*sqrt(k*m)) = 0.5 / (2*sqrt(2)) ~ 0.176
    # damp = 0.5. m=1, k=2.
    # r = -0.25 +/- i * sqrt(2 - 0.25^2)
    # x(t) = exp(-0.25t) * (A cos(wt) + B sin(wt))

    alpha = 0.25
    omega = np.sqrt(2.0 - 0.25**2)

    # x(0) = A = x0
    A = x0
    # v(0) = -0.25 A + omega B = v0
    # B = (v0 + 0.25 A) / omega
    B = (v0 + alpha * x0) / omega

    x = np.exp(-alpha * t) * (A * np.cos(omega * t) + B * np.sin(omega * t))
    return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--adam_steps", type=int, default=20)
    parser.add_argument("--nsga_gens", type=int, default=20)
    parser.add_argument("--pop_size", type=int, default=50)
    parser.add_argument("--output_plot", type=str, default="examples/node_oscillator_comparison.png")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--pareto_limits", nargs=2, type=float, default=None)

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Running on {device}")

    # 1. Generate Training Data (Simulated Experiment)
    # We use the analytical solution to generate 'clean' data
    t_train = torch.linspace(0, 10, 50).to(device)
    y0_train = torch.tensor([1.0, 0.0]).to(device) # x=1, v=0

    # Analytical states
    x_true_np = analytical_solution(t_train.cpu().numpy())
    # v_true? Numerical derivative or analytical
    # Let's just use numerical for simplicity or analytical derivation
    # v = -alpha * x + exp * (-A w sin + B w cos)
    alpha = 0.25
    omega = np.sqrt(2.0 - 0.25**2)
    A = 1.0; B = (0 + 0.25)/omega
    t_np = t_train.cpu().numpy()
    v_true_np = np.exp(-alpha * t_np) * (-alpha*(A*np.cos(omega*t_np)+B*np.sin(omega*t_np)) + omega*(-A*np.sin(omega*t_np)+B*np.cos(omega*t_np)))

    y_true = torch.tensor(np.stack([x_true_np, v_true_np], axis=1), dtype=torch.float32).to(device)

    # 2. Setup Models
    hybrid_model = OscillatorModel().to(device)
    baseline_model = OscillatorModel().to(device) # Pure ADAM
    # Ensure same init?
    baseline_model.load_state_dict(hybrid_model.state_dict())

    interface = PytorchGenomeInterface(hybrid_model)

    # 3. Setup Evaluators
    evaluator_hybrid = VectorizedNodeEvaluator(
        hybrid_model, interface, t_train, y0_train, y_true,
        correction_fn=get_correction_term
    )

    # 4. Train Hybrid
    print(f"\n--- Training Hybrid NSGA-NODE, GA generations: {args.nsga_gens} ---")
    problem_cls = NsgaNodeProblem
    selector = ParetoSelector()

    orchestrator = HybridNodeOrchestrator(
        hybrid_model, interface, evaluator_hybrid, problem_cls, selector,
        optimizer_kwargs={'lr': 0.01}
    )

    start_time = time.time()
    hist_hybrid = orchestrator.train(
        epochs=args.epochs,
        adam_steps_per_epoch=args.adam_steps,
        nsga_gens_per_epoch=args.nsga_gens,
        pop_size=args.pop_size,
        verbose=args.verbose,
        pareto_gif_path=join("examples", "pareto_front_node.gif"),
        pareto_gif_fps=1,
        pareto_gif_repeat_last=True,
        pareto_limits=args.pareto_limits,
    )
    hybrid_time = time.time() - start_time
    print(f"Hybrid Time: {hybrid_time:.2f}s")

    # 5. Train Baseline (Pure ADAM)
    print("\n--- Training Baseline (Pure ADAM) ---")
    # We can use the same orchestrator but with 0 NSGA gens
    # But we need a separate orchestrator instance/model

    baseline_interface = PytorchGenomeInterface(baseline_model)
    evaluator_baseline = VectorizedNodeEvaluator(
        baseline_model, baseline_interface, t_train, y0_train, y_true,
        correction_fn=get_correction_term
    )

    # We iterate manually for baseline to match 'Epochs' concept purely on ADAM
    # Or use orchestrator with 0 gens.
    # Total ADAM steps = epochs * adam_steps.
    # If hybrid does (adam + nsga) * epochs.
    # Baseline should probably do same amount of wall time or same amount of evaluations?
    # Fair comparison usually means same epochs/steps if we assume NSGA adds value.
    # Let's match the number of ADAM steps per epoch.

    orchestrator_baseline = HybridNodeOrchestrator(
        baseline_model, baseline_interface, evaluator_baseline, problem_cls, selector,
        optimizer_kwargs={'lr': 0.01}
    )

    start_time = time.time()
    hist_baseline = orchestrator_baseline.train(
        epochs=args.epochs,
        adam_steps_per_epoch=args.adam_steps,
        nsga_gens_per_epoch=0, # Disable NSGA
        pop_size=args.pop_size,
        adapt_adam_steps=False,
        verbose=args.verbose
    )
    baseline_time = time.time() - start_time
    print(f"Baseline Time: {baseline_time:.2f}s")

    # 6. Train Classical Baseline (NN + Residual)
    print("\n--- Training Classical Baseline (NN Residual) ---")
    w0 = np.sqrt(2.0)
    xi = 0.5
    x0 = 1.0
    v0 = 0.0
    classical_steps = args.epochs * args.adam_steps
    start_time = time.time()
    classical_model = train_classical_baseline(
        t_train.view(-1, 1),
        x0=x0,
        v0=v0,
        w0=w0,
        xi=xi,
        steps=classical_steps,
        lr=0.01,
        verbose=args.verbose
    )
    classical_time = time.time() - start_time
    print(f"Classical Time: {classical_time:.2f}s")
    t_eval = t_train.view(-1, 1).clone().detach().requires_grad_(True)
    x_pred = classical_model(t_eval)
    v_pred = torch.autograd.grad(x_pred, t_eval, torch.ones_like(x_pred), create_graph=False)[0]
    y_classical_t = torch.cat([x_pred, v_pred], dim=1).detach()
    y_classical = y_classical_t.cpu().numpy()
    tail_idx = int(0.8 * len(y_classical_t))
    tail_classical = y_classical_t[tail_idx:]
    tm_classical = torch.mean(tail_classical, dim=0).cpu().numpy()
    ts_classical = torch.std(tail_classical, dim=0).cpu().numpy()

    # 7. Evaluation & Statistics
    print("\n--- Evaluation ---")

    def evaluate_model(mod, name):
        # Predict on same t_train
        with torch.no_grad():
            # Use evaluator helper
            # returns (Data, Correction)
            loss = evaluator_baseline.evaluate_module(mod) # Reusing evaluator as it's stateless w.r.t model instance passed to evaluate_module
            # Wait, evaluate_module uses self.dynamics which wraps self.model_template.
            # Evaluator is bound to a specific model_template instance.
            # So evaluator_hybrid is bound to hybrid_model.
            # evaluator_baseline is bound to baseline_model.
            pass

        # Let's get trajectory
        # We need to solve it.
        # Can use evaluator internal solver or just a manual solve loop?
        # Use the evaluator logic but we need the output ys.
        # VectorizedNodeEvaluator doesn't expose 'predict'.
        # Let's use the underlying dynamics wrapper and torchode manually.

        wrapper = evaluator_baseline.dynamics # same wrapper class
        # But wait, evaluator_baseline.dynamics is bound to baseline_model.
        # We need a wrapper bound to 'mod'.

        # Just use evaluator corresponding to the model
        if mod is hybrid_model:
            ev = evaluator_hybrid
        else:
            ev = evaluator_baseline

        # Access internal solve (hacky but quick)
        # Re-instantiate needed parts
        import torchode
        term = torchode.ODETerm(ev.dynamics, with_args=True)
        step_method = torchode.Dopri5(term=term)
        controller = torchode.IntegralController(atol=1e-6, rtol=1e-3, term=term)
        solver = torchode.AutoDiffAdjoint(step_method, controller)

        y0_exp = ev.y0.unsqueeze(0)
        t_exp = ev.t_eval.unsqueeze(0)

        # Current params
        params = dict(mod.named_parameters())
        batched = {k: v.unsqueeze(0) for k, v in params.items()}

        problem = torchode.InitialValueProblem(y0=y0_exp, t_eval=t_exp)
        sol = solver.solve(problem, args=batched)
        ys = sol.ys[0] # (T, D)

        # MSE
        mse = torch.mean((ys - ev.target_data)**2).item()

        # Tail stats (last 20%)
        tail_idx = int(0.8 * len(ys))
        tail = ys[tail_idx:]
        tail_mean = torch.mean(tail, dim=0)
        tail_std = torch.std(tail, dim=0)

        return ys.detach().cpu().numpy(), mse, tail_mean.detach().cpu().numpy(), tail_std.detach().cpu().numpy()

    y_hybrid, mse_hybrid, tm_hybrid, ts_hybrid = evaluate_model(hybrid_model, "Hybrid")
    y_base, mse_base, tm_base, ts_base = evaluate_model(baseline_model, "Baseline")
    mse_classical = torch.mean((y_classical_t - y_true) ** 2).item()
    hybrid_steps = sum(item["adam_steps_per_epoch"] for item in hist_hybrid)
    baseline_steps = sum(item["adam_steps_per_epoch"] for item in hist_baseline)

    print(f"Hybrid NODE MSE   ({hybrid_steps:04d} ADAM Steps, {hybrid_time:6.2f}s): {mse_hybrid:.4f}")
    print(f"Baseline NODE MSE ({baseline_steps:04d} ADAM Steps, {baseline_time:6.2f}s): {mse_base:.4f}")
    print(f"Classical NN MSE  ({classical_steps:04d} ADAM Steps, {classical_time:6.2f}s): {mse_classical:.4f}")
    formatted_list_m = ' '.join(f"{num:7.4f}" for num in tm_hybrid)
    formatted_list_s = ' '.join(f"{num:7.4f}" for num in ts_hybrid)
    print(f"Hybrid NODE Tail Mean: {formatted_list_m}, Std: {formatted_list_s}")
    formatted_list_m = ' '.join(f"{num:7.4f}" for num in tm_base)
    formatted_list_s = ' '.join(f"{num:7.4f}" for num in ts_base)
    print(f"Baseline NODE Tail Mean: {formatted_list_m}, Std: {formatted_list_s}")
    formatted_list_m = ' '.join(f"{num:7.4f}" for num in tm_classical)
    formatted_list_s = ' '.join(f"{num:7.4f}" for num in ts_classical)
    print(f"Classical NN Tail Mean: {formatted_list_m}, Std: {formatted_list_s}")
    exact_tail = y_true[int(0.8*len(y_true)):].mean(dim=0).cpu().numpy()
    formatted_list = ' '.join(f"{num:7.4f}" for num in exact_tail)
    print(f"Exact Tail Mean: {formatted_list}")

    # 7. Plotting
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(t_np, x_true_np, 'k-', label='Exact', linewidth=2)
        ax.plot(t_np, y_hybrid[:, 0], 'r--', label='NSGA-NODE', linewidth=2)
        ax.plot(t_np, y_base[:, 0], 'b:', label='Baseline (ADAM)', linewidth=2)
        ax.plot(t_np, y_classical[:, 0], 'g-.', label='Classical NN Baseline', linewidth=2)

        ax.set_title("Damped Oscillator Trajectory Comparison")
        ax.set_xlabel("Time")
        ax.set_ylabel("Position (x)")
        ax.legend()
        ax.grid(True)

        plt.savefig(args.output_plot)
        print(f"Plot saved to {args.output_plot}")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    main()
