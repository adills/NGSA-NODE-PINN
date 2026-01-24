import argparse
import torch
import torch.nn as nn
import numpy as np
import time
from os.path import join
from torch.func import vmap, functional_call, jacrev, jacfwd

from src.nsga_neuro_evolution_core.interface import PytorchGenomeInterface
from src.nsga_neuro_evolution_core.selector import ParetoSelector
from src.nsga_pinn.evaluator import VectorizedPinnEvaluator
from src.nsga_pinn.problem import NsgaPinnProblem
from src.nsga_pinn.orchestrator import HybridPinnOrchestrator
from .third_party_burgers_runner import run_third_party_burgers, evaluate_third_party

# This test compares NSGA-PINN with PINN for the Burgers PDE.
# The third party code comes from https://github.com/ShotaDeguchi/PINN_Torch

# --- Burgers' Equation Setup ---
NU = 0.01 / np.pi

class PinnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )
    def forward(self, x):
        return self.net(x)

def burgers_physics_residual_fn(model_template, params, inputs):
    # inputs: (N, 2) -> (t, x)
    # Returns MSE of residual

    # Define single point function for derivatives
    def u_fn(tx):
        # tx: (2,)
        # output: (1,)
        # We assume functional_call handles 'params' dict correctly
        res = functional_call(model_template, params, (tx,))
        return res.squeeze() # scalar

    def compute_point_residual(tx):
        # First derivative: [u_t, u_x]
        # jacrev produces shape (2,)
        du_dtx = jacrev(u_fn)(tx)
        u_t = du_dtx[0]
        u_x = du_dtx[1]

        u = u_fn(tx)

        # Second derivative: Hessian
        # hessian = jacfwd(jacrev(u_fn))(tx) -> (2, 2)
        hess = jacfwd(jacrev(u_fn))(tx)
        u_xx = hess[1, 1]

        # Burgers: u_t + u*u_x - nu*u_xx = 0
        res = u_t + u * u_x - NU * u_xx
        return res ** 2

    # Vectorize over inputs
    # inputs is (N, 2)
    # vmap over dim 0
    residuals = vmap(compute_point_residual)(inputs)

    return torch.mean(residuals)

def generate_data(n_data, n_phys):
    # Domain: t in [0, 1], x in [-1, 1]

    # IC: t=0, x in [-1, 1]. u = -sin(x)
    # BC: x=-1, u=0. x=1, u=0. t in [0, 1]

    # Data points (IC + BC)
    # Randomly sample? Or grid?
    # Let's use random sampling for simplicity

    # IC
    x_ic = np.random.uniform(-1, 1, n_data // 2)
    t_ic = np.zeros_like(x_ic)
    u_ic = -np.sin(x_ic) # User spec: -sin(x)

    # BC
    t_bc = np.random.uniform(0, 1, n_data // 2)
    x_bc = np.random.choice([-1.0, 1.0], n_data // 2)
    u_bc = np.zeros_like(t_bc)

    inputs_data = np.vstack([
        np.stack([t_ic, x_ic], axis=1),
        np.stack([t_bc, x_bc], axis=1)
    ])
    targets_data = np.concatenate([u_ic, u_bc])[:, None] # (N, 1)

    # Physics Collocation Points
    t_phys = np.random.uniform(0, 1, n_phys)
    x_phys = np.random.uniform(-1, 1, n_phys)
    inputs_phys = np.stack([t_phys, x_phys], axis=1)

    return (
        torch.tensor(inputs_data, dtype=torch.float32),
        torch.tensor(targets_data, dtype=torch.float32),
        torch.tensor(inputs_phys, dtype=torch.float32)
    )

def plot_comparison(
    input_data,
    target_data,
    nsga_model,
    pinn_model=None,
    output_path=join("examples", "pinn_comparison.png"),
    n_times=4,
    x_points=200,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plot_comparison") from exc

    nsga_model.eval()
    if pinn_model is None:
        raise ValueError("pinn_model is required to plot PINN vs NSGA-PINN comparison")

    data_np = input_data.detach().cpu().numpy()
    t_min = float(data_np[:, 0].min())
    t_max = float(data_np[:, 0].max())
    x_min = float(data_np[:, 1].min())
    x_max = float(data_np[:, 1].max())

    if n_times < 1:
        raise ValueError("n_times must be >= 1")
    times = np.linspace(t_min, t_max, num=n_times)
    x_grid = np.linspace(x_min, x_max, num=x_points, dtype=np.float32)

    ncols = 2 if n_times > 1 else 1
    nrows = int(np.ceil(n_times / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    nsga_device = next(nsga_model.parameters()).device
    pinn_device = pinn_model.device

    with torch.no_grad():
        for i, t_val in enumerate(times):
            ax = axes[i]
            x_nsga = torch.from_numpy(x_grid).to(nsga_device)
            t_nsga = torch.full_like(x_nsga, float(t_val))
            tx_nsga = torch.stack([t_nsga, x_nsga], dim=1)
            nsga_pred = nsga_model(tx_nsga).detach().cpu().numpy().squeeze()

            x_pinn = torch.from_numpy(x_grid).to(pinn_device)
            t_pinn = torch.full_like(x_pinn, float(t_val))
            xt_pinn = torch.stack([x_pinn, t_pinn], dim=1)
            pinn_pred = pinn_model.forward(xt_pinn).detach().cpu().numpy().squeeze()

            ax.plot(x_grid, pinn_pred, label="PINN")
            ax.plot(x_grid, nsga_pred, label="NSGA-PINN")
            ax.set_title(f"t = {t_val:.2f}")
            ax.set_xlabel("x")
            ax.set_ylabel("u")
            ax.grid(alpha=0.3)

    for j in range(n_times, len(axes)):
        axes[j].axis("off")

    axes[0].legend(loc="best")

    if output_path:
        fig.savefig(output_path, dpi=200)
    return fig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--adam_steps", type=int, default=10)
    parser.add_argument("--nsga_gens", type=int, default=100)
    parser.add_argument("--no_compare_third_party", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--no_plot_comparison", action="store_true")
    parser.add_argument("--plot_path", type=str, default=join("examples", "pinn_comparison.png"))
    parser.add_argument("--third_party_epochs", type=int, default=2000)
    parser.add_argument("--third_party_f_mntr", type=int, default=10)
    args = parser.parse_args()
    if args.no_compare_third_party:
        args.compare_third_party = False
    else:
        args.compare_third_party = True
    if args.no_plot_comparison:
        args.plot_comparison = False    # noqa: E701
    else:
        args.plot_comparison = True    # noqa: E701

    device = torch.device(args.device)
    print(f"Running on {device}")

    # Setup
    model = PinnNet().to(device)
    interface = PytorchGenomeInterface(model)

    input_data, target_data, input_phys = generate_data(n_data=200, n_phys=1000)
    input_data = input_data.to(device)
    target_data = target_data.to(device)
    input_phys = input_phys.to(device)
    input_phys.requires_grad_(True) # Important for some contexts, though torch.func might ignore

    evaluator = VectorizedPinnEvaluator(
        model, interface, input_data, target_data, input_phys, burgers_physics_residual_fn
    )

    selector = ParetoSelector()

    orchestrator = HybridPinnOrchestrator(
        model, interface, evaluator, NsgaPinnProblem, selector,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs={'lr': 1e-3}
    )

    print("Starting Hybrid Training...")
    start_time = time.time()

    history = orchestrator.train(
        epochs=args.epochs,
        adam_steps_per_epoch=args.adam_steps,
        nsga_gens_per_epoch=args.nsga_gens,
        pop_size=20, # Small pop for speed in verification
        verbose=args.verbose,
        pareto_gif_path=join("tests", "pareto_front_pinn.gif"),
        pareto_gif_fps=1,
        pareto_gif_repeat_last=True,
    )

    end_time = time.time()
    hybrid_time = end_time - start_time
    print(f"Training complete in {hybrid_time:.2f}s")

    # Final eval
    final_loss = evaluator.evaluate_module(model)
    print(f"Final Loss: Data={final_loss[0].item():.6f}, Phys={final_loss[1].item():.6f}")

    if args.compare_third_party:
        third_party_epochs = args.epochs if args.third_party_epochs is None else args.third_party_epochs
        pinn_third_party, pinn_time = run_third_party_burgers(
            input_data=input_data,
            target_data=target_data,
            input_phys=input_phys,
            device=device,
            epochs=third_party_epochs,
            f_mntr=args.third_party_f_mntr,
            verbose=args.verbose
        )
        tp_data_loss, tp_phys_loss = evaluate_third_party(
            pinn_third_party, input_data, target_data, input_phys
        )
        print(f"Third-party PINN Loss: Data={tp_data_loss.item():.6f}, Phys={tp_phys_loss.item():.6f}")
        print(f"           Number of epochs: {third_party_epochs}")
        print(f"           Training complete in {pinn_time:.2f}s")
        print(f"NSGA-PINN  Final Loss: Data={final_loss[0].item():.6f}, Phys={final_loss[1].item():.6f}")
        print(f"           Number of epochs: {args.epochs}")
        print(f"           Number of ADAM steps: {args.adam_steps}")
        print(f"           Number of NSGA gens: {args.nsga_gens}")
        print(f"           Training complete in {hybrid_time:.2f}s")
        if args.plot_comparison:
            plot_comparison(
                input_data=input_data,
                target_data=target_data,
                nsga_model=model,
                pinn_model=pinn_third_party,
                output_path=args.plot_path
            )
            print(f"Saved comparison plot to {args.plot_path}")
    elif args.plot_comparison:
        print("--plot_comparison requires --compare_third_party to provide a PINN baseline.")
if __name__ == "__main__":
    main()
