import argparse
import torch
import torch.nn as nn
import numpy as np
import time
from torch.func import vmap, functional_call, jacrev, jacfwd

from src.nsga_neuro_evolution_core.interface import PytorchGenomeInterface
from src.nsga_pinn.evaluator import VectorizedPinnEvaluator
from src.nsga_pinn.problem import NsgaPinnProblem
from src.nsga_pinn.selector import ParetoSelector
from src.nsga_pinn.orchestrator import HybridPinnOrchestrator
from .third_party_burgers_runner import run_third_party_burgers, evaluate_third_party

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--adam_steps", type=int, default=10)
    parser.add_argument("--nsga_gens", type=int, default=50)
    parser.add_argument("--compare_third_party", action="store_true")
    parser.add_argument("--third_party_epochs", type=int, default=None)
    parser.add_argument("--third_party_f_mntr", type=int, default=None)
    args = parser.parse_args()

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
        pop_size=20 # Small pop for speed in verification
    )

    end_time = time.time()
    print(f"Training complete in {end_time - start_time:.2f}s")

    # Final eval
    final_loss = evaluator.evaluate_module(model)
    print(f"Final Loss: Data={final_loss[0].item():.6f}, Phys={final_loss[1].item():.6f}")

    if args.compare_third_party:
        third_party_epochs = args.epochs if args.third_party_epochs is None else args.third_party_epochs
        pinn_third_party = run_third_party_burgers(
            input_data=input_data,
            target_data=target_data,
            input_phys=input_phys,
            device=device,
            epochs=third_party_epochs,
            f_mntr=args.third_party_f_mntr
        )
        tp_data_loss, tp_phys_loss = evaluate_third_party(
            pinn_third_party, input_data, target_data, input_phys
        )
        print(f"Third-party Loss: Data={tp_data_loss.item():.6f}, Phys={tp_phys_loss.item():.6f}")

if __name__ == "__main__":
    main()
