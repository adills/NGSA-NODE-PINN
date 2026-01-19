import torch
import torch.nn as nn
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from src.nsga_neuro_evolution_core.utils import adam_update_context

class HybridPinnOrchestrator:
    def __init__(self, model, interface, evaluator, problem_cls, selector,
                 optimizer_cls=torch.optim.Adam, optimizer_kwargs=None):
        self.model = model
        self.interface = interface
        self.evaluator = evaluator
        self.problem_cls = problem_cls
        self.selector = selector
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {'lr': 1e-3}

        # Initial optimizer
        self.optimizer = self._create_optimizer()

    def _create_optimizer(self):
        return self.optimizer_cls(self.model.parameters(), **self.optimizer_kwargs)

    def train(self, epochs, adam_steps_per_epoch, nsga_gens_per_epoch, pop_size=50, verbose=True):
        """
        Run the hybrid training loop.
        """
        history = []

        for epoch in range(epochs):
            # --- ADAM Phase ---
            self.model.train()
            adam_loss_accum = 0.0

            with adam_update_context():
                for i in range(adam_steps_per_epoch):
                    self.optimizer.zero_grad()

                    # Evaluate module directly
                    loss_tuple = self.evaluator.evaluate_module(self.model)

                    # Scalarize: Simple Sum (Data + Physics)
                    total_loss = loss_tuple[0] + loss_tuple[1]

                    total_loss.backward()
                    self.optimizer.step()
                    adam_loss_accum += total_loss.item()

            avg_adam_loss = adam_loss_accum / max(1, adam_steps_per_epoch)

            # --- NSGA Phase ---
            # 1. Get current genome
            current_genome = self.interface.to_genome(self.model)

            # 2. Setup Problem
            # Center bounds around current weights
            problem = self.problem_cls(self.evaluator, current_genome, bounds_radius=0.1)

            # 3. Run NSGA-II
            # Initialize population?
            # Pymoo initializes randomly within bounds.
            # Since bounds are centered on current genome, it explores the neighborhood.
            # We can also seed the population with the current genome if we want.
            # But Random around neighborhood is standard "Jump" strategy.

            algorithm = NSGA2(pop_size=pop_size)

            res = minimize(problem,
                           algorithm,
                           ('n_gen', nsga_gens_per_epoch),
                           verbose=False)

            # --- Handoff ---
            # Select best from Pareto Front
            if len(res.F) > 0:
                best_genome = self.selector.select_knee_point(res.X, res.F)

                # Load weights
                state_dict = self.interface.genome_to_state_dict(best_genome)
                self.model.load_state_dict(state_dict)

                # Reset Optimizer to clear momentum
                self.optimizer = self._create_optimizer()

                nsga_best_f = res.F.min(axis=0)
            else:
                nsga_best_f = [np.nan, np.nan]

            if verbose:
                print(f"Epoch {epoch}: ADAM Loss={avg_adam_loss:.6f}, NSGA Best F={nsga_best_f}")

            history.append({
                'epoch': epoch,
                'adam_loss': avg_adam_loss,
                'nsga_front': res.F,
                'nsga_best_f': nsga_best_f
            })

        return history
