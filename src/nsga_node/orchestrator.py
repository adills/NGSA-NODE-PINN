import torch
import torch.nn as nn
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from src.nsga_neuro_evolution_core.utils import adam_update_context

class HybridNodeOrchestrator:
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

    def train(
        self,
        epochs,
        adam_steps_per_epoch,
        nsga_gens_per_epoch,
        pop_size=50,
        verbose=True,
        adapt_adam_steps=True,
        min_adam_steps=1,
        max_adam_steps=50,
        improvement_window=3,
        nsga_improve_threshold=0.02,
        adam_improve_threshold=0.001,
        nsga_stagnation_threshold=0.005,
        nsga_noise_threshold=0.1,
        warmup_epochs_on_stagnation=0,
        adam_step_adjust=1,
    ):
        """
        Run the hybrid training loop for NODE.
        """
        history = []
        adam_loss_history = []
        adam_improve_history = []
        nsga_best_sum_history = []
        nsga_improve_history = []
        skip_nsga_epochs_remaining = 0

        for epoch in range(epochs):
            # --- ADAM Phase ---
            self.model.train()
            adam_loss_accum = 0.0
            schedule_action = None

            with adam_update_context():
                for i in range(adam_steps_per_epoch):
                    self.optimizer.zero_grad()

                    # Evaluate module directly (Graph attached)
                    loss_tuple = self.evaluator.evaluate_module(self.model)

                    # Scalarize: Simple Sum (Data + Correction)
                    total_loss = loss_tuple[0] + loss_tuple[1]

                    total_loss.backward()
                    self.optimizer.step()
                    adam_loss_accum += total_loss.item()

            avg_adam_loss = adam_loss_accum / max(1, adam_steps_per_epoch)
            if adam_loss_history:
                prev_adam_loss = adam_loss_history[-1]
                adam_improve = (prev_adam_loss - avg_adam_loss) / max(prev_adam_loss, 1e-12)
                adam_improve_history.append(adam_improve)
            adam_loss_history.append(avg_adam_loss)

            # --- NSGA Phase ---
            if skip_nsga_epochs_remaining > 0 or nsga_gens_per_epoch <= 0:
                if skip_nsga_epochs_remaining > 0:
                    skip_nsga_epochs_remaining -= 1
                res = None
                nsga_best_f = [np.nan, np.nan]
                nsga_best_sum = np.nan
            else:
                # 1. Get current genome
                current_genome = self.interface.to_genome(self.model)

                # 2. Setup Problem
                problem = self.problem_cls(self.evaluator, current_genome, bounds_radius=0.1)

                # 3. Run NSGA-II
                algorithm = NSGA2(pop_size=pop_size)

                res = minimize(
                    problem,
                    algorithm,
                    ('n_gen', nsga_gens_per_epoch),
                    verbose=False
                )

                # --- Handoff ---
                if res.F is not None and len(res.F) > 0:
                    best_genome = self.selector.select_knee_point(res.X, res.F)

                    # Load weights
                    state_dict = self.interface.genome_to_state_dict(best_genome)
                    self.model.load_state_dict(state_dict)

                    # Reset Optimizer
                    self.optimizer = self._create_optimizer()

                    nsga_best_f = res.F.min(axis=0)
                    nsga_best_sum = float(np.min(res.F.sum(axis=1)))
                else:
                    nsga_best_f = [np.nan, np.nan]
                    nsga_best_sum = np.nan

            if np.isfinite(nsga_best_sum):
                if nsga_best_sum_history:
                    prev_nsga_best_sum = nsga_best_sum_history[-1]
                    nsga_improve = (
                        (prev_nsga_best_sum - nsga_best_sum) / max(prev_nsga_best_sum, 1e-12)
                    )
                    nsga_improve_history.append(nsga_improve)
                nsga_best_sum_history.append(nsga_best_sum)

            # Schedule Adaptation
            if adapt_adam_steps:
                if (
                    len(nsga_improve_history) >= improvement_window
                    and len(adam_improve_history) >= improvement_window
                ):
                    recent_nsga = nsga_improve_history[-improvement_window:]
                    recent_adam = adam_improve_history[-improvement_window:]
                    nsga_consistent = all(val >= nsga_improve_threshold for val in recent_nsga)
                    adam_flat = all(val <= adam_improve_threshold for val in recent_adam)

                    if nsga_consistent and adam_flat:
                        new_steps = max(min_adam_steps, adam_steps_per_epoch - adam_step_adjust)
                        if new_steps != adam_steps_per_epoch:
                            adam_steps_per_epoch = new_steps
                            schedule_action = "reduce_adam_steps"
                    else:
                        nsga_stagnant = all(val <= nsga_stagnation_threshold for val in recent_nsga)
                        if len(nsga_best_sum_history) >= improvement_window:
                            recent_best = nsga_best_sum_history[-improvement_window:]
                            mean_best = np.mean(recent_best)
                            noise_ratio = np.std(recent_best) / max(mean_best, 1e-12)
                        else:
                            noise_ratio = 0.0

                        if nsga_stagnant or noise_ratio >= nsga_noise_threshold:
                            new_steps = min(max_adam_steps, adam_steps_per_epoch + adam_step_adjust)
                            if new_steps != adam_steps_per_epoch:
                                adam_steps_per_epoch = new_steps
                                schedule_action = "increase_adam_steps"
                            if warmup_epochs_on_stagnation > 0:
                                skip_nsga_epochs_remaining = max(
                                    skip_nsga_epochs_remaining, warmup_epochs_on_stagnation
                                )
                                schedule_action = "adam_warmup"

            nsga_front = res.F if res is not None else None

            if verbose:
                step_info = f"steps={adam_steps_per_epoch}"
                action_info = f", schedule={schedule_action}" if schedule_action else ""
                print(
                    f"Epoch {epoch}: ADAM Loss={avg_adam_loss:.6f}, "
                    f"NSGA Best F={nsga_best_f}, {step_info}{action_info}"
                )

            history.append({
                'epoch': epoch,
                'adam_loss': avg_adam_loss,
                'nsga_front': nsga_front,
                'nsga_best_f': nsga_best_f,
                'adam_steps_per_epoch': adam_steps_per_epoch,
                'nsga_ran': res is not None,
                'schedule_action': schedule_action
            })

        return history
