import torch
import torch.nn as nn
import numpy as np
import torchode
from torch.func import vmap, functional_call
from src.nsga_neuro_evolution_core.utils import nsga_evaluation_context
from src.nsga_node.dynamics import NodeDynamicsWrapper

class VectorizedNodeEvaluator:
    def __init__(self, model_template, interface, t_eval, y0, target_data, correction_fn=None,
                 solver_cls=torchode.Dopri5, atol=1e-6, rtol=1e-3):
        """
        Evaluator for Neural ODEs using torchode.

        Args:
            model_template: nn.Module defining the dynamics f(t, y).
            interface: Genome interface.
            t_eval: Time points to evaluate (T,).
            y0: Initial conditions (1, State_Dim) or (N_data, State_Dim).
                If (1, D), broadcasted to population.
                If (Pop, D), implies different IC per individual?
                Usually y0 is fixed for the problem.
                If we have multiple trajectories (batches of data), y0 would be (B, D).
                For now, assume single trajectory training: y0 is (1, D) or (D,).
            target_data: Ground truth trajectory (T, State_Dim) corresponding to t_eval.
            correction_fn: Callable(model, params, t, y) -> tensor. Returns the correction term magnitude.
                           If None, correction loss is 0.
            solver_cls: torchode solver class.
        """
        self.model_template = model_template
        self.interface = interface
        self.t_eval = t_eval
        self.y0 = y0
        self.target_data = target_data
        self.correction_fn = correction_fn

        # Wrapped dynamics that handles batched params
        self.dynamics = NodeDynamicsWrapper(model_template)

        # Solver config
        self.solver_cls = solver_cls
        self.atol = atol
        self.rtol = rtol

    def evaluate_population(self, population: np.ndarray, mode='fitness'):
        """
        Evaluate population.
        Args:
            population: (Pop_Size, Genome_Size)
            mode: 'fitness' or 'gradient'
        """
        if mode == 'fitness':
            return self._evaluate_fitness(population)
        elif mode == 'gradient':
            return self._evaluate_gradient(population)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _evaluate_fitness(self, population: np.ndarray):
        pop_size = population.shape[0]

        # 1. Prepare params
        batched_params = self.interface.batch_to_state_dict(population)

        # 2. Prepare ODE problem
        # Term needs to be created with with_args=True to accept params
        term = torchode.ODETerm(self.dynamics, with_args=True)

        # Step size controller
        step_method = self.solver_cls(term=term)
        step_size_controller = torchode.IntegralController(atol=self.atol, rtol=self.rtol, term=term)
        solver = torchode.AutoDiffAdjoint(step_method, step_size_controller)
        # Note: AutoDiffAdjoint is just a wrapper, we can control backprop via torch.no_grad or similar.
        # Ideally for fitness we don't build graph.

        # Problem construction
        # y0: We need to broadcast y0 to (Pop_Size, State_Dim)
        # Assuming self.y0 is (State_Dim,) or (1, State_Dim)
        if self.y0.ndim == 1:
            y0_expanded = self.y0.unsqueeze(0).repeat(pop_size, 1) # (Pop, D)
        else:
            y0_expanded = self.y0.repeat(pop_size, 1) # (Pop, D) if y0 is (1, D)

        # t_eval: (T,)
        # torchode expects t_eval to be (Batch, T) if batching?
        # Let's check torchode docs/usage in memory. "batched parameters".
        # If we want to solve separate ODEs, we need t_eval to be (Pop, T).
        t_eval_expanded = self.t_eval.unsqueeze(0).repeat(pop_size, 1) # (Pop, T)

        problem = torchode.InitialValueProblem(
            y0=y0_expanded,
            t_eval=t_eval_expanded
        )

        # 3. Solve
        with nsga_evaluation_context(self.model_template, self.y0): # y0 requires grad? Maybe not for ODE.
            # We strictly don't need gradients for weights.
            # We might want gradients w.r.t Inputs if we were doing PINN-style residuals,
            # but here we solve the ODE.
            # torchode handles the solve.

            # Note: nsga_evaluation_context sets requires_grad_(False) for model params.
            # And True for inputs.

            # solve
            sol = solver.solve(problem, args=batched_params)

            # sol.ys: (Pop, T, D)
            # Check for NaNs or failures?

            # 4. Compute Losses
            # Data Loss: MSE against target_data
            # target_data: (T, D) -> Broadcast to (Pop, T, D)
            preds = sol.ys
            target = self.target_data.unsqueeze(0) # (1, T, D)

            # MSE per individual
            # (Pop, T, D) -> (Pop,)
            data_loss = torch.mean((preds - target)**2, dim=(1, 2))

            # Correction Loss
            if self.correction_fn:
                # We need to evaluate correction term along the trajectory
                # preds: (Pop, T, D)
                # t_eval: (T,)

                # We can vmap the correction_fn over the population AND time?
                # Or just iterate/reshape.
                # correction_fn(model, params, t, y)
                # params is batched. y is (Pop, T, D).

                # Helper to map over batch
                def get_traj_correction(p_params, p_y, p_t):
                    # p_params: single dict
                    # p_y: (T, D)
                    # p_t: (T,)

                    # We need to vectorize over T inside here
                    def point_correction(t_scal, y_vec):
                        return self.correction_fn(self.model_template, p_params, t_scal, y_vec)

                    # Map over T
                    # in_dims: t_scal (0), y_vec (0)
                    corrs = vmap(point_correction)(p_t, p_y) # (T, ...)
                    return torch.mean(corrs**2) # L2 norm squared mean

                # Map over population
                # in_dims: params(0), y(0), t(0)
                correction_losses = vmap(get_traj_correction)(batched_params, preds, t_eval_expanded)
            else:
                correction_losses = torch.zeros_like(data_loss)

            # Stack objectives
            # (Pop, 2)
            losses = torch.stack([data_loss, correction_losses], dim=1)

            return losses.detach().cpu().numpy()

    def _evaluate_gradient(self, population: np.ndarray):
        # Usually single individual for ADAM
        results = []
        for i in range(population.shape[0]):
            genome = population[i]
            params = self.interface.genome_to_state_dict(genome)

            # Enable grads
            for p in params.values():
                p.requires_grad_(True)

            # We solve for SINGLE individual
            # Reshape inputs for batch size 1
            y0_single = self.y0.unsqueeze(0) if self.y0.ndim == 1 else self.y0
            t_eval_single = self.t_eval.unsqueeze(0)

            # We must batch params for torchode even if size 1
            # Dictionary of tensors (1, ...)
            batched_params_single = {k: v.unsqueeze(0) for k, v in params.items()}

            term = torchode.ODETerm(self.dynamics, with_args=True)
            step_method = self.solver_cls(term=term)
            step_size_controller = torchode.IntegralController(atol=self.atol, rtol=self.rtol, term=term)
            solver = torchode.AutoDiffAdjoint(step_method, step_size_controller)

            problem = torchode.InitialValueProblem(y0=y0_single, t_eval=t_eval_single)

            # Solve with autograd
            sol = solver.solve(problem, args=batched_params_single)
            preds = sol.ys # (1, T, D)

            target = self.target_data.unsqueeze(0)
            data_loss = torch.mean((preds - target)**2)

            if self.correction_fn:
                # Compute correction loss using functional_call directly on unbatched/single batch
                # preds[0]: (T, D)
                # t_eval: (T,)

                # We can loop over T or use vmap over T with unbatched params
                def point_correction(t_val, y_val):
                    return self.correction_fn(self.model_template, params, t_val, y_val)

                corrs = vmap(point_correction)(self.t_eval, preds[0])
                correction_loss = torch.mean(corrs**2)
            else:
                correction_loss = torch.tensor(0.0, device=preds.device)

            results.append(torch.stack([data_loss, correction_loss]))

        if len(results) == 1:
            return results[0].unsqueeze(0)
        return torch.stack(results)

    def evaluate_module(self, model: nn.Module):
        """
        Evaluate module directly using its current weights (for ADAM).
        """
        # Create params dict from model
        params = dict(model.named_parameters())

        # Reuse _evaluate_gradient logic but with current params
        # But we need to handle the 'params' dict format carefully.
        # Easier to just call _evaluate_gradient with current genome?
        # But that triggers to_genome -> to_state_dict conversion.
        # Efficient way:

        y0_single = self.y0.unsqueeze(0) if self.y0.ndim == 1 else self.y0
        t_eval_single = self.t_eval.unsqueeze(0)

        # Batch params (1, ...)
        batched_params = {k: v.unsqueeze(0) for k, v in params.items()}

        # Use existing wrapper which uses functional_call
        # IMPORTANT: Model weights in 'model' are updated by optimizer.
        # functional_call(model, params, ...) uses the tensors in params.
        # Since params = dict(model.named_parameters()), they ARE the leaves of the graph.

        term = torchode.ODETerm(self.dynamics, with_args=True)
        step_method = self.solver_cls(term=term)
        step_size_controller = torchode.IntegralController(atol=self.atol, rtol=self.rtol, term=term)
        solver = torchode.AutoDiffAdjoint(step_method, step_size_controller)

        problem = torchode.InitialValueProblem(y0=y0_single, t_eval=t_eval_single)

        sol = solver.solve(problem, args=batched_params)
        preds = sol.ys

        target = self.target_data.unsqueeze(0)
        data_loss = torch.mean((preds - target)**2)

        if self.correction_fn:
            def point_correction(t_val, y_val):
                return self.correction_fn(self.model_template, params, t_val, y_val)
            corrs = vmap(point_correction)(self.t_eval, preds[0])
            correction_loss = torch.mean(corrs**2)
        else:
            correction_loss = torch.tensor(0.0, device=preds.device)

        return torch.stack([data_loss, correction_loss])
