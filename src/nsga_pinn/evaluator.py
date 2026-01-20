import torch
import torch.nn as nn
from torch.func import vmap, functional_call
import numpy as np
from src.nsga_neuro_evolution_core.utils import nsga_evaluation_context

class VectorizedPinnEvaluator:
    def __init__(self, model_template, interface, input_data, target_data, input_physics, physics_residual_fn):
        """
        Args:
            model_template: The PyTorch model structure.
            interface: PytorchGenomeInterface instance.
            input_data: (N_data, D_in) tensor for data loss.
            target_data: (N_data, D_out) tensor for data loss.
            input_physics: (N_phys, D_in) tensor for physics loss.
            physics_residual_fn: Callable(model_fn, params, input_physics) -> loss scalar/tensor.
                                 Must be compatible with torch.func (vmap).
        """
        self.model_template = model_template
        self.interface = interface
        self.input_data = input_data
        self.target_data = target_data
        self.input_physics = input_physics
        self.physics_residual_fn = physics_residual_fn

    def evaluate_population(self, population: np.ndarray, mode='fitness'):
        """
        Evaluate the entire population.
        Args:
            population: (Pop_Size, Genome_Size) numpy array.
            mode: 'fitness' (vmap, no weight grads) or 'gradient' (standard autograd).
        Returns:
            losses: (Pop_Size, 2) array [DataLoss, PhysicsLoss].
                    Detached in fitness mode, Attached in gradient mode.
        """
        if mode == 'fitness':
            return self._evaluate_fitness(population)
        elif mode == 'gradient':
            return self._evaluate_gradient(population)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _evaluate_fitness(self, population: np.ndarray):
        population = population.astype(np.float32, copy=False)
        # Convert population to batched state_dict
        # Note: batch_to_state_dict returns tensors on device
        batched_params = self.interface.batch_to_state_dict(population)

        # We need to preserve the ability to compute gradients w.r.t inputs
        # so we use our context manager
        # Note: Since we use torch.func, the context manager mainly ensures
        # global model safety and input requires_grad.
        with nsga_evaluation_context(self.model_template, self.input_physics):

            # Define per-individual loss function
            def compute_single_loss(params):
                # Data Loss
                # functional_call uses the params to run the model
                # We define a stateless callable for the model
                def model_fn(x):
                    return functional_call(self.model_template, params, (x,))

                # Compute predictions
                preds = model_fn(self.input_data)
                data_loss = torch.mean((preds - self.target_data) ** 2)

                # Physics Loss
                # Delegate to the physics function
                # It must handle the functional model_fn and inputs
                phys_loss = self.physics_residual_fn(self.model_template, params, self.input_physics)

                return torch.stack([data_loss, phys_loss])

            # Vectorize over params (dim 0 of every tensor in params dict)
            # in_dims=(0,) means the first argument (params) is batched.
            # input data is closed over or fixed, so not in in_dims.
            # But wait, compute_single_loss only takes params.
            losses = vmap(compute_single_loss)(batched_params)

            return losses.detach().cpu().numpy()

    def _evaluate_gradient(self, population: np.ndarray):
        # For gradient mode, we usually evaluate a SINGLE individual (the current ADAM one)
        # But if population has size > 1, we can handle it.
        # However, 'gradient' mode implies we want a graph. vmap with autograd is tricky.
        # We will fallback to loop or single eval.

        results = []
        # Convert to list of state_dicts (unbatched) or just loop
        # Usually population is size 1 in this mode
        for i in range(population.shape[0]):
            genome = population[i]
            params = self.interface.genome_to_state_dict(genome)

            # Enable gradients for parameters to ensure graph construction
            for p in params.values():
                p.requires_grad_(True)

            # Load params into model? Or use functional call?
            # Standard ADAM uses the model directly usually.
            # But to keep consistency, let's use functional_call so we don't mutate self.model state persistently?
            # Actually, Orchestrator Step A says "Run standard gradient descent on current_model".
            # This implies using the standard model.

            # HOWEVER, this method is 'evaluate_population'.
            # If the orchestrator calls this for ADAM step, it expects graph.
            # Let's use functional_call with autograd enabled.

            def model_fn(x):
                return functional_call(self.model_template, params, (x,))

            # Data Loss
            preds = model_fn(self.input_data)
            data_loss = torch.mean((preds - self.target_data) ** 2)

            # Physics Loss
            # In gradient mode, physics_residual_fn needs to work with standard autograd if desired,
            # OR torch.func. But if we want to backward() through it to params, torch.func.grad inside is fine.
            phys_loss = self.physics_residual_fn(self.model_template, params, self.input_physics)

            results.append(torch.stack([data_loss, phys_loss]))

        if len(results) == 1:
            return results[0].unsqueeze(0) # (1, 2)
        return torch.stack(results) # (Pop, 2)

    def evaluate_module(self, model: nn.Module):
        """
        Evaluate a standard nn.Module (forward pass) for ADAM training.
        Returns (DataLoss, PhysicsLoss) attached to graph.
        """
        # Data Loss
        preds = model(self.input_data)
        data_loss = torch.mean((preds - self.target_data) ** 2)

        # Physics Loss
        # Use current parameters of the model
        params = dict(model.named_parameters())

        # Delegate to the physics function
        phys_loss = self.physics_residual_fn(self.model_template, params, self.input_physics)

        return torch.stack([data_loss, phys_loss])
