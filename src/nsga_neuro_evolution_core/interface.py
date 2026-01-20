import torch
import torch.nn as nn
import numpy as np
from typing import Dict, OrderedDict

class PytorchGenomeInterface:
    """
    Bridge between flat genetic vectors (numpy) and hierarchical PyTorch state dictionaries.
    """
    def __init__(self, model_template: nn.Module):
        self.model_template = model_template
        self.param_shapes = {}
        self.param_names = []
        self.total_params = 0

        # Analyze model structure
        for name, param in model_template.named_parameters():
            self.param_names.append(name)
            self.param_shapes[name] = param.shape
            self.total_params += param.numel()

    def to_genome(self, model: nn.Module) -> np.ndarray:
        """Flatten all trainable parameters into a single 1D numpy array."""
        genome = []
        for param in model.parameters():
            genome.append(param.detach().cpu().numpy().flatten())
        return np.concatenate(genome)

    def genome_to_state_dict(self, individual: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Convert a single genome (1D array) back to a state_dict.
        Useful for loading weights into a model.
        """
        if individual.ndim != 1:
            raise ValueError(f"Expected 1D individual, got {individual.shape}")

        state_dict = {}
        idx = 0
        for name in self.param_names:
            shape = self.param_shapes[name]
            count = int(np.prod(shape))
            param_data = individual[idx : idx + count]
            idx += count

            # Convert to tensor
            state_dict[name] = torch.from_numpy(param_data).reshape(shape).to(dtype=self.model_template.parameters().__next__().dtype, device=self.model_template.parameters().__next__().device)

        return state_dict

    def batch_to_state_dict(self, population: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Convert a population of genomes (Pop_Size, Num_Params) to a batched state_dict.
        Output tensors will have shape (Pop_Size, ...original_dims...).
        Useful for torch.func.functional_call with vmap.
        """
        if population.ndim != 2:
             raise ValueError(f"Expected 2D population, got {population.shape}")

        pop_size = population.shape[0]
        state_dict = {}
        idx = 0

        # Get device/dtype from template
        ref_param = next(self.model_template.parameters())
        device = ref_param.device
        dtype = ref_param.dtype

        for name in self.param_names:
            shape = self.param_shapes[name]
            count = int(np.prod(shape))

            # Slice batch: (Pop_Size, count)
            param_batch_flat = population[:, idx : idx + count]
            idx += count

            # Reshape to (Pop_Size, *shape)
            batched_shape = (pop_size,) + shape

            # Convert to tensor
            tensor_batch = torch.from_numpy(param_batch_flat).to(device=device, dtype=dtype)
            state_dict[name] = tensor_batch.reshape(batched_shape)

        return state_dict
