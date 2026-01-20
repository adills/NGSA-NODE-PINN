import pytest
import torch
import torch.nn as nn
import numpy as np
from src.nsga_neuro_evolution_core.interface import PytorchGenomeInterface

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def test_flatten_unflatten_consistency():
    model = SimpleModel()
    interface = PytorchGenomeInterface(model)

    # Get initial genome
    genome_orig = interface.to_genome(model)

    # Convert back to state_dict
    state_dict = interface.genome_to_state_dict(genome_orig)

    # Load into new model
    model2 = SimpleModel()
    model2.load_state_dict(state_dict)
    genome_new = interface.to_genome(model2)

    np.testing.assert_allclose(genome_orig, genome_new, atol=1e-6)

def test_single_genome_unflatten():
    model = SimpleModel()
    interface = PytorchGenomeInterface(model)

    genome = interface.to_genome(model)
    state_dict = interface.genome_to_state_dict(genome)

    for name, param in state_dict.items():
        # Should not have batch dim
        assert param.ndim == interface.param_shapes[name].numel() or len(interface.param_shapes[name]) == param.ndim
        # Actually param.ndim should equal len(shape)
        assert param.shape == interface.param_shapes[name]

def test_batch_unflatten_shapes():
    model = SimpleModel()
    interface = PytorchGenomeInterface(model)

    pop_size = 10
    total_params = interface.total_params

    # Random population
    population = np.random.randn(pop_size, total_params).astype(np.float32)

    batched_state_dict = interface.batch_to_state_dict(population)

    for name, tensor in batched_state_dict.items():
        expected_shape = (pop_size,) + interface.param_shapes[name]
        assert tensor.shape == expected_shape
        assert tensor.device == next(model.parameters()).device
