import torch
import torch.nn as nn
from src.nsga_node.dynamics import NodeDynamicsWrapper
from src.nsga_neuro_evolution_core.interface import PytorchGenomeInterface

class SimpleODE(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, t, y):
        # f(t, y) = A*y
        return self.linear(y)

def test_dynamics_batched_state_dict_input():
    # 1. Setup
    model = SimpleODE()
    interface = PytorchGenomeInterface(model)
    wrapper = NodeDynamicsWrapper(model)

    pop_size = 5
    state_dim = 2

    # 2. Create population
    # 5 random genomes
    population_genomes = []
    for _ in range(pop_size):
        genome = interface.to_genome(model) # Just copy initial
        # Mutate slightly to make them different
        genome += np.random.randn(*genome.shape) * 0.1
        population_genomes.append(genome)

    population = np.vstack(population_genomes) # (5, N_params)

    # Batch params
    batched_state_dict = interface.batch_to_state_dict(population)

    # 3. Inputs
    t = torch.tensor(0.5)
    y = torch.randn(pop_size, state_dim)

    # 4. Forward
    dydt = wrapper(t, y, batched_state_dict)

    # 5. Verify shape
    assert dydt.shape == (pop_size, state_dim)

    # 6. Verify values
    # Manually compute for first individual
    ind_0_params = interface.genome_to_state_dict(population[0])
    ind_0_y = y[0:1] # (1, 2)

    # Run model normally with functional_call for ind 0
    from torch.func import functional_call
    expected_0 = functional_call(model, ind_0_params, (t, ind_0_y))

    assert torch.allclose(dydt[0], expected_0[0], atol=1e-5)

import numpy as np
