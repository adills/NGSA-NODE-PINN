import torch
import torch.nn as nn
import numpy as np
import pytest
from src.nsga_node.evaluator import VectorizedNodeEvaluator
from src.nsga_neuro_evolution_core.interface import PytorchGenomeInterface
from torch.func import functional_call

class SimpleODE(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        # Initialize identity
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(2))
            self.linear.bias.zero_()

    def forward(self, t, y):
        return self.linear(y)

def correction_dummy(model, params, t, y):
    # Just return norm of y as dummy correction
    # y is (D,)
    return torch.norm(y)

def test_evaluator_modes():
    # Setup
    model = SimpleODE()
    interface = PytorchGenomeInterface(model)
    t_eval = torch.linspace(0, 1, 10)
    y0 = torch.tensor([1.0, 0.0])
    target_data = torch.zeros(10, 2) # Dummy target

    evaluator = VectorizedNodeEvaluator(
        model, interface, t_eval, y0, target_data,
        correction_fn=correction_dummy
    )

    # Population
    pop_size = 3
    genome = interface.to_genome(model)
    population = np.tile(genome, (pop_size, 1))

    # 1. Fitness Mode
    losses_fitness = evaluator.evaluate_population(population, mode='fitness')
    assert losses_fitness.shape == (pop_size, 2)
    assert not isinstance(losses_fitness, torch.Tensor) # Should be numpy

    # 2. Gradient Mode
    # Evaluate single individual (pop size 1 for simplicity in test check)
    losses_grad = evaluator.evaluate_population(population[:1], mode='gradient')
    assert isinstance(losses_grad, torch.Tensor)
    assert losses_grad.shape == (1, 2)
    assert losses_grad.requires_grad or losses_grad.grad_fn is not None

    # 3. Evaluate Module
    loss_mod = evaluator.evaluate_module(model)
    assert isinstance(loss_mod, torch.Tensor)
    assert loss_mod.shape == (2,)
    assert loss_mod.requires_grad or loss_mod.grad_fn is not None

def test_evaluator_no_correction():
    # Setup
    model = SimpleODE()
    interface = PytorchGenomeInterface(model)
    t_eval = torch.linspace(0, 1, 5)
    y0 = torch.tensor([1.0, 0.0])
    target_data = torch.zeros(5, 2)

    evaluator = VectorizedNodeEvaluator(
        model, interface, t_eval, y0, target_data,
        correction_fn=None
    )

    loss_mod = evaluator.evaluate_module(model)
    assert loss_mod[1] == 0.0
